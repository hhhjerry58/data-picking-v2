"""Frozen loss-drop labels, resumable upstream fastText, and candidate scoring."""
from __future__ import annotations

import concurrent.futures
import math
import multiprocessing
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, f1_score

from . import _fasttext_native
from .common import config, paths, read_json, write_json, file_hash, digest, complete_file, verified_file, rank_key, emit, utc_now
from .corpus_io import assignment_table, prepared_shard


def classifier_text(text):
    # Prevent web content from being parsed as extra supervised labels.
    return text.replace("\r", " ").replace("\n", " ").replace("__label__", "__escaped_label__")


def train_threshold(frame, score_name, fraction, tie_seed):
    train = frame[frame["partition"] == "label_train"].copy()
    train["tie_key"] = train["id"].map(lambda value: rank_key(value, tie_seed))
    train = train.sort_values([score_name, "tie_key"], ascending=[False, True])
    cumulative = train["pythia_tokens"].to_numpy(dtype=np.int64).cumsum()
    target = math.ceil(int(cumulative[-1]) * fraction)
    index = int(np.searchsorted(cumulative, target, side="left"))
    boundary = train.iloc[index]
    return {"score": score_name, "threshold": float(boundary[score_name]), "tie_key": boundary["tie_key"],
            "include_documents": index+1, "include_tokens": int(cumulative[index]), "total_tokens": int(cumulative[-1]),
            "target_fraction": fraction, "actual_fraction": float(cumulative[index]/cumulative[-1]), "tie_seed": tie_seed}


def apply_threshold(frame, threshold):
    scores = frame[threshold["score"]].to_numpy()
    keys = frame["id"].map(lambda value: rank_key(value, threshold["tie_seed"])).to_numpy()
    return (scores > threshold["threshold"]) | ((scores == threshold["threshold"]) & (keys <= threshold["tie_key"]))


def make_training_data(cfg=None):
    cfg = cfg or config()
    p = paths(cfg)
    scores_path = p["run"] / "label_scores.parquet"
    if not verified_file(scores_path):
        raise RuntimeError("Complete paired teacher scores are required")
    frame = pd.read_parquet(scores_path)
    text = pq.read_table(p["data"] / "label_documents.parquet", columns=["id", "text"]).to_pandas()
    frame = frame.merge(text, on="id", validate="one_to_one")
    frame["training_order"] = frame["id"].map(lambda value: rank_key(value, cfg["fasttext"]["seed"]))
    frame = frame.sort_values("training_order")
    for metric in cfg["fasttext"]["metrics"]:
        folder = p["run"] / "fasttext" / metric
        folder.mkdir(parents=True, exist_ok=True)
        threshold = train_threshold(frame, metric, cfg["fasttext"]["include_token_fraction"], cfg["fasttext"]["tie_seed"])
        labels = apply_threshold(frame, threshold)
        labelled = frame.copy()
        labelled["include"] = labels
        train = labelled[labelled["partition"] == "label_train"]
        fingerprint = digest({"scores_sha256": file_hash(scores_path), "threshold": threshold, "fasttext": cfg["fasttext"],
                              "text_transform": "CR/LF to space, escape __label__ in body, v1"})
        path = folder / "train.txt"
        if verified_file(path):
            if read_json(str(path) + ".complete.json")["fingerprint"] != fingerprint:
                raise RuntimeError("Existing classifier labels changed")
        else:
            with path.open("w", encoding="utf-8", buffering=8*1024**2) as output:
                for row in train.itertuples():
                    label = "include" if row.include else "exclude"
                    output.write(f"__label__{label} {classifier_text(row.text)}\n")
            complete_file(path, fingerprint=fingerprint)
        if len(set(train["include"])) != 2:
            raise RuntimeError("Both supervised classes are required")
        labelled.drop(columns=["text", "training_order"]).to_parquet(folder / "labels.parquet", index=False)
        write_json(folder / "threshold.json", threshold)
        write_json(folder / "label_summary.json", {"fingerprint": fingerprint, "threshold": threshold,
                   "documents_by_split_label": {str(key): int(len(part)) for key, part in labelled.groupby(["partition", "include"])},
                   "positive_delta_fraction": float((labelled["delta_loss"] > 0).mean()),
                   "escaped_label_prefix_documents": int(frame["text"].str.contains("__label__", regex=False).sum())})
        emit("fasttext_labels_ready", metric=metric, training_documents=len(train), include_documents=int(train["include"].sum()))


def argv_for_training(cfg, train_path, output_path):
    s = cfg["fasttext"]
    argv = ["fasttext", "supervised", "-input", str(train_path), "-output", str(output_path)]
    for key in ("dim", "epoch", "lr", "wordNgrams", "minCount", "bucket", "minn", "maxn", "loss", "thread", "seed"):
        argv.extend(["-" + key, str(s[key])])
    argv.extend(["-verbose", "0"])
    return argv


def checkpoint_records(folder, fingerprint):
    result = []
    for checkpoint in sorted((folder / "checkpoints").glob("epoch-*")):
        marker = checkpoint / "COMPLETED.json"
        if not marker.exists():
            continue
        manifest = read_json(checkpoint / "manifest.json")
        if manifest["fingerprint"] != fingerprint or read_json(marker)["manifest_hash"] != digest(manifest):
            raise RuntimeError("Classifier checkpoint identity mismatch")
        for name, metadata in manifest["files"].items():
            path = checkpoint / name
            if path.stat().st_size != metadata["size"] or file_hash(path) != metadata["sha256"]:
                raise RuntimeError("Corrupt classifier checkpoint")
        result.append((checkpoint, manifest))
    return result


def train_classifier(metric, cfg=None, stop_after_epoch=None):
    cfg = cfg or config()
    p, spec = paths(cfg), cfg["fasttext"]
    if not spec["save_optimizer"] or not spec["auto_resume"]:
        raise ValueError("Full SGD state and auto-resume are required")
    folder = p["run"] / "fasttext" / metric
    train_path = folder / "train.txt"
    fingerprint = digest({"training_sha256": file_hash(train_path), "labels": read_json(folder / "label_summary.json")["fingerprint"],
                          "spec": spec, "native_extension_sha256": file_hash(_fasttext_native.__file__),
                          "adapter_sha256": file_hash(__file__)})
    effective = {"fingerprint": fingerprint, "parameters": spec, "training_data": str(train_path),
                 "training_sha256": file_hash(train_path), "save_optimizer": True, "auto_resume": True,
                 "optimizer": "upstream single-thread SGD; weights, LR progress, committed/pending token counters, RNG, stream cursor persisted"}
    if (folder / "effective_config.json").exists() and read_json(folder / "effective_config.json")["fingerprint"] != fingerprint:
        raise RuntimeError("Classifier training identity changed")
    write_json(folder / "effective_config.json", effective)
    checkpoints = checkpoint_records(folder, fingerprint)
    latest, manifest = checkpoints[-1] if checkpoints else (None, None)
    model = _fasttext_native.Classifier()
    args = argv_for_training(cfg, train_path, folder / "model")
    if latest:
        model.initialize(args, str(latest / "model.bin"), str(latest / "trainer.state"))
        if model.status() != manifest["trainer"]:
            raise RuntimeError("Restored SGD state differs from saved state")
        emit("fasttext_resumed", metric=metric, **model.status())
    else:
        model.initialize(args)
    start_epoch = int(manifest["epoch"]) if manifest else 0
    for epoch in range(start_epoch + 1, spec["epoch"] + 1):
        begin = time.monotonic()
        model.advance(epoch)
        checkpoint = folder / "checkpoints" / f"epoch-{epoch:02d}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save(str(checkpoint / "model.bin"), str(checkpoint / "trainer.state"))
        files = {name: {"size": (checkpoint / name).stat().st_size, "sha256": file_hash(checkpoint / name)}
                 for name in ("model.bin", "trainer.state")}
        manifest = {"fingerprint": fingerprint, "epoch": epoch, "trainer": model.status(), "files": files,
                    "save_optimizer": True, "auto_resume": True, "optimizer": effective["optimizer"]}
        write_json(checkpoint / "manifest.json", manifest)
        write_json(checkpoint / "COMPLETED.json", {"manifest_hash": digest(manifest), "time": utc_now()})
        latest = checkpoint
        emit("fasttext_epoch_complete", metric=metric, epoch=epoch, seconds=round(time.monotonic()-begin, 2), **model.status())
        if stop_after_epoch and epoch >= stop_after_epoch:
            return
    if not model.status()["finished"]:
        raise RuntimeError("Classifier did not finish its prescribed token schedule")
    final = folder / "model.bin"
    shutil.copyfile(latest / "model.bin", final)
    complete_file(final, fingerprint=fingerprint)
    write_json(folder / "TRAINING_COMPLETED.json", {"fingerprint": fingerprint, "checkpoint": str(latest), "trainer": model.status(), "time": utc_now()})
    evaluate_classifier(metric, cfg, model)


def top_token_mask(values, weights, ids, fraction, tie_seed):
    order = sorted(range(len(values)), key=lambda i: (-float(values[i]), rank_key(ids[i], tie_seed)))
    cumulative = np.cumsum(np.asarray(weights, dtype=np.int64)[order])
    stop = int(np.searchsorted(cumulative, math.ceil(int(cumulative[-1])*fraction), side="left")) + 1
    mask = np.zeros(len(values), dtype=bool)
    mask[np.asarray(order[:stop])] = True
    return mask


def evaluate_classifier(metric, cfg=None, model=None):
    cfg = cfg or config()
    p = paths(cfg)
    folder = p["run"] / "fasttext" / metric
    if model is None:
        model = _fasttext_native.Classifier()
        model.load_model(str(folder / "model.bin"))
    labels = pd.read_parquet(folder / "labels.parquet")
    labels = labels[labels["partition"].isin(["label_dev", "label_test"])]
    texts = pq.read_table(p["data"] / "label_documents.parquet", columns=["id", "text"]).to_pandas()
    frame = labels.merge(texts, on="id", validate="one_to_one")
    probabilities = []
    for start in range(0, len(frame), 256):
        probabilities.extend(model.predict_include([classifier_text(t) for t in frame["text"].iloc[start:start+256]]))
    frame["probability"] = probabilities
    summaries = {}
    for split, group in frame.groupby("partition"):
        truth = group["include"].to_numpy(bool)
        probability = group["probability"].to_numpy()
        weights = group["pythia_tokens"].to_numpy(np.int64)
        selected = top_token_mask(probability, weights, group["id"].tolist(), 0.2, cfg["fasttext"]["tie_seed"])
        overlap = selected & truth
        summaries[split] = {"documents": len(group), "include_documents": int(truth.sum()),
            "average_precision": float(average_precision_score(truth, probability)) if truth.any() else None,
            "f1_at_half": float(f1_score(truth, probability >= 0.5, zero_division=0)),
            "top_token_precision": float(weights[overlap].sum()/max(1, weights[selected].sum())),
            "top_token_recall": float(weights[overlap].sum()/max(1, weights[truth].sum())),
            "top_document_precision": float(overlap.sum()/max(1, selected.sum())),
            "top_document_recall": float(overlap.sum()/max(1, truth.sum())),
            "actual_selected_token_fraction": float(weights[selected].sum()/weights.sum())}
    frame.drop(columns="text").to_parquet(folder / "heldout_predictions.parquet", index=False)
    write_json(folder / "heldout_metrics.json", summaries)
    emit("fasttext_heldout_evaluated", metric=metric, metrics=summaries)


_SCAN = {}


def initialize_scan(cfg):
    p = paths(cfg)
    table = assignment_table(cfg, ["candidate"]).select(["id", "source_index"])
    by_shard = {}
    for index, ids in table.to_pandas().groupby("source_index")["id"]:
        by_shard[int(index)] = set(ids)
    models, fingerprints = {}, {}
    for metric in cfg["fasttext"]["metrics"]:
        file = p["run"] / "fasttext" / metric / "model.bin"
        if not verified_file(file):
            raise RuntimeError("Final classifier is missing")
        models[metric] = _fasttext_native.Classifier()
        models[metric].load_model(str(file))
        fingerprints[metric] = read_json(str(file) + ".complete.json")["sha256"]
    _SCAN.update(cfg=cfg, p=p, ids=by_shard, models=models,
                 fingerprint=digest({"assignments": file_hash(p["data"] / "assignments.parquet"), "models": fingerprints, "implementation": file_hash(__file__)}))


def scan_shard(index):
    s = _SCAN
    folder = s["p"]["run"] / "candidate_scores"
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / f"part-{index:05d}.parquet"
    if verified_file(file):
        if read_json(str(file) + ".complete.json")["fingerprint"] != s["fingerprint"]:
            raise RuntimeError("Candidate scorer changed during resume")
        return {"index": index, "documents": pq.ParquetFile(file).metadata.num_rows}
    columns = ["id", "text", "source_index", "source_row", "domain", "crawl", "text_hash", "sample_key", "pythia_tokens"]
    expected = s["ids"][index]
    rows_out = []
    begin = time.monotonic()
    for batch in pq.ParquetFile(prepared_shard(s["cfg"], index)).iter_batches(batch_size=512, columns=columns):
        rows = [r for r in batch.to_pylist() if r["id"] in expected]
        if not rows:
            continue
        texts = [classifier_text(row["text"]) for row in rows]
        predictions = {metric: model.predict_include(texts) for metric, model in s["models"].items()}
        for i, row in enumerate(rows):
            row.pop("text")
            for metric, values in predictions.items():
                value = values[i]
                if not math.isfinite(value) or not 0 <= value <= 1.0001:
                    raise RuntimeError("Invalid native fastText probability")
                row["p_" + metric] = value
            rows_out.append(row)
    if len(rows_out) != len(expected):
        raise RuntimeError("Candidate shard coverage mismatch")
    output_schema = pa.schema([("id",pa.string()),("source_index",pa.int32()),("source_row",pa.int32()),
                               ("domain",pa.string()),("crawl",pa.string()),("text_hash",pa.string()),
                               ("sample_key",pa.uint64()),("pythia_tokens",pa.int64()),
                               ("p_delta_loss",pa.float64()),("p_relative_gain",pa.float64())])
    pq.write_table(pa.Table.from_pylist(rows_out,schema=output_schema), file, compression="zstd")
    complete_file(file, fingerprint=s["fingerprint"])
    result = {"index": index, "documents": len(rows_out), "seconds": round(time.monotonic()-begin, 2)}
    emit("candidate_shard_scored", **result)
    return result


def scan_candidates(cfg=None, workers=16):
    cfg = cfg or config()
    p = paths(cfg)
    table = assignment_table(cfg, ["candidate"])
    indexes = sorted(set(table["source_index"].to_pylist()))
    expected = len(table)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                               initializer=initialize_scan, initargs=(cfg,)) as pool:
        results = list(pool.map(scan_shard, indexes))
    if sum(r["documents"] for r in results) != expected:
        raise RuntimeError("Candidate score coverage mismatch")
    write_json(p["run"] / "candidate_scores/COMPLETED.json", {"documents": expected, "shards": results, "time": utc_now()})
    emit("candidate_scan_complete", documents=expected)
