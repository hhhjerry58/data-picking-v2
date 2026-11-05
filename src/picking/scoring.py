"""Resume exact FP32 per-window teacher NLL and aggregate paired document losses."""
from __future__ import annotations

import gc
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import config, paths, read_json, write_json, file_hash, digest, rank_key, complete_file, verified_file, emit, seed_all, utc_now
from .corpus_io import batches, assignment_table

SCORING_VERSION = "fp32_sorted_short_single_v2"
SHORT_WINDOW_THRESHOLD = 128


def document_windows(ids, max_length=2048):
    if len(ids) < 2:
        raise ValueError("A document needs at least two Qwen tokens")
    for index, start in enumerate(range(0, len(ids)-1, max_length-1)):
        yield index, start, ids[start:start+max_length]


def per_example_nll(logits, input_ids, mask, chunk_size=128):
    target_mask = mask[:, 1:]
    counts = target_mask.sum(1)
    if (counts <= 0).any():
        raise ValueError("No valid prediction targets")
    positions = target_mask.nonzero(as_tuple=False)
    sums = torch.zeros(logits.shape[0], dtype=torch.float32, device=logits.device)
    for part in positions.split(chunk_size):
        batch, time_index = part[:, 0], part[:, 1]
        values = F.cross_entropy(logits[batch, time_index].float(), input_ids[batch, time_index+1], reduction="none")
        sums = sums.index_add(0, batch, values)
    return sums, counts


def prepare_label_cache(cfg=None):
    cfg = cfg or config()
    p = paths(cfg)
    destination = p["data"] / "label_documents.parquet"
    expected = assignment_table(cfg, ["label_train", "label_dev", "label_test"])
    fingerprint = digest({"assignment": file_hash(p["data"] / "assignments.parquet"), "teacher": cfg["teacher"],
                          "label_validity": "reject fewer than two Qwen tokens; replace from lowest-hash disjoint unused reserve, v1",
                          "tokenizer": file_hash(Path(cfg["teacher"]["runtime_model"]) / "tokenizer.json")})
    if verified_file(destination):
        if read_json(str(destination) + ".complete.json")["fingerprint"] != fingerprint:
            raise RuntimeError("Label token cache identity changed")
        return
    tokenizer = Tokenizer.from_file(str(Path(cfg["teacher"]["runtime_model"]) / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    schema = pa.schema([("id", pa.string()), ("partition", pa.string()), ("domain", pa.string()), ("crawl", pa.string()),
                        ("text", pa.large_string()), ("pythia_tokens", pa.int64()), ("qwen_tokens", pa.int64()),
                        ("input_ids", pa.large_list(pa.uint32())), ("tokenization_hash", pa.string())])
    total, tokens = 0, 0
    unscorable, replacements, label_manifest = [], [], []
    writer = pq.ParquetWriter(destination, schema, compression="zstd", compression_level=3)
    try:
        for source_rows in batches(cfg, ["label_train", "label_dev", "label_test"], columns=["id", "domain", "crawl", "text", "pythia_tokens"], batch_size=1024):
            encodings = tokenizer.encode_batch([row["text"] for row in source_rows], add_special_tokens=False)
            rows = []
            for row, encoding in zip(source_rows, encodings):
                ids = encoding.ids
                if len(ids) < 2:
                    unscorable.append({"id": row["id"], "qwen_tokens": len(ids), "reason": "fewer_than_two_qwen_tokens"})
                    continue
                rows.append({**row, "input_ids": ids, "qwen_tokens": len(ids), "tokenization_hash": digest(ids)})
                label_manifest.append({"id": row["id"], "partition": row["partition"], "domain": row["domain"]})
                total += 1
                tokens += len(ids)-1
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        if unscorable:
            reserve = assignment_table(cfg, ["unused"]).sort_by([("sample_key", "ascending")])
            reserve = reserve.slice(0, max(1024, len(unscorable)*4)).select(["id", "source_index", "sample_key"])
            reserve_rows = reserve.to_pylist()
            wanted = {row["id"] for row in reserve_rows}
            keys = {row["id"]: row["sample_key"] for row in reserve_rows}
            sources = set(row["source_index"] for row in reserve_rows)
            candidates = []
            for source_rows in batches(cfg, ["unused"], columns=["id", "domain", "crawl", "text", "pythia_tokens"], source_indexes=sources):
                candidates.extend(row for row in source_rows if row["id"] in wanted)
            candidates.sort(key=lambda row: keys[row["id"]])
            needed = len(unscorable)
            for row in candidates:
                ids = tokenizer.encode(row["text"], add_special_tokens=False).ids
                if len(ids) < 2:
                    continue
                domain_key = int(rank_key(row["domain"], cfg["source"]["data_seed"])[:16], 16)/2**64
                row["partition"] = "label_train" if domain_key < 0.8 else ("label_dev" if domain_key < 0.9 else "label_test")
                enriched = {**row, "input_ids": ids, "qwen_tokens": len(ids), "tokenization_hash": digest(ids)}
                writer.write_table(pa.Table.from_pylist([enriched], schema=schema))
                label_manifest.append({"id": row["id"], "partition": row["partition"], "domain": row["domain"]})
                replacements.append({"id": row["id"], "partition": row["partition"], "qwen_tokens": len(ids)})
                total += 1
                tokens += len(ids)-1
                if len(replacements) == needed:
                    break
            if len(replacements) != needed:
                raise RuntimeError("Not enough valid disjoint reserve documents to restore 100,000 labels")
    finally:
        writer.close()
    if total != cfg["source"]["label_documents"] or total != len(expected):
        raise RuntimeError("Label document coverage mismatch")
    if len({row["id"] for row in label_manifest}) != total:
        raise RuntimeError("Duplicate final label document")
    pq.write_table(pa.Table.from_pylist(label_manifest), p["data"] / "final_label_manifest.parquet", compression="zstd")
    write_json(p["data"] / "label_validity.json", {"unscorable": unscorable, "replacements": replacements,
               "rule": "Basic tokenizer validity only, before any teacher forward; reserve documents are disjoint from candidate/general-evaluation pools",
               "actual_partition_counts": {part: sum(r["partition"] == part for r in label_manifest) for part in ("label_train", "label_dev", "label_test")}})
    complete_file(destination, fingerprint=fingerprint, documents=total, scored_tokens=tokens)
    emit("label_cache_complete", documents=total, scored_tokens=tokens)


def verify_teacher_assets(cfg):
    teacher = cfg["teacher"]
    runtime = Path(teacher["runtime_model"])
    persistent = Path(teacher["persistent_model"])
    record = {}
    for name in ["config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "model.safetensors"]:
        meta = read_json(persistent / (name + ".complete.json"))
        path = runtime / name
        if not path.exists() or path.stat().st_size != meta["size"] or file_hash(path) != meta["sha256"]:
            raise RuntimeError(f"Teacher runtime asset mismatch: {name}")
        record[name] = meta["sha256"]
    checkpoint = Path(teacher["checkpoint"])
    manifest = read_json(checkpoint / "manifest.json")
    complete = read_json(checkpoint / "COMPLETED.json")
    if complete["manifest_hash"] != digest(manifest) or manifest["global_step"] != teacher["sft_step"]:
        raise RuntimeError("Teacher checkpoint completion/step mismatch")
    state = Path(teacher["runtime_checkpoint"])
    expected = manifest["files"]["state.pt"]
    if not state.exists() or state.stat().st_size != expected["size"] or file_hash(state) != expected["sha256"]:
        raise RuntimeError("Staged SFT state does not match the persistent checkpoint")
    record["sft_checkpoint_sha256"] = expected["sha256"]
    write_json(paths(cfg)["run"] / "teacher_assets_verified.json", {"files": record, "verified_at_utc": utc_now()})
    return record


@torch.inference_mode()
def score_batch(model, windows, pad_token_id):
    maximum = max(len(row["ids"]) for row in windows)
    input_ids = torch.full((len(windows), maximum), pad_token_id, dtype=torch.long, device="cuda")
    attention = torch.zeros_like(input_ids)
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for i, row in enumerate(windows):
        length = len(row["ids"])
        input_ids[i, :length] = torch.tensor(row["ids"], dtype=torch.long, device="cuda")
        attention[i, :length] = 1
        mask[i, 1:length] = True
    with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
        output = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
        sums, counts = per_example_nll(output.logits, input_ids, mask)
    sums, counts = sums.tolist(), counts.tolist()
    result = []
    for row, value, count in zip(windows, sums, counts):
        if count != len(row["ids"])-1 or not np.isfinite(value):
            raise RuntimeError("Scoring count/numerical failure")
        result.append({k: row[k] for k in ("id", "window_index", "token_start", "tokenization_hash")} |
                      {"sum_nll": value, "scored_tokens": count})
    return result


def score_with_policy(model, windows, pad_token_id):
    result = []
    position = 0
    while position < len(windows):
        if len(windows[position]["ids"]) < SHORT_WINDOW_THRESHOLD:
            result.extend(score_batch(model, [windows[position]], pad_token_id))
            position += 1
        else:
            end = position
            while end < len(windows) and len(windows[end]["ids"]) >= SHORT_WINDOW_THRESHOLD:
                end += 1
            result.extend(score_batch(model, windows[position:end], pad_token_id))
            position = end
    return result


def sanity(model, windows, pad_token_id, tolerance, batch_size):
    # Validate actual length-sorted batches, including the short-window boundary.
    # Tiny windows always run singly, avoiding extreme artificial right padding.
    starts = {int(i)//batch_size*batch_size for i in np.linspace(0, max(0,len(windows)-batch_size), 6)}
    short = next((i for i,row in enumerate(windows) if len(row["ids"]) < SHORT_WINDOW_THRESHOLD),None)
    if short is not None:
        starts.update(max(0,(short//batch_size+offset)*batch_size) for offset in (-1,0,1) if (short//batch_size+offset)*batch_size<len(windows))
    groups = [windows[start:start+batch_size] for start in sorted(starts)]
    selected = [row for group in groups for row in group]
    batch = [r for group in groups for r in score_with_policy(model, group, pad_token_id)]
    single = [score_batch(model, [row], pad_token_id)[0] for row in selected]
    repeated = [r for group in groups for r in score_with_policy(model, group, pad_token_id)]
    error = max(abs(a["sum_nll"]-b["sum_nll"])/a["scored_tokens"] for a, b in zip(batch, single))
    repeat_error = max(abs(a["sum_nll"]-b["sum_nll"])/a["scored_tokens"] for a, b in zip(batch, repeated))
    if max(error, repeat_error) > tolerance:
        raise RuntimeError(f"FP32 batch/padding invariance failed: {error}, {repeat_error}")
    return {"batch_padding_max_nll_difference": error, "repeat_max_nll_difference": repeat_error, "tolerance": tolerance,
            "policy": SCORING_VERSION, "short_window_threshold": SHORT_WINDOW_THRESHOLD, "windows_checked": len(selected)}


def score_shard(shard_index, num_shards, cfg=None):
    cfg = cfg or config()
    p, spec = paths(cfg), cfg["teacher"]
    if spec["forward_dtype"] != "float32" or spec["attention_backend"] != "math":
        raise ValueError("Formal scoring must use validated FP32/math settings")
    assets = verify_teacher_assets(cfg)
    cache = p["data"] / "label_documents.parquet"
    docs = pq.read_table(cache, columns=["id", "input_ids", "tokenization_hash"]).to_pylist()
    docs = sorted(docs, key=lambda row: row["id"])[shard_index::num_shards]
    windows = []
    for doc in docs:
        for index, start, ids in document_windows(doc["input_ids"], spec["max_length"]):
            windows.append({"id": doc["id"], "window_index": index, "token_start": start, "ids": ids,
                            "tokenization_hash": doc["tokenization_hash"]})
    del docs
    windows.sort(key=lambda row: (-len(row["ids"]), row["id"], row["window_index"]))
    identity = {"assets": assets, "cache": file_hash(cache), "scoring": spec,
                "shard_index": shard_index, "num_shards": num_shards, "implementation": file_hash(__file__)}
    tokenizer = AutoTokenizer.from_pretrained(spec["runtime_model"], local_files_only=True)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    chunk_size = 512
    for label in ("reference", "sft"):
        fingerprint = digest({**identity, "model_label": label})
        folder = p["run"] / "scores" / SCORING_VERSION / label / f"shard-{shard_index:02d}-of-{num_shards:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        existing = folder / "effective_config.json"
        if existing.exists() and read_json(existing)["fingerprint"] != fingerprint:
            raise RuntimeError("Scoring configuration changed during resume")
        write_json(existing, {"fingerprint": fingerprint, "model": label, "shard": shard_index, "windows": len(windows),
                       "forward_dtype": "float32", "attention_backend": "math", "assets": assets,
                       "policy": SCORING_VERSION, "short_window_threshold": SHORT_WINDOW_THRESHOLD})
        pending = []
        for start in range(0, len(windows), chunk_size):
            file = folder / f"chunk-{start:07d}.parquet"
            if not verified_file(file):
                pending.append(start)
        if not pending:
            write_json(folder / "COMPLETED.json", {"fingerprint": fingerprint, "windows": len(windows), "time": utc_now()})
            emit("scores_already_complete", model=label, shard=shard_index)
            continue
        seed_all(42)
        model = AutoModelForCausalLM.from_pretrained(spec["runtime_model"], dtype=torch.float32,
                                                    attn_implementation="sdpa", local_files_only=True).to("cuda").eval()
        model.config.use_cache = False
        if label == "sft":
            payload = torch.load(spec["runtime_checkpoint"], map_location="cpu", mmap=False, weights_only=False)
            if payload["trainer"]["global_step"] != 64 or {int(v["step"]) for v in payload["optimizer"]["state"].values()} != {64}:
                raise RuntimeError("SFT model/optimizer state step mismatch")
            model.load_state_dict(payload["model"])
            del payload
            gc.collect()
        write_json(folder / "sanity.json", sanity(model, windows, pad, spec["sanity_tolerance"], spec["batch_size"]))
        for start in pending:
            begin = time.monotonic()
            chunk = windows[start:start+chunk_size]
            rows = [r for i in range(0, len(chunk), spec["batch_size"]) for r in score_with_policy(model, chunk[i:i+spec["batch_size"]], pad)]
            file = folder / f"chunk-{start:07d}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), file, compression="zstd")
            complete_file(file, fingerprint=fingerprint)
            emit("teacher_score_chunk", model=label, shard=shard_index, windows_done=start+len(rows),
                 total_windows=len(windows), tokens=sum(r["scored_tokens"] for r in rows), seconds=round(time.monotonic()-begin, 2))
        write_json(folder / "COMPLETED.json", {"fingerprint": fingerprint, "windows": len(windows), "time": utc_now()})
        del model
        gc.collect()
        torch.cuda.empty_cache()


def aggregate_scores(cfg=None):
    cfg = cfg or config()
    p = paths(cfg)
    cache = pq.read_table(p["data"] / "label_documents.parquet", columns=["id", "partition", "domain", "pythia_tokens", "qwen_tokens", "tokenization_hash"]).to_pandas()
    windows = {}
    for label in ("reference", "sft"):
        files = sorted((p["run"] / "scores" / SCORING_VERSION / label).glob("shard-*/chunk-*.parquet"))
        if not files or any(not verified_file(file) for file in files):
            raise RuntimeError("Incomplete score chunks")
        frame = pa.concat_tables([pq.read_table(file) for file in files]).to_pandas()
        if frame.duplicated(["id", "window_index"]).any():
            raise RuntimeError("Duplicate scoring windows")
        frame = frame.sort_values(["id", "window_index"]).reset_index(drop=True)
        windows[label] = frame
    identity = ["id", "window_index", "token_start", "tokenization_hash", "scored_tokens"]
    if not windows["reference"][identity].equals(windows["sft"][identity]):
        raise RuntimeError("Reference and SFT scored different tokens")
    for label, frame in windows.items():
        grouped = frame.groupby("id").agg(sum_nll=("sum_nll", "sum"), counts=("scored_tokens", "sum"))
        cache = cache.merge(grouped.rename(columns={"sum_nll": f"{label}_sum_nll", "counts": f"{label}_counts"}), on="id", validate="one_to_one")
    if len(cache) != cfg["source"]["label_documents"]:
        raise RuntimeError("Formal label coverage is not 100,000 documents")
    if not np.all(cache["reference_counts"] == cache["qwen_tokens"]-1) or not np.all(cache["sft_counts"] == cache["reference_counts"]):
        raise RuntimeError("Not every formal label token was scored exactly once")
    cache["ref_loss"] = cache["reference_sum_nll"] / cache["reference_counts"]
    cache["sft_loss"] = cache["sft_sum_nll"] / cache["sft_counts"]
    cache["delta_loss"] = cache["ref_loss"] - cache["sft_loss"]
    cache["relative_gain"] = cache["delta_loss"] / np.maximum(cache["ref_loss"], 1e-6)
    file = p["run"] / "label_scores.parquet"
    cache.to_parquet(file, index=False)
    complete_file(file, documents=len(cache))
    write_json(p["run"] / "label_score_summary.json", {"documents": len(cache), "scored_tokens_per_model": int(cache["reference_counts"].sum()),
               "positive_delta_fraction": float((cache["delta_loss"] > 0).mean()), "denominator_clamps": int((cache["ref_loss"] < 1e-6).sum()),
               "statistics": cache[["ref_loss", "sft_loss", "delta_loss", "relative_gain"]].describe().to_dict()})
    emit("teacher_scores_aggregated", documents=len(cache))
