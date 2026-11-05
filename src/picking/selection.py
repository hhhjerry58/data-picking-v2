"""Select equal useful-token budgets and retain per-document token stores."""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .common import config, paths, read_json, write_json, file_hash, digest, rank_key, complete_file, verified_file, emit, utc_now
from .corpus_io import assignment_table, prepared_shard


def input_tokens_for_targets(target_tokens, sequence_length):
    sequences = math.ceil(target_tokens / (sequence_length-1))
    return target_tokens + sequences


def select(cfg=None):
    cfg = cfg or config()
    p = paths(cfg)
    complete = p["run"] / "selection/COMPLETED.json"
    if complete.exists():
        for path, sha in read_json(complete)["files"].items():
            if file_hash(path) != sha:
                raise RuntimeError("Selected document set changed")
        return
    score_folder = p["run"] / "candidate_scores"
    coverage = read_json(score_folder / "COMPLETED.json")
    files = sorted(score_folder.glob("part-*.parquet"))
    if any(not verified_file(f) for f in files):
        raise RuntimeError("Incomplete candidate scores")
    frame = pa.concat_tables([pq.read_table(file) for file in files]).to_pandas()
    if len(frame) != coverage["documents"] or frame["id"].duplicated().any():
        raise RuntimeError("Candidate scoring coverage mismatch")
    frame["tie_key"] = frame["id"].map(lambda value: rank_key(value, cfg["fasttext"]["tie_seed"]))
    frame["random_key"] = frame["id"].map(lambda value: rank_key("random:" + value, cfg["source"]["data_seed"]))
    target = input_tokens_for_targets(cfg["pretraining"]["total_prediction_tokens"], cfg["pretraining"]["sequence_length"])
    summaries, outputs = {}, {}
    sets = {}
    for arm in cfg["pretraining"]["arms"]:
        ranked = frame.sort_values("random_key") if arm == "random" else frame.sort_values(["p_" + arm, "tie_key"], ascending=[False, True])
        cumulative = ranked["pythia_tokens"].to_numpy(np.int64).cumsum()
        if cumulative[-1] < target:
            raise RuntimeError("Insufficient unique input tokens")
        count = int(np.searchsorted(cumulative, target, side="left")) + 1
        chosen = ranked.iloc[:count].copy()
        chosen["take_tokens"] = chosen["pythia_tokens"].astype(np.int64)
        chosen.iloc[-1, chosen.columns.get_loc("take_tokens")] -= int(cumulative[count-1])-target
        chosen["selection_rank"] = np.arange(count, dtype=np.int64)
        if chosen["take_tokens"].sum() != target or chosen["id"].duplicated().any():
            raise RuntimeError("Selected token budget or uniqueness failure")
        folder = p["run"] / "selection" / arm
        folder.mkdir(parents=True, exist_ok=True)
        file = folder / "documents.parquet"
        chosen.drop(columns=["random_key", "tie_key"]).to_parquet(file, index=False)
        outputs[str(file)] = file_hash(file)
        summaries[arm] = {"documents": len(chosen), "input_tokens": target,
                          "prediction_tokens": cfg["pretraining"]["total_prediction_tokens"],
                          "boundary_clipped_tokens": int(cumulative[count-1])-target,
                          "token_length_quantiles": chosen["pythia_tokens"].quantile([0, .25, .5, .75, .9, .99, 1]).to_dict(),
                          "top_domains_by_tokens": chosen.groupby("domain")["take_tokens"].sum().nlargest(20).to_dict()}
        sets[arm] = set(chosen["id"])
        emit("selection_complete", arm=arm, documents=len(chosen), input_tokens=target)
    overlaps = {f"{a}/{b}": {"documents": len(sets[a] & sets[b]), "jaccard": len(sets[a] & sets[b])/len(sets[a] | sets[b])}
                for i, a in enumerate(sets) for b in list(sets)[i+1:]}
    write_json(p["run"] / "selection/summary.json", {"arms": summaries, "pairwise_overlap": overlaps,
               "input_tokens_formula": "D + ceil(D/(context_length-1)); packing counts every target after each block's first token",
               "random_order": "BLAKE2b(data_seed, 'random:' + document_id)"})
    write_json(complete, {"files": outputs, "time": utc_now()})


def build_token_stores(cfg=None):
    cfg = cfg or config()
    p = paths(cfg)
    arms = cfg["pretraining"]["arms"]
    pending = []
    for arm in arms:
        folder = p["run"] / "selection" / arm
        if not (folder / "TOKEN_STORE_COMPLETED.json").exists():
            pending.append(arm)
    if not pending:
        return
    wanted, outputs, offsets, records, seen = {}, {}, {}, {}, {}
    for arm in pending:
        folder = p["run"] / "selection" / arm
        selected = pd.read_parquet(folder / "documents.parquet")
        wanted[arm] = dict(zip(selected["id"], selected["take_tokens"].astype(int)))
        outputs[arm] = (folder / "tokens.bin").open("wb", buffering=16*1024**2)
        offsets[arm] = 0
        records[arm] = []
        seen[arm] = set()
    shards = sorted(set(assignment_table(cfg, ["candidate"])["source_index"].to_pylist()))
    try:
        for index in shards:
            path = prepared_shard(cfg, index)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["id", "input_ids", "pythia_tokens"]):
                for row in batch.to_pylist():
                    interested = [arm for arm in pending if row["id"] in wanted[arm]]
                    if not interested:
                        continue
                    ids = np.asarray(row["input_ids"] + [0], dtype=np.uint32)  # Pinned Pythia EOS.
                    if len(ids) != row["pythia_tokens"]:
                        raise RuntimeError("Source token count differs from selected count")
                    for arm in interested:
                        if row["id"] in seen[arm]:
                            raise RuntimeError("Repeated selected document")
                        length = wanted[arm][row["id"]]
                        if not 0 < length <= len(ids):
                            raise RuntimeError("Invalid boundary truncation")
                        outputs[arm].write(ids[:length].tobytes())
                        records[arm].append({"id": row["id"], "offset": offsets[arm], "length": length})
                        offsets[arm] += length
                        seen[arm].add(row["id"])
            emit("token_store_progress", source_index=index, written_tokens=offsets)
    finally:
        for output in outputs.values():
            output.close()
    for arm in pending:
        folder = p["run"] / "selection" / arm
        expected = sum(wanted[arm].values())
        if offsets[arm] != expected or seen[arm] != set(wanted[arm]):
            raise RuntimeError("Selected token-store coverage mismatch")
        pq.write_table(pa.Table.from_pylist(records[arm]), folder / "token_offsets.parquet", compression="zstd")
        meta = complete_file(folder / "tokens.bin", tokens=expected, dtype="uint32")
        write_json(folder / "TOKEN_STORE_COMPLETED.json", {"tokens": expected, "documents": len(seen[arm]),
                   "tokens_sha256": meta["sha256"], "offsets_sha256": file_hash(folder / "token_offsets.parquet"),
                   "selection_sha256": file_hash(folder / "documents.parquet"), "time": utc_now()})


def pack(arm, seed, cfg=None):
    cfg = cfg or config()
    p = paths(cfg)
    folder = p["run"] / "selection" / arm
    store = read_json(folder / "TOKEN_STORE_COMPLETED.json")
    destination = folder / f"packed-seed-{seed}.bin"
    identity = {"store": store, "seed": seed, "permutation": "numpy PCG64 over document IDs sorted lexicographically",
                "implementation": file_hash(__file__)}
    fingerprint = digest(identity)
    if verified_file(destination):
        if read_json(str(destination) + ".complete.json")["fingerprint"] != fingerprint:
            raise RuntimeError("Packed input identity changed")
        return destination
    source = folder / "tokens.bin"
    if file_hash(source) != store["tokens_sha256"] or file_hash(folder / "token_offsets.parquet") != store["offsets_sha256"]:
        raise RuntimeError("Token store modified before packing")
    rows = pq.read_table(folder / "token_offsets.parquet").to_pandas().sort_values("id").reset_index(drop=True)
    order = np.random.default_rng(seed).permutation(len(rows))
    offsets = rows["offset"].to_numpy(np.int64)
    lengths = rows["length"].to_numpy(np.int64)
    token_map = np.memmap(source, mode="r", dtype=np.uint32)
    begin = time.monotonic()
    with destination.open("wb", buffering=32*1024**2) as output:
        for index in order:
            start, length = int(offsets[index]), int(lengths[index])
            output.write(token_map[start:start+length].tobytes())
    if destination.stat().st_size != store["tokens"]*4:
        raise RuntimeError("Packing changed token budget")
    complete_file(destination, fingerprint=fingerprint, identity=identity, tokens=store["tokens"], dtype="uint32")
    emit("training_data_packed", arm=arm, seed=seed, tokens=store["tokens"], seconds=round(time.monotonic()-begin, 2))
    return destination
