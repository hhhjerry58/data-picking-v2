"""Corpus coordinator with verified RAM reads; source processing stays frozen."""
from __future__ import annotations

import concurrent.futures
import multiprocessing
import json
import shutil
import time
from pathlib import Path
import pyarrow.parquet as pq

from . import _data_processing_v1 as core
from .common import config, paths, read_json, write_json, file_hash, digest, emit, utc_now
from .corpus_io import prepared_shard
from .finalize_corpus import freeze_assignments

TargetOverlap = core.TargetOverlap
CorpusDedup = core.CorpusDedup
normalize = core.normalize
words = core.words
shingles = core.shingles
prepare_targets = core.prepare_targets


def prepare_corpus(cfg=None):
    cfg = cfg or config()
    p = paths(cfg)
    core.prepare_targets(cfg)
    if (p["data"] / "CORPUS_COMPLETED.json").exists():
        marker = read_json(p["data"] / "CORPUS_COMPLETED.json")
        if file_hash(p["data"] / "assignments.parquet") != marker["assignment_sha256"]:
            raise RuntimeError("Frozen assignments changed")
        emit("corpus_already_complete")
        return
    manifest = read_json(p["raw"] / "source_manifest.json")
    shards = manifest["shards"]
    processor_sha = file_hash(core.__file__)
    target_sha = file_hash(p["data"] / "targets/protected_targets.jsonl")
    scratch = p["scratch"] / "corpus_state"
    scratch.mkdir(parents=True, exist_ok=True)
    dedup = core.CorpusDedup(cfg["source"]["dedup_jaccard"])
    results = []
    columns = ["id", "source_index", "source_row", "text", "domain", "crawl", "text_hash", "sample_key", "pythia_tokens", "minhash_bands"]
    def cached(index):
        source = core.shard_paths(p, shards[index], index)["prepared"]
        stats_path = source.with_suffix(".stats.json")
        if not Path(str(source) + ".complete.json").exists() or not stats_path.exists():
            return None
        record = read_json(stats_path)
        expected = digest({"source": cfg["source"], "shard": shards[index], "targets": target_sha, "implementation": processor_sha})
        if record["fingerprint"] != expected:
            raise RuntimeError(f"Cached source processor identity changed for shard {index}")
        prepared_shard(cfg, index)
        return record
    def ready(value):
        future = concurrent.futures.Future()
        future.set_result(value)
        return future
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=cfg["source"]["workers"],
        mp_context=multiprocessing.get_context("spawn"), initializer=core.initialize_worker, initargs=(cfg,))
    downloaders = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    downloads, processes = {}, {}
    next_download = 0
    def prefetch(current):
        nonlocal next_download
        while next_download < min(len(shards), current + 32):
            index = next_download
            record = cached(index)
            if record is None:
                downloads[index] = downloaders.submit(core.download_shard, p, shards[index], index)
            else:
                downloads[index] = ready(index)
                processes[index] = ready(record)
            next_download += 1
    edges_path = scratch / "dedup_edges.jsonl"
    edges = edges_path.open("w", buffering=8*1024**2)
    begin_all = time.monotonic()
    try:
        prefetch(0)
        for index, shard in enumerate(shards):
            for j, future in list(downloads.items()):
                if future.done() and j not in processes:
                    future.result()
                    processes[j] = executor.submit(core.process_shard, shards[j], j)
            if index not in processes:
                downloads[index].result()
                processes[index] = executor.submit(core.process_shard, shard, index)
            result = processes.pop(index).result()
            del downloads[index]
            results.append(result)
            begin = time.monotonic()
            source = prepared_shard(cfg, index)
            for batch in pq.ParquetFile(source).iter_batches(batch_size=2048, columns=columns):
                for row in batch.to_pylist():
                    edge = dedup.add(row)
                    if edge:
                        edges.write(json.dumps(edge) + "\n")
            edges.flush()
            progress = {"completed_source_shards": index+1, "kept_documents": len(dedup.meta), "kept_tokens": dedup.tokens,
                        "dedup_counts": dict(dedup.counts), "processor_sha256": processor_sha, "time": utc_now()}
            write_json(scratch / "data_progress.json", progress)
            emit("global_dedup_progress", **progress, seconds=round(time.monotonic()-begin, 2))
            if (index+1) % 20 == 0:
                shutil.copyfile(scratch / "data_progress.json", p["run"] / "data_progress.json")
            if dedup.tokens >= cfg["source"]["candidate_tokens"] + 1000000000:
                edges.close()
                shutil.copyfile(edges_path, p["data"] / "dedup_edges.jsonl")
                if freeze_assignments(cfg, dedup, results):
                    shutil.copyfile(scratch / "data_progress.json", p["run"] / "data_progress.json")
                    emit("corpus_coordinator_complete", seconds=round(time.monotonic()-begin_all, 2))
                    return
                edges = edges_path.open("a", buffering=8*1024**2)
            prefetch(index+1)
        edges.close()
        shutil.copyfile(edges_path, p["data"] / "dedup_edges.jsonl")
        if not freeze_assignments(cfg, dedup, results):
            raise RuntimeError("Insufficient unique tokens in the approved corpus")
    finally:
        if not edges.closed:
            edges.close()
        for future in downloads.values():
            future.cancel()
        for future in processes.values():
            future.cancel()
        downloaders.shutdown(wait=True, cancel_futures=True)
        executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    prepare_corpus()
