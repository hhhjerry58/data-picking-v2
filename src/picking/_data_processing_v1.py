"""Stream pinned RPJv2 shards, quarantine targets, deduplicate, then freeze S/P."""
from __future__ import annotations

import concurrent.futures
from collections import Counter, defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
import multiprocessing
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit

import ahocorasick
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tldextract
import xxhash
from datasketch import MinHash, MinHashLSH
from tokenizers import Tokenizer

from .common import (config, paths, read_json, read_jsonl, write_json, write_jsonl, digest,
                     complete_file, verified_file, emit, file_hash, rank_key, utc_now)
from .download import fetch, fetch_many


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def words(text):
    return re.findall(r"\w+|[^\w\s]", normalize(text))


def shingles(tokens, n=5):
    n = min(n, len(tokens))
    return {"\x1f".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)} if n else set()


class TargetOverlap:
    """Rare-anchor lookup, followed by exact/5-token containment verification."""
    def __init__(self, targets, spec):
        self.passages = []
        for row in targets:
            for field, minimum in [("prompt", spec["overlap_problem_min_tokens"]),
                                   ("response", spec["overlap_solution_min_tokens"])]:
                ts = words(row[field])
                if len(ts) >= minimum:
                    self.passages.append({"id": row["id"], "field": field,
                                          "normal": normalize(row[field]), "parts": shingles(ts)})
        frequency = Counter(p for row in self.passages for p in row["parts"])
        anchors = defaultdict(list)
        for i, row in enumerate(self.passages):
            for part in sorted(row["parts"], key=lambda x: (frequency[x], x))[:8]:
                anchors[part].append(i)
        self.automaton = ahocorasick.Automaton()
        for anchor, ids in anchors.items():
            self.automaton.add_word("\x1f" + anchor + "\x1f", ids)
        self.automaton.make_automaton()
        self.threshold = spec["overlap_containment"]

    def matches(self, text):
        normal = normalize(text)
        ts = re.findall(r"\w+|[^\w\s]", normal)
        serialized = "\x1f" + "\x1f".join(ts) + "\x1f"
        candidates = set()
        for _, ids in self.automaton.iter(serialized):
            candidates.update(ids)
        if not candidates:
            return []
        parts = shingles(ts)
        matches = []
        for i in sorted(candidates):
            row = self.passages[i]
            overlap = len(parts & row["parts"]) / len(row["parts"])
            exact = row["normal"] in normal
            if exact or overlap >= self.threshold:
                matches.append({"target_id": row["id"], "field": row["field"],
                                "exact": exact, "shingle_containment": overlap})
        return matches


def prepare_targets(cfg):
    p = paths(cfg)
    folder = p["data"] / "targets"
    if (folder / "COMPLETED.json").exists():
        m = read_json(folder / "COMPLETED.json")
        for name, sha in m["files"].items():
            if file_hash(folder / name) != sha:
                raise RuntimeError(f"Target file changed: {name}")
        return
    folder.mkdir(parents=True, exist_ok=True)
    e = cfg["evaluation"]
    old = Path(e["old_target_bundle"])
    protected = []
    for path in [old / "target/train.jsonl", old / "target/validation.jsonl",
                 Path(e["legacy_target_bundle"]) / "A2.jsonl", Path(e["legacy_target_bundle"]) / "B.jsonl"]:
        for row in read_jsonl(path):
            protected.append({k: row[k] for k in ("id", "prompt", "response", "subject", "type", "level", "group_id") if k in row})
    protected = list({row["id"]: row for row in protected}.values())
    development = [{k: row[k] for k in ("id", "prompt", "response", "subject", "type", "level", "group_id") if k in row}
                   for row in read_jsonl(old / "target/validation.jsonl")]
    lsh = MinHashLSH(threshold=0.8, num_perm=64, params=(16, 4))
    gram_sets, identities = [], []
    def near_matches(row, insert=True):
        text = normalize(row["prompt"])
        grams = {text[i:i+5] for i in range(max(1, len(text)-4))}
        sig = MinHash(num_perm=64, seed=cfg["source"]["data_seed"])
        sig.update_batch([s.encode() for s in sorted(grams)])
        matches = []
        for previous in lsh.query(sig):
            j = int(previous)
            score = len(grams & gram_sets[j]) / max(1, len(grams | gram_sets[j]))
            if score >= 0.8:
                matches.append({"id": identities[j], "jaccard": score})
        if insert:
            lsh.insert(str(len(gram_sets)), sig)
            gram_sets.append(grams)
            identities.append(row["id"])
        return matches
    for row in protected:
        near_matches(row)
    image_pattern = re.compile(r"\[asy\]|<img\b|\\includegraphics|https?://\S+\.(?:png|jpg|gif)|(?:diagram|figure|graph)\s+(?:below|above)", re.I)
    evaluation, excluded, raw_test = [], [], []
    for path in sorted((p["raw"] / "math_test").glob("*/test-*.parquet")):
        subject = path.parent.name
        for index, row in enumerate(pq.read_table(path).to_pylist()):
            record = {"id": f"{subject}:test:{index}", "subject": subject, "type": row["type"], "level": row["level"],
                      "prompt": row["problem"].strip(), "response": row["solution"].strip()}
            record["group_id"] = digest(normalize(record["prompt"]))
            raw_test.append(record)
            matches = near_matches(record)
            reason = "empty" if not record["prompt"] or not record["response"] else (
                "image_or_diagram" if image_pattern.search(record["prompt"] + "\n" + record["response"]) else (
                    "overlap_with_used_or_earlier_test_problem" if matches else None))
            if reason:
                excluded.append({"id": record["id"], "reason": reason, "matches": matches})
            else:
                evaluation.append(record)
    if len(raw_test) != 5000 or len(development) != 512:
        raise RuntimeError("Unexpected official MATH test/development coverage")
    write_jsonl(folder / "protected_targets.jsonl", protected + raw_test)
    write_jsonl(folder / "development.jsonl", development)
    write_jsonl(folder / "math_test.jsonl", evaluation)
    write_jsonl(folder / "excluded_math_test.jsonl", excluded)
    summary = {"raw_official_test": len(raw_test), "retained_test": len(evaluation), "development": len(development),
               "protected_old": len(protected), "exclusions": dict(Counter(r["reason"] for r in excluded)),
               "test_subjects": dict(Counter(r["subject"] for r in evaluation)), "source": e,
               "duplicate_rule": "64-permutation 16x4 character 5-gram LSH candidates; actual Jaccard >= 0.8"}
    write_json(folder / "summary.json", summary)
    write_json(folder / "COMPLETED.json", {"files": {path.name: file_hash(path) for path in folder.glob("*.json*") if path.name != "COMPLETED.json"}, "time": utc_now()})
    emit("targets_prepared", **{k: summary[k] for k in ("raw_official_test", "retained_test", "development")})


_WORKER = {}


def initialize_worker(cfg):
    p = paths(cfg)
    tokenizer = Tokenizer.from_file(str(p["assets"] / "pythia160m/tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    _WORKER.update(cfg=cfg, p=p, tokenizer=tokenizer,
                   overlap=TargetOverlap(read_jsonl(p["data"] / "targets/protected_targets.jsonl"), cfg["source"]),
                   domains=tldextract.TLDExtract(cache_dir=str(p["root"] / "cache/tldextract"), suffix_list_urls=()))


def shard_paths(p, shard, index):
    tag = shard["tag"]
    return {"documents": p["raw"] / "rpjv2/documents" / (tag + ".json.gz"),
            "duplicates": p["raw"] / "rpjv2/duplicates" / (tag + ".duplicates.parquet"),
            "minhash": p["raw"] / "rpjv2/minhash" / (tag + ".minhash.parquet"),
            "prepared": p["data"] / "source_shards" / f"part-{index:05d}.parquet"}


def download_shard(p, shard, index):
    out = shard_paths(p, shard, index)
    for kind in ("documents", "duplicates", "minhash"):
        fetch(shard[kind + "_url"], out[kind])
    return index


def process_shard(shard, index):
    cfg, p = _WORKER["cfg"], _WORKER["p"]
    local = shard_paths(p, shard, index)
    destination = local["prepared"]
    stats_path = destination.with_suffix(".stats.json")
    fingerprint = digest({"source": cfg["source"], "shard": shard, "targets": file_hash(p["data"] / "targets/protected_targets.jsonl"),
                          "implementation": file_hash(__file__)})
    if verified_file(destination) and stats_path.exists():
        saved = read_json(stats_path)
        if saved["fingerprint"] != fingerprint:
            raise RuntimeError("Prepared shard identity changed")
        return saved
    destination.parent.mkdir(parents=True, exist_ok=True)
    duplicate_ids = set(pq.read_table(local["duplicates"], columns=["doc_id"]).column(0).to_pylist())
    table = pq.read_table(local["minhash"], columns=["id", "signature_sim0.8"])
    signatures = {row["id"]: row["signature_sim0.8"] for row in table.to_pylist()}
    del table
    schema = pa.schema([("id", pa.string()), ("source_index", pa.int32()), ("source_row", pa.int32()),
                        ("text", pa.large_string()), ("url", pa.string()), ("domain", pa.string()), ("crawl", pa.string()),
                        ("text_hash", pa.string()), ("sample_key", pa.uint64()), ("pythia_tokens", pa.int64()),
                        ("input_ids", pa.large_list(pa.uint32())), ("minhash_bands", pa.list_(pa.uint64()))])
    writer = pq.ParquetWriter(destination, schema, compression="zstd", compression_level=3, use_dictionary=["domain", "crawl"])
    counters = Counter()
    quarantine_path = destination.with_suffix(".quarantine.jsonl")
    batch = []
    def flush():
        if not batch:
            return
        encodings = _WORKER["tokenizer"].encode_batch([r["text"] for r in batch], add_special_tokens=False)
        accepted = []
        for row, encoding in zip(batch, encodings):
            if len(encoding.ids) < 2:
                counters["unscorable"] += 1
                continue
            row["input_ids"] = encoding.ids
            row["pythia_tokens"] = len(encoding.ids) + 1  # Includes one EOS when pretraining texts are packed.
            counters["tokens"] += row["pythia_tokens"]
            counters["valid"] += 1
            accepted.append(row)
        if accepted:
            writer.write_table(pa.Table.from_pylist(accepted, schema=schema), row_group_size=1024)
        batch.clear()
    begin = time.monotonic()
    try:
        with gzip.open(local["documents"], "rt", encoding="utf-8") as source, quarantine_path.open("w") as quarantine:
            for row_number, line in enumerate(source):
                counters["raw"] += 1
                identifier = f"{shard['tag']}.json.gz/{row_number}"
                record = json.loads(line)
                if identifier in duplicate_ids:
                    counters["upstream_exact_duplicate"] += 1
                    continue
                if record.get("language") != "en":
                    counters["non_english"] += 1
                    continue
                text = record.get("raw_content", "")
                if not text.strip():
                    counters["empty"] += 1
                    continue
                matches = _WORKER["overlap"].matches(text)
                if matches:
                    counters["target_overlap"] += 1
                    quarantine.write(json.dumps({"id": identifier, "matches": matches}) + "\n")
                    continue
                if identifier not in signatures:
                    raise RuntimeError(f"Missing upstream MinHash row for {identifier}")
                bands = signatures[identifier] or []
                if len(bands) not in (0, 9):
                    raise RuntimeError("Unexpected RPJv2 0.8 MinHash band count")
                if not bands:
                    counters["empty_minhash_signature"] += 1
                domain = _WORKER["domains"](record.get("url", "")).top_domain_under_public_suffix
                if not domain:
                    domain = record.get("source_domain") or urlsplit(record.get("url", "")).hostname or identifier
                batch.append({"id": identifier, "source_index": index, "source_row": row_number, "text": text,
                              "url": record.get("url", ""), "domain": domain, "crawl": shard["tag"].split("/")[0],
                              "text_hash": hashlib.sha256(normalize(text).encode()).hexdigest(),
                              "sample_key": int(rank_key(identifier, cfg["source"]["data_seed"])[:16], 16),
                              "minhash_bands": [xxhash.xxh64_intdigest(b) for b in bands]})
                if len(batch) >= 512:
                    flush()
            flush()
    finally:
        writer.close()
    result = {"index": index, "tag": shard["tag"], "fingerprint": fingerprint, "counts": dict(counters),
              "path": str(destination), "seconds": time.monotonic()-begin}
    complete_file(destination, fingerprint=fingerprint)
    write_json(stats_path, result)
    emit("shard_prepared", index=index, tag=shard["tag"], **counters, seconds=round(result["seconds"], 2))
    return result


class CorpusDedup:
    """Greedy seeded-source representatives; upstream LSH plus verified Jaccard."""
    def __init__(self, threshold):
        self.threshold = threshold
        self.exact = {}
        self.bands = [defaultdict(list) for _ in range(9)]
        self.texts = []
        self.meta = []
        self.tokens = 0
        self.counts = Counter()
        self.parts = lru_cache(maxsize=2048)(lambda index: shingles(words(self.texts[index])))

    def add(self, row):
        existing = self.exact.get(row["text_hash"])
        if existing is not None:
            self.counts["exact_duplicate"] += 1
            return {"id": row["id"], "representative": self.meta[existing]["id"], "kind": "exact", "jaccard": 1.0}
        candidates = set()
        for band, value in enumerate(row["minhash_bands"]):
            candidates.update(self.bands[band].get(value, ()))
        if candidates:
            current_parts = shingles(words(row["text"]))
            for previous in sorted(candidates):
                previous_parts = self.parts(previous)
                if min(len(current_parts), len(previous_parts)) < self.threshold * max(len(current_parts), len(previous_parts)):
                    continue
                similarity = len(current_parts & previous_parts) / max(1, len(current_parts | previous_parts))
                if similarity >= self.threshold:
                    self.counts["near_duplicate"] += 1
                    return {"id": row["id"], "representative": self.meta[previous]["id"], "kind": "near", "jaccard": similarity}
        index = len(self.meta)
        self.exact[row["text_hash"]] = index
        for band, value in enumerate(row["minhash_bands"]):
            self.bands[band][value].append(index)
        self.texts.append(row["text"])
        self.meta.append({k: row[k] for k in ("id", "source_index", "source_row", "domain", "crawl", "text_hash", "sample_key", "pythia_tokens")})
        self.tokens += row["pythia_tokens"]
        self.counts["kept"] += 1
        return None


def freeze_assignments(cfg, dedup, results):
    p = paths(cfg)
    s = cfg["source"]
    keys = np.asarray([r["sample_key"] for r in dedup.meta], dtype=np.uint64)
    order = np.argsort(keys, kind="stable")
    nlabels, ngeneral = s["label_documents"], s["general_eval_documents"]
    if len(order) < nlabels + ngeneral:
        return False
    totals = np.asarray([r["pythia_tokens"] for r in dedup.meta], dtype=np.int64)
    remaining = order[nlabels+ngeneral:]
    cumulative = np.cumsum(totals[remaining])
    if not len(cumulative) or cumulative[-1] < s["candidate_tokens"]:
        return False
    n_candidate = int(np.searchsorted(cumulative, s["candidate_tokens"]) + 1)
    assignment = np.full(len(order), "unused", dtype=object)
    for i in order[:nlabels]:
        key = int(rank_key(dedup.meta[i]["domain"], s["data_seed"])[:16], 16) / 2**64
        assignment[i] = "label_train" if key < 0.8 else ("label_dev" if key < 0.9 else "label_test")
    assignment[order[nlabels:nlabels+ngeneral]] = "general_eval"
    assignment[remaining[:n_candidate]] = "candidate"
    for i, row in enumerate(dedup.meta):
        row["partition"] = assignment[i]
        row["group_id"] = row["text_hash"]
    filename = p["data"] / "assignments.parquet"
    pq.write_table(pa.Table.from_pylist(dedup.meta), filename, compression="zstd", row_group_size=100000)
    complete_file(filename)
    partition_counts = Counter(assignment)
    token_counts = {part: int(totals[assignment == part].sum()) for part in partition_counts}
    if sum(partition_counts[x] for x in ("label_train", "label_dev", "label_test")) != nlabels:
        raise RuntimeError("Formal label set size mismatch")
    domain_sets = {part: {r["domain"] for r in dedup.meta if r["partition"] == part} for part in ("label_train", "label_dev", "label_test")}
    if any(domain_sets[a] & domain_sets[b] for a, b in (("label_train", "label_dev"), ("label_train", "label_test"), ("label_dev", "label_test"))):
        raise RuntimeError("Classifier split domain leakage")
    summary = {"document_counts": dict(partition_counts), "pythia_tokens_including_eos": token_counts,
               "source_shards": results, "deduplication": dict(dedup.counts), "source_config": s,
               "dedup_rule": "Upstream exact-duplicate IDs, normalized SHA256, upstream 9x13 RPJv2 MinHash candidate bands, actual normalized punctuation-aware 5-token Jaccard >= 0.8; seeded source order keeps first representative. Approximate candidate recall.",
               "assignment_sha256": file_hash(filename), "created_at_utc": utc_now()}
    write_json(p["data"] / "corpus_summary.json", summary)
    write_json(p["data"] / "CORPUS_COMPLETED.json", {"assignment_sha256": summary["assignment_sha256"], "source_files": {r["path"]: read_json(r["path"] + ".complete.json")["sha256"] for r in results}, "time": utc_now()})
    emit("corpus_frozen", documents=summary["document_counts"], tokens=token_counts)
    return True


def prepare_corpus(cfg=None):
    cfg = cfg or config()
    p = paths(cfg)
    prepare_targets(cfg)
    if (p["data"] / "CORPUS_COMPLETED.json").exists():
        marker = read_json(p["data"] / "CORPUS_COMPLETED.json")
        if file_hash(p["data"] / "assignments.parquet") != marker["assignment_sha256"]:
            raise RuntimeError("Frozen assignment changed")
        emit("corpus_already_complete")
        return
    manifest = read_json(p["raw"] / "source_manifest.json")
    shards = manifest["shards"]
    dedup = CorpusDedup(cfg["source"]["dedup_jaccard"])
    results = []
    columns = ["id", "source_index", "source_row", "text", "domain", "crawl", "text_hash", "sample_key", "pythia_tokens", "minhash_bands"]
    p["data"].mkdir(parents=True, exist_ok=True)
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=cfg["source"]["workers"],
        mp_context=multiprocessing.get_context("spawn"), initializer=initialize_worker, initargs=(cfg,))
    downloaders = concurrent.futures.ThreadPoolExecutor(max_workers=8)
    download_jobs, process_jobs = {}, {}
    prefetch = 32
    next_download = 0
    def ensure_prefetch(current):
        nonlocal next_download
        while next_download < min(len(shards), current + prefetch):
            j = next_download
            download_jobs[j] = downloaders.submit(download_shard, p, shards[j], j)
            next_download += 1
    try:
        ensure_prefetch(0)
        with (p["data"] / "dedup_edges.jsonl").open("w") as edges:
            for i, shard in enumerate(shards):
                for j, future in list(download_jobs.items()):
                    if future.done() and j not in process_jobs:
                        future.result()
                        process_jobs[j] = executor.submit(process_shard, shards[j], j)
                if i not in process_jobs:
                    download_jobs[i].result()
                    process_jobs[i] = executor.submit(process_shard, shard, i)
                result = process_jobs.pop(i).result()
                del download_jobs[i]
                results.append(result)
                begin = time.monotonic()
                for batch in pq.ParquetFile(result["path"]).iter_batches(batch_size=2048, columns=columns):
                    for row in batch.to_pylist():
                        edge = dedup.add(row)
                        if edge:
                            edges.write(json.dumps(edge) + "\n")
                edges.flush()
                emit("global_dedup_progress", shards=i+1, kept_documents=len(dedup.meta), kept_tokens=dedup.tokens,
                     duplicates=dict(dedup.counts), seconds=round(time.monotonic()-begin, 2))
                write_json(p["run"] / "data_progress.json", {"completed_source_shards": i+1, "kept_documents": len(dedup.meta),
                           "kept_tokens": dedup.tokens, "dedup_counts": dict(dedup.counts), "time": utc_now()})
                if dedup.tokens >= cfg["source"]["candidate_tokens"] + 1000000000:
                    if freeze_assignments(cfg, dedup, results):
                        return
                ensure_prefetch(i+1)
        if not freeze_assignments(cfg, dedup, results):
            raise RuntimeError("Insufficient unique tokens in the authorized source pool")
    finally:
        for future in download_jobs.values():
            future.cancel()
        for future in process_jobs.values():
            future.cancel()
        downloaders.shutdown(wait=True, cancel_futures=True)
        executor.shutdown(wait=True, cancel_futures=True)


def partition_batches(cfg, partitions, columns=None, batch_size=1024):
    """Yield source rows selected by the frozen, disjoint assignment manifest."""
    p = paths(cfg)
    assignments = pq.read_table(p["data"] / "assignments.parquet", columns=["id", "source_index", "partition"])
    by_shard = defaultdict(dict)
    wanted = set(partitions)
    for row in assignments.to_pylist():
        if row["partition"] in wanted:
            by_shard[row["source_index"]][row["id"]] = row["partition"]
    columns = list(columns) if columns else None
    if columns is not None and "id" not in columns:
        columns.insert(0, "id")
    for index in sorted(by_shard):
        selected = by_shard[index]
        path = p["data"] / "source_shards" / f"part-{index:05d}.parquet"
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
            rows = []
            for row in batch.to_pylist():
                if row["id"] in selected:
                    rows.append({**row, "partition": selected[row["id"]]})
            if rows:
                yield rows


if __name__ == "__main__":
    prepare_corpus()
