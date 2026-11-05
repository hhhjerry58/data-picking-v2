"""Frozen answer-only MATH NLL, auxiliary general NLL, and greedy generation."""
from __future__ import annotations

import gc
import math
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .common import config, paths, read_json, read_jsonl, write_json, write_jsonl, file_hash, digest, complete_file, verified_file, emit, seed_all, utc_now
from .corpus_io import batches
from .scoring import per_example_nll
from .storage import verified_checkpoint_mirror
from .pretraining import stage_input
from .math_answers import extract_answer, equivalent


def prepare_evaluation(cfg=None):
    cfg = cfg or config()
    p = paths(cfg)
    folder = p["data"] / "evaluation"
    marker = folder / "COMPLETED.json"
    if marker.exists():
        for name, sha in read_json(marker)["files"].items():
            if file_hash(folder / name) != sha:
                raise RuntimeError("Frozen evaluation cache changed")
        return
    folder.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(p["assets"] / "pythia160m", local_files_only=True)
    tokenizer.model_max_length = 10**12
    max_length = cfg["evaluation"]["max_length"]
    def windows(identifier, ids, mask, extra):
        for number, start in enumerate(range(0, len(ids)-1, max_length-1)):
            part = ids[start:start+max_length]
            targets = [False] + mask[start+1:start+len(part)]
            if any(targets):
                yield {"id": identifier, "window_index": number, "input_ids": part, "mask": targets,
                       "scored_tokens": sum(targets), **extra}
    outputs = {}
    for split in ("math_test", "development"):
        rows = read_jsonl(p["data"] / "targets" / (split + ".jsonl"))
        all_windows, generation = [], []
        for row in rows:
            prompt = f"Question: {row['prompt']}\nAnswer: "
            full = prompt + row["response"]
            encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
            mask = [end > len(prompt) and start < len(full) for start, end in encoded["offset_mapping"]]
            if not any(mask):
                raise RuntimeError("Math question has no answer targets")
            extra = {"group_id": row.get("group_id") or row["id"],
                     "subject": str(row.get("subject") or row.get("type") or "unknown"),
                     "level": str(row.get("level") or "unknown")}
            all_windows.extend(windows(row["id"], encoded["input_ids"], mask, extra))
            generation.append({"id": row["id"], "prompt": f"Question: {row['prompt']}\nAnswer:",
                               "reference_answer": extract_answer(row["response"]), **extra})
        filename = split + "_windows.parquet"
        pq.write_table(pa.Table.from_pylist(all_windows), folder / filename, compression="zstd")
        outputs[filename] = file_hash(folder / filename)
        if split == "math_test":
            write_jsonl(folder / "generation.jsonl", generation)
            outputs["generation.jsonl"] = file_hash(folder / "generation.jsonl")
    general = []
    for rows in batches(cfg, ["general_eval"], columns=["id", "input_ids", "domain", "crawl"]):
        for row in rows:
            general.extend(windows(row["id"], row["input_ids"], [True]*len(row["input_ids"]),
                                   {"group_id": row["id"], "subject": row["domain"], "level": "general"}))
    pq.write_table(pa.Table.from_pylist(general), folder / "general_windows.parquet", compression="zstd")
    outputs["general_windows.parquet"] = file_hash(folder / "general_windows.parquet")
    write_json(marker, {"files": outputs, "config": cfg["evaluation"], "time": utc_now()})
    emit("evaluation_inputs_frozen", math_questions=len(read_jsonl(folder / "generation.jsonl")), general_documents=cfg["source"]["general_eval_documents"])


@torch.inference_mode()
def evaluate_batch(model, rows):
    maximum = max(len(row["input_ids"]) for row in rows)
    ids = torch.zeros((len(rows), maximum), dtype=torch.long, device="cuda")
    attention = torch.zeros_like(ids)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, row in enumerate(rows):
        length = len(row["input_ids"])
        ids[i, :length] = torch.tensor(row["input_ids"], dtype=torch.long, device="cuda")
        attention[i, :length] = 1
        mask[i, :length] = torch.tensor(row["mask"], dtype=torch.bool, device="cuda")
    with sdpa_kernel(SDPBackend.MATH), torch.autocast("cuda", enabled=False):
        logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
        sums, counts = per_example_nll(logits, ids, mask)
    results = []
    for row, total, count in zip(rows, sums.tolist(), counts.tolist()):
        if count != row["scored_tokens"] or not math.isfinite(total):
            raise RuntimeError("Evaluation coverage or numerical failure")
        results.append({k: row[k] for k in ("id", "window_index", "group_id", "subject", "level")} |
                       {"sum_nll": total, "scored_tokens": count})
    return results


def score_set(model, cfg, destination, name, identity):
    p = paths(cfg)
    source = p["data"] / "evaluation" / f"{name}_windows.parquet"
    folder = destination / name
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = digest({"model": identity, "input": file_hash(source), "implementation": file_hash(__file__)})
    rows = pq.read_table(source).to_pylist()
    rows.sort(key=lambda row: (-len(row["input_ids"]), row["id"], row["window_index"]))
    chunk_size, batch_size = 512, 16
    for start in range(0, len(rows), chunk_size):
        file = folder / f"chunk-{start:07d}.parquet"
        if verified_file(file):
            if read_json(str(file) + ".complete.json")["fingerprint"] != fingerprint:
                raise RuntimeError("Evaluation identity changed")
            continue
        begin = time.monotonic()
        chunk = rows[start:start+chunk_size]
        measured = [r for i in range(0, len(chunk), batch_size) for r in evaluate_batch(model, chunk[i:i+batch_size])]
        pq.write_table(pa.Table.from_pylist(measured), file, compression="zstd")
        complete_file(file, fingerprint=fingerprint)
        emit("evaluation_chunk", run=destination.name, dataset=name, windows=start+len(measured), total=len(rows), seconds=round(time.monotonic()-begin, 2))
    frame = pa.concat_tables([pq.read_table(file) for file in sorted(folder.glob("chunk-*.parquet"))]).to_pandas()
    if len(frame) != len(rows) or frame.duplicated(["id", "window_index"]).any():
        raise RuntimeError("Evaluation window coverage differs")
    documents = frame.groupby(["id", "group_id", "subject", "level"], as_index=False).agg(sum_nll=("sum_nll", "sum"), scored_tokens=("scored_tokens", "sum"))
    documents["nll"] = documents["sum_nll"] / documents["scored_tokens"]
    documents.to_parquet(destination / f"{name}_nll.parquet", index=False)
    return {"documents": len(documents), "mean_document_nll": float(documents["nll"].mean()),
            "token_weighted_nll": float(documents["sum_nll"].sum()/documents["scored_tokens"].sum()),
            "scored_tokens": int(documents["scored_tokens"].sum())}


def generate_answers(model, cfg, destination, identity):
    p = paths(cfg)
    tokenizer = AutoTokenizer.from_pretrained(p["assets"] / "pythia160m", local_files_only=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    rows = read_jsonl(p["data"] / "evaluation/generation.jsonl")
    eligible, exclusions = [], []
    for row in rows:
        count = len(tokenizer.encode(row["prompt"], add_special_tokens=False))
        if row["reference_answer"] is None:
            exclusions.append({"id": row["id"], "reason": "reference_answer_not_extracted"})
        elif count + cfg["evaluation"]["max_new_tokens"] > cfg["evaluation"]["max_length"]:
            exclusions.append({"id": row["id"], "reason": "prompt_exceeds_generation_context_budget"})
        else:
            eligible.append({**row, "prompt_tokens": count})
    eligible.sort(key=lambda row: (-row["prompt_tokens"], row["id"]))
    folder = destination / "generation"
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = digest({"model": identity, "config": cfg["evaluation"], "source": file_hash(p["data"] / "evaluation/generation.jsonl"),
                          "grading": file_hash(Path(__file__).with_name("math_answers.py")), "implementation": file_hash(__file__)})
    chunk_size = 64
    output_schema = pa.schema([("id", pa.string()), ("prompt", pa.large_string()), ("reference_answer", pa.string()),
                               ("group_id", pa.string()), ("subject", pa.string()), ("level", pa.string()),
                               ("prompt_tokens", pa.int64()), ("generation", pa.large_string()),
                               ("extracted_answer", pa.string()), ("correct", pa.bool_())])
    for start in range(0, len(eligible), chunk_size):
        file = folder / f"chunk-{start:06d}.parquet"
        if verified_file(file):
            if read_json(str(file) + ".complete.json")["fingerprint"] != fingerprint:
                raise RuntimeError("Generation policy changed")
            continue
        chunk = eligible[start:start+chunk_size]
        inputs = tokenizer([r["prompt"] for r in chunk], add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
        begin = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = model.generate(**inputs, max_new_tokens=cfg["evaluation"]["max_new_tokens"], do_sample=False,
                                       use_cache=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                       logits_to_keep=1)
        texts = tokenizer.batch_decode(generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        output = []
        for row, text in zip(chunk, texts):
            answer = extract_answer(text)
            output.append({**row, "generation": text, "extracted_answer": answer, "correct": equivalent(answer, row["reference_answer"])})
        pq.write_table(pa.Table.from_pylist(output, schema=output_schema), file, compression="zstd")
        complete_file(file, fingerprint=fingerprint)
        emit("generation_chunk", run=destination.name, questions=start+len(chunk), total=len(eligible), seconds=round(time.monotonic()-begin, 2))
    frame = pa.concat_tables([pq.read_table(file) for file in sorted(folder.glob("chunk-*.parquet"))]).to_pandas()
    if len(frame) != len(eligible) or frame["id"].duplicated().any():
        raise RuntimeError("Generation question coverage differs")
    frame.to_parquet(destination / "generation_answers.parquet", index=False)
    write_json(destination / "generation_exclusions.json", exclusions)
    return {"questions": len(frame), "correct": int(frame["correct"].sum()), "accuracy": float(frame["correct"].mean()),
            "answer_extraction_rate": float(frame["extracted_answer"].notna().mean()), "excluded": len(exclusions),
            "grader": "boxed/fbox or explicit final answer; deterministic LaTeX string normalization plus exact rational equivalence, no general symbolic judge"}


def evaluate(arm, seed, cfg=None):
    cfg = cfg or config()
    p = paths(cfg)
    destination = p["run"] / "evaluation" / f"{arm}-seed-{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "COMPLETED.json").exists():
        return
    if not (p["data"] / "evaluation/COMPLETED.json").exists():
        raise RuntimeError("Freeze shared evaluation inputs before parallel model evaluation")
    trained = p["run"] / "pretraining" / f"{arm}-seed-{seed}"
    complete = read_json(trained / "TRAINING_COMPLETED.json")
    if complete["seen_tokens"] != cfg["pretraining"]["total_prediction_tokens"]:
        raise RuntimeError("Final evaluation requires the complete token budget")
    checkpoint = Path(complete["checkpoint"])
    manifest = read_json(checkpoint / "manifest.json")
    if read_json(checkpoint / "COMPLETED.json")["manifest_hash"] != digest(manifest):
        raise RuntimeError("Incomplete final checkpoint")
    sha = manifest["files"]["state.pt"]["sha256"]
    source = verified_checkpoint_mirror(trained, checkpoint, sha, cfg)
    if source is None:
        source = stage_input(checkpoint / "state.pt", p["scratch"] / "evaluation_states" / destination.name / "state.pt", sha)
    payload = torch.load(source, map_location="cpu", mmap=False, weights_only=False)
    seed_all(42)
    torch.use_deterministic_algorithms(True)
    architecture = AutoConfig.from_pretrained(p["assets"] / "pythia160m", local_files_only=True)
    model = AutoModelForCausalLM.from_config(architecture, dtype=torch.float32, attn_implementation="sdpa")
    model.load_state_dict(payload["model"])
    del payload
    gc.collect()
    model.to("cuda").eval()
    identity = {"arm": arm, "seed": seed, "checkpoint_sha256": sha, "training_fingerprint": complete["fingerprint"]}
    metrics = {"identity": identity, "training_tokens": complete["seen_tokens"]}
    for name in ("math_test", "development", "general"):
        metrics[name] = score_set(model, cfg, destination, name, identity)
    metrics["generation"] = generate_answers(model, cfg, destination, identity)
    write_json(destination / "metrics.json", metrics)
    write_json(destination / "COMPLETED.json", {"metrics_sha256": file_hash(destination / "metrics.json"), "time": utc_now()})
    emit("evaluation_complete", **metrics)
