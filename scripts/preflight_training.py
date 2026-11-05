#!/usr/bin/env python
"""Validate the real 160M trainer's finite updates, optimizer and cursor restore."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pyarrow.parquet as pq
import torch
from picking.common import config, paths, write_json, file_hash, complete_file, emit, utc_now
from picking.pretraining import train, create_model
from picking.storage import verified_checkpoint_mirror
from picking.common import read_json
from torch.nn.attention import SDPBackend, sdpa_kernel


def main():
    cfg = config()
    p = paths(cfg)
    folder = p["run"] / "preflight"
    folder.mkdir(parents=True, exist_ok=True)
    target_tokens = 3 * cfg["pretraining"]["effective_batch_size"] * (cfg["pretraining"]["sequence_length"]-1)
    input_tokens = 3 * cfg["pretraining"]["effective_batch_size"] * cfg["pretraining"]["sequence_length"]
    input_path = folder / "input.bin"
    if not input_path.exists():
        written = 0
        with input_path.open("wb") as output:
            for source in sorted((p["data"] / "source_shards").glob("part-*.parquet")):
                if not Path(str(source) + ".complete.json").exists():
                    continue
                for batch in pq.ParquetFile(source).iter_batches(batch_size=64, columns=["input_ids"]):
                    for row in batch.to_pylist():
                        ids = np.asarray(row["input_ids"] + [0], dtype=np.uint32)
                        take = min(len(ids), input_tokens-written)
                        output.write(ids[:take].tobytes())
                        written += take
                        if written == input_tokens:
                            break
                    if written == input_tokens:
                        break
                if written == input_tokens:
                    break
        if written != input_tokens:
            raise RuntimeError("Insufficient prepared tokens for integrity check")
        complete_file(input_path, tokens=written, purpose="trainer integrity only; not a selection experiment")
    model, _ = create_model(cfg, 42, fixed_causal=False)
    model.eval()
    probe = torch.arange(64, device="cuda", dtype=torch.long).reshape(2, 32) + 10
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        standard = model(probe, use_cache=False).logits
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        ALL_ATTENTION_FUNCTIONS.register("picking_fixed_causal_sdpa", sdpa_attention_forward)
        model.config._attn_implementation = "picking_fixed_causal_sdpa"
        fixed = model(probe, use_cache=False).logits
        causal_error = float((standard-fixed).abs().max())
        future_changed = probe.clone()
        future_changed[:, 16:] += 100
        changed = model(future_changed, use_cache=False).logits
        future_leakage = float((fixed[:, :16]-changed[:, :16]).abs().max())
    if causal_error > 2e-5 or future_leakage > 2e-5:
        raise RuntimeError(f"Fixed causal SDPA equivalence/causality failure: {causal_error}, {future_leakage}")
    del model, standard, fixed, changed
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    write_json(folder / "causal_attention_verification.json", {"max_logit_error": causal_error, "future_token_leakage": future_leakage, "tolerance": 2e-5})
    common = dict(cfg=cfg, input_override=input_path, token_budget_override=target_tokens,
                  compile_model=True, attention_backend="flash")
    full_path = train("random", 42, run_suffix="-preflight-v5-continuous", **common)
    train("random", 42, stop_after_steps=1, run_suffix="-preflight-v5-resumed", **common)
    resumed_path = train("random", 42, run_suffix="-preflight-v5-resumed", **common)
    full_sha = read_json(full_path / "manifest.json")["files"]["state.pt"]["sha256"]
    resumed_sha = read_json(resumed_path / "manifest.json")["files"]["state.pt"]["sha256"]
    full_file = verified_checkpoint_mirror(full_path.parent.parent, full_path, full_sha, cfg) or full_path / "state.pt"
    resumed_file = verified_checkpoint_mirror(resumed_path.parent.parent, resumed_path, resumed_sha, cfg) or resumed_path / "state.pt"
    full = torch.load(full_file, map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_file, map_location="cpu", weights_only=False)
    if full["trainer"] != resumed["trainer"] or full["scheduler"] != resumed["scheduler"]:
        raise RuntimeError("Trainer/scheduler restore mismatch")
    max_model_error = max(float((full["model"][key]-resumed["model"][key]).abs().max()) for key in full["model"])
    max_optimizer_error = 0.0
    for key, state in full["optimizer"]["state"].items():
        for field, value in state.items():
            other = resumed["optimizer"]["state"][key][field]
            if isinstance(value, torch.Tensor):
                max_optimizer_error = max(max_optimizer_error, float((value-other).abs().max()))
    if max_model_error > 2e-6 or max_optimizer_error > 2e-6:
        raise RuntimeError(f"Resumed update discrepancy exceeds tolerance: {max_model_error}, {max_optimizer_error}")
    result = {"passed": True, "save_optimizer": True, "auto_resume": True, "attention_backend": "flash", "compiled": True,
              "target_tokens": target_tokens, "seen_tokens": full["trainer"]["seen_tokens"], "global_step": full["trainer"]["global_step"],
              "max_model_error": max_model_error, "max_optimizer_error": max_optimizer_error, "tolerance": 2e-6,
              "initial_parameters_sha256": full["trainer"]["initial_parameters_sha256"],
              "pretraining_implementation_sha256": file_hash(Path(__file__).resolve().parents[1] / "src/picking/pretraining.py"),
              "verified_at_utc": utc_now()}
    write_json(folder / "training_resume_verification.json", result)
    emit("training_preflight_passed", **result)


if __name__ == "__main__":
    main()
