"""Equal-token GPT-NeoX pretraining with full-state, fingerprinted resume."""
from __future__ import annotations

import gc
import math
import os
import random
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from .common import config, paths, read_json, write_json, file_hash, digest, complete_file, verified_file, emit, seed_all, utc_now
from .selection import input_tokens_for_targets
from .storage import checkpoint_transport, verified_checkpoint_mirror
from .training_log import TrainingLog


def parameter_hash(model):
    import hashlib
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


def latest_checkpoint(folder, fingerprint):
    candidates = []
    for path in (folder / "checkpoints").glob("step-*"):
        if not (path / "COMPLETED.json").exists():
            continue
        manifest = read_json(path / "manifest.json")
        if read_json(path / "COMPLETED.json")["manifest_hash"] != digest(manifest) or manifest["fingerprint"] != fingerprint:
            raise RuntimeError("Checkpoint fingerprint mismatch")
        candidates.append((manifest["global_step"], path, manifest))
    return max(candidates, default=None, key=lambda x: x[0])


def save_checkpoint(folder, model, optimizer, scheduler, trainer, fingerprint, cfg, log_path, keep=2):
    checkpoint = folder / "checkpoints" / f"step-{trainer['global_step']:06d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    payload = {"model": cpu_copy(model.state_dict()), "optimizer": cpu_copy(optimizer.state_dict()),
               "scheduler": scheduler.state_dict(), "trainer": dict(trainer), "rng": rng_state(), "fingerprint": fingerprint}
    transport = checkpoint_transport(cfg, folder)
    state_file = transport.scratch / checkpoint.name / "state.pt"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with state_file.open("wb") as output:
        torch.save(payload, output)
        output.flush()
        os.fsync(output.fileno())
    del payload
    first_checkpoint = not any((folder / "checkpoint_read_records").glob("*.json"))
    transport_record = transport.persist(state_file, checkpoint / "state.pt", full_readback=first_checkpoint)
    metadata = {"size": transport_record["bytes"], "sha256": transport_record["sha256"]}
    log_target = checkpoint / "train.jsonl"
    shutil.copyfile(log_path, log_target)
    log_meta = {"size": log_target.stat().st_size, "sha256": file_hash(log_target)}
    manifest = {"fingerprint": fingerprint, "global_step": trainer["global_step"], "seen_tokens": trainer["seen_tokens"],
                "data_cursor": trainer["sequence_cursor"], "save_optimizer": True, "auto_resume": True,
                "files": {"state.pt": metadata, "train.jsonl": log_meta}, "time": utc_now()}
    write_json(checkpoint / "manifest.json", manifest)
    write_json(checkpoint / "COMPLETED.json", {"manifest_hash": digest(manifest)})
    completed = sorted([path for path in (folder / "checkpoints").glob("step-*") if (path / "COMPLETED.json").exists()])
    for old in completed[:-keep]:
        if old != checkpoint:
            shutil.rmtree(old)
            mirror = transport.scratch / old.name
            if mirror.exists():
                shutil.rmtree(mirror)
    emit("pretraining_checkpoint", path=str(checkpoint), step=trainer["global_step"], tokens=trainer["seen_tokens"])
    return checkpoint


def stage_input(source, destination, expected_sha256):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    marker = Path(str(destination) + ".verified.json")
    if marker.exists() and destination.exists():
        record = read_json(marker)
        if record["sha256"] == expected_sha256 and file_hash(destination) == expected_sha256:
            return destination
    shutil.copyfile(source, destination)
    if file_hash(destination) != expected_sha256:
        raise RuntimeError("Runtime copy differs from persistent input")
    write_json(marker, {"source": str(source), "sha256": expected_sha256, "size": destination.stat().st_size})
    return destination


def create_model(cfg, seed, fixed_causal=True):
    seed_all(seed)
    torch.use_deterministic_algorithms(cfg["pretraining"].get("deterministic_algorithms", True))
    architecture = AutoConfig.from_pretrained(paths(cfg)["assets"] / "pythia160m", local_files_only=True)
    architecture.use_cache = False
    # Training inputs are independent fixed-length causal blocks with no left
    # padding or reset positions. Tail padding is to the RIGHT and unscored.
    # HF otherwise materializes a redundant dense causal mask while compiling,
    # preventing SDPA Flash dispatch. A named interface keeps the same causal
    # operation and lets SDPA use is_causal=True without that explicit mask.
    implementation = "sdpa"
    if fixed_causal:
        implementation = "picking_fixed_causal_sdpa"
        ALL_ATTENTION_FUNCTIONS.register(implementation, sdpa_attention_forward)
    model = AutoModelForCausalLM.from_config(architecture, dtype=torch.float32, attn_implementation=implementation)
    initial_hash = parameter_hash(model)
    return model.to("cuda").train(), initial_hash


def make_optimizer(model, spec):
    decay, no_decay = [], []
    for _, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": spec["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=spec["learning_rate"],
                            betas=tuple(spec["betas"]), eps=spec["adam_epsilon"], foreach=False)


def make_scheduler(optimizer, spec):
    full_tokens = spec["effective_batch_size"] * (spec["sequence_length"]-1)
    total = spec["total_prediction_tokens"]
    warmup = spec["warmup_ratio"] * total
    ratio = spec["min_learning_rate"] / spec["learning_rate"]
    def multiplier(step):
        next_tokens = min(total, (step+1)*full_tokens)
        if next_tokens <= warmup:
            return next_tokens / max(1, warmup)
        progress = (next_tokens-warmup)/(total-warmup)
        return ratio + (1-ratio)*(1+math.cos(math.pi*progress))/2
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def make_loss(model, compile_model):
    def loss_sum(input_ids, labels):
        logits = model(input_ids=input_ids, use_cache=False).logits
        return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                               labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
    return torch.compile(loss_sum, fullgraph=True) if compile_model else loss_sum


def separate_gradient_aliases(model):
    """AOTAutograd may alias the equal bias gradients of parallel residuals.

    Those values are mathematically equal, but their accumulators must not share
    memory: accumulation and clipping update gradients in place. Clone duplicate
    buffers immediately after backward, before another microbatch is added.
    """
    addresses, aliases = {}, []
    for name, parameter in model.named_parameters():
        if parameter.grad is None or parameter.grad.numel() == 0:
            continue
        address = parameter.grad.data_ptr()
        if address in addresses:
            aliases.append((addresses[address], name))
            parameter.grad = parameter.grad.clone(memory_format=torch.preserve_format)
        else:
            addresses[address] = name
    return aliases


def train(arm, seed, cfg=None, stop_after_steps=None, run_suffix="", input_override=None,
          token_budget_override=None, compile_model=True, attention_backend="flash"):
    cfg = cfg or config()
    p = paths(cfg)
    spec = dict(cfg["pretraining"])
    if token_budget_override is not None:
        spec["total_prediction_tokens"] = int(token_budget_override)
    if not spec["save_optimizer"] or not spec["auto_resume"]:
        raise ValueError("Full optimizer state and auto-resume are required")
    folder = p["run"] / "pretraining" / f"{arm}-seed-{seed}{run_suffix}"
    folder.mkdir(parents=True, exist_ok=True)
    source = Path(input_override) if input_override else p["run"] / "selection" / arm / f"packed-seed-{seed}.bin"
    source_sha = file_hash(source)
    expected_inputs = input_tokens_for_targets(spec["total_prediction_tokens"], spec["sequence_length"])
    if source.stat().st_size != expected_inputs * 4:
        raise RuntimeError(f"Input token length mismatch: {source.stat().st_size//4}, expected {expected_inputs}")
    identity = {"arm": arm, "seed": seed, "training": spec, "input_sha256": source_sha,
                "architecture_sha256": file_hash(p["assets"] / "pythia160m/config.json"),
                "implementation_sha256": file_hash(__file__), "compile_model": compile_model,
                "attention_backend": attention_backend, "torch_version": torch.__version__}
    fingerprint = digest(identity)
    effective_path = folder / "effective_config.json"
    if effective_path.exists() and read_json(effective_path)["fingerprint"] != fingerprint:
        raise RuntimeError("Training configuration/data changed during resume")
    latest = latest_checkpoint(folder, fingerprint)
    if latest and latest[2]["seen_tokens"] == spec["total_prediction_tokens"]:
        write_json(folder / "TRAINING_COMPLETED.json", {"checkpoint": str(latest[1]), "fingerprint": fingerprint,
                   "global_step": latest[0], "seen_tokens": latest[2]["seen_tokens"], "time": utc_now()})
        emit("pretraining_already_complete", arm=arm, seed=seed)
        return latest[1]
    runtime_input = stage_input(source, p["scratch"] / "training_inputs" / folder.name / "tokens.bin", source_sha)
    token_array = np.memmap(runtime_input, dtype=np.uint32, mode="r")
    model, initial_hash = create_model(cfg, seed)
    optimizer = make_optimizer(model, spec)
    scheduler = make_scheduler(optimizer, spec)
    trainer = {"global_step": 0, "seen_tokens": 0, "sequence_cursor": 0, "initial_parameters_sha256": initial_hash}
    if latest:
        state_file = latest[1] / "state.pt"
        expected = latest[2]["files"]["state.pt"]
        local = verified_checkpoint_mirror(folder, latest[1], expected["sha256"], cfg)
        if local is None:
            local = stage_input(state_file, p["scratch"] / "checkpoints" / folder.name / "resume.pt", expected["sha256"])
        payload = torch.load(local, map_location="cpu", mmap=False, weights_only=False)
        if payload["fingerprint"] != fingerprint:
            raise RuntimeError("Restored trainer fingerprint mismatch")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        trainer = payload["trainer"]
        restore_rng(payload["rng"])
        optimizer_steps = {int(value["step"]) for value in optimizer.state.values()}
        if optimizer_steps != {trainer["global_step"]} or scheduler.last_epoch != trainer["global_step"]:
            raise RuntimeError("Optimizer/scheduler cursor not restored")
        if trainer["initial_parameters_sha256"] != initial_hash:
            raise RuntimeError("Paired initialization changed")
        write_json(folder / "resume_verification.json", {"checkpoint": str(latest[1]), "optimizer_steps": sorted(optimizer_steps),
                   "scheduler_step": scheduler.last_epoch, "trainer": trainer, "rng_restored": True, "time": utc_now()})
        del payload
        gc.collect()
    effective = {"fingerprint": fingerprint, "identity": identity, "save_optimizer": True, "auto_resume": True,
                 "initial_parameters_sha256": initial_hash, "parameters": sum(v.numel() for v in model.parameters()),
                 "output_dir": str(folder), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                 "gpu": torch.cuda.get_device_name(), "runtime_input_sha256": source_sha}
    write_json(effective_path, effective)
    emit("pretraining_effective_config", arm=arm, seed=seed, fingerprint=fingerprint, save_optimizer=True, auto_resume=True,
         parameters=effective["parameters"], initial_hash=initial_hash, target_tokens=spec["total_prediction_tokens"],
         attention_backend=attention_backend, compiled=compile_model)
    loss_function = make_loss(model, compile_model)
    backend = SDPBackend.FLASH_ATTENTION if attention_backend == "flash" else SDPBackend.MATH
    length, batch_size, micro = spec["sequence_length"], spec["effective_batch_size"], spec["micro_batch_size"]
    total_sequences = math.ceil(len(token_array)/length)
    last_checkpoint = latest[1] if latest else None
    log = TrainingLog(cfg, folder, trainer["global_step"], last_checkpoint)
    while trainer["seen_tokens"] < spec["total_prediction_tokens"]:
        begin = time.monotonic()
        sequence_start = trainer["sequence_cursor"]
        n_sequences = min(batch_size, total_sequences-sequence_start)
        rows, targets = [], []
        valid = 0
        for index in range(sequence_start, sequence_start+n_sequences):
            start = index*length
            data = np.asarray(token_array[start:min(start+length, len(token_array))], dtype=np.int64).copy()
            row = np.zeros(length, dtype=np.int64)
            label = np.full(length, -100, dtype=np.int64)
            row[:len(data)] = data
            label[:len(data)] = data
            label[0] = -100
            valid += max(0, len(data)-1)
            rows.append(row)
            targets.append(label)
        if valid <= 0 or trainer["seen_tokens"] + valid > spec["total_prediction_tokens"]:
            raise RuntimeError("Training token budget/cursor failure")
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        used_lr = scheduler.get_last_lr()[0]
        alias_count = 0
        for start in range(0, n_sequences, micro):
            ids = torch.from_numpy(np.stack(rows[start:start+micro])).to("cuda")
            labels = torch.from_numpy(np.stack(targets[start:start+micro])).to("cuda")
            with sdpa_kernel(backend):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    summed_loss = loss_function(ids, labels)
                if not torch.isfinite(summed_loss):
                    raise RuntimeError("Non-finite training loss")
                (summed_loss/valid).backward()
            alias_count += len(separate_gradient_aliases(model))
            loss_total += float(summed_loss.detach())
            del ids, labels, summed_loss
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec["max_grad_norm"], error_if_nonfinite=True, foreach=False)
        optimizer.step()
        scheduler.step()
        trainer["global_step"] += 1
        trainer["seen_tokens"] += valid
        trainer["sequence_cursor"] += n_sequences
        record = {"step": trainer["global_step"], "seen_tokens": trainer["seen_tokens"], "step_tokens": valid,
                  "loss": loss_total/valid, "lr": used_lr, "next_lr": scheduler.get_last_lr()[0],
                  "grad_norm": float(norm), "gradient_alias_buffers_separated": alias_count, "seconds": time.monotonic()-begin}
        log.write(record)
        if trainer["global_step"] % 100 == 0:
            log.persist()
        if trainer["global_step"] <= 3 or trainer["global_step"] % 10 == 0:
            emit("pretraining_step", arm=arm, seed=seed, **record)
            write_json(folder / "progress.json", record)
        stop = stop_after_steps is not None and trainer["global_step"] >= stop_after_steps
        complete = trainer["seen_tokens"] == spec["total_prediction_tokens"]
        if trainer["global_step"] % spec["save_steps"] == 0 or stop or complete:
            log.flush()
            last_checkpoint = save_checkpoint(folder, model, optimizer, scheduler, trainer, fingerprint, cfg, log.local, spec["keep_checkpoints"])
            log.persist()
        if stop:
            break
    log.close()
    if trainer["seen_tokens"] == spec["total_prediction_tokens"]:
        write_json(folder / "TRAINING_COMPLETED.json", {"checkpoint": str(last_checkpoint), "fingerprint": fingerprint,
                   "global_step": trainer["global_step"], "seen_tokens": trainer["seen_tokens"], "time": utc_now()})
    del optimizer, scheduler, model, loss_function
    gc.collect()
    torch.cuda.empty_cache()
    return last_checkpoint
