from __future__ import annotations

import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for data in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(data)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def config(path=None):
    cfg = yaml.safe_load(Path(path or REPO / "configs/experiment.yaml").read_text())
    if os.environ.get('DATA_PICKING_RUNTIME_ROOT'):
        cfg['runtime_root'] = str(Path(os.environ['DATA_PICKING_RUNTIME_ROOT']).expanduser().resolve())
    if cfg["source"]["label_documents"] != 100000:
        raise ValueError("The authorized formal label set has 100,000 documents")
    for stage in ("fasttext", "pretraining"):
        if not cfg[stage]["save_optimizer"] or not cfg[stage]["auto_resume"]:
            raise ValueError(f"{stage}: optimizer state and auto-resume are required")
    return cfg


def paths(cfg):
    root = Path(cfg["artifact_root"])
    scratch = Path(cfg["runtime_root"])
    return {"root": root, "scratch": scratch, "raw": root / "data/raw", "data": root / "data/processed",
            "run": root / "runs" / cfg["run_id"], "assets": root / "cache/assets"}


def emit(event, **fields):
    print(json.dumps({"event": event, "time": utc_now(), **fields}, ensure_ascii=False), flush=True)


def rank_key(identifier, seed):
    return hashlib.blake2b(f"{seed}:{identifier}".encode(), digest_size=16).hexdigest()


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def implementation_hash(names):
    return {str(name): file_hash(REPO / name) for name in names}


def complete_file(path, **metadata):
    path = Path(path)
    result = {"size": path.stat().st_size, "sha256": file_hash(path), "completed_at_utc": utc_now(), **metadata}
    write_json(str(path) + ".complete.json", result)
    return result


def verified_file(path, verify_hash=True):
    path = Path(path)
    marker = Path(str(path) + ".complete.json")
    if not path.is_file() or not marker.is_file():
        return False
    meta = read_json(marker)
    return path.stat().st_size == meta["size"] and (not verify_hash or file_hash(path) == meta["sha256"])
