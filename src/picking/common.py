from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

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


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def paths(cfg):
    root = Path(cfg["artifact_root"])
    scratch = Path(cfg["runtime_root"])
    return {"root": root, "scratch": scratch, "run": root / "runs" / cfg["run_id"]}


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
