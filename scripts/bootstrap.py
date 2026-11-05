#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hashlib
import json
import random
import tarfile
import urllib.request

from picking.common import config, paths, read_json, utc_now, write_json
from picking.download import fetch, fetch_many


def main():
    cfg = config()
    p = paths(cfg)
    for path in p.values():
        path.mkdir(parents=True, exist_ok=True)
    source = cfg["source"]
    hf = f"https://huggingface.co/datasets/{source['dataset']}/resolve/{source['revision']}"
    files = ["_CC_SNAPSHOT_IDS", "RedPajama-Data-V2.py", "README.md", "urls/missing-documents.txt", "urls/missing-duplicates.txt"]
    items = [(f"{hf}/{name}", p["raw"] / "source_metadata" / name) for name in files]
    model = cfg["pretraining"]
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        items.append((f"https://huggingface.co/{model['architecture']}/resolve/{model['revision']}/{name}", p["assets"] / "pythia160m" / name))
    evaluation = cfg["evaluation"]
    for subject in ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]:
        name = f"{subject}/test-00000-of-00001.parquet"
        items.append((f"https://huggingface.co/datasets/{evaluation['dataset']}/resolve/{evaluation['revision']}/{name}", p["raw"] / "math_test" / name))
    fetch_many(items, workers=8)
    snapshots = (p["raw"] / "source_metadata/_CC_SNAPSHOT_IDS").read_text().splitlines()
    missing = set((p["raw"] / "source_metadata/urls/missing-documents.txt").read_text().splitlines())
    base = "https://data.together.xyz/redpajama-data-v2/v1.0.0"
    shards = []
    for snapshot in snapshots:
        for number in range(10):
            for partition in ["head", "middle"]:
                tag = f"{snapshot}/{number:04d}/en_{partition}"
                url = f"{base}/documents/{tag}.json.gz"
                if url not in missing:
                    shards.append({"tag": tag, "documents_url": url, "duplicates_url": f"{base}/duplicates/{tag}.duplicates.parquet",
                                   "minhash_url": f"{base}/minhash/{tag}.minhash.parquet"})
    random.Random(source["data_seed"]).shuffle(shards)
    write_json(p["raw"] / "source_manifest.json", {"dataset": source, "created_at_utc": utc_now(), "sampling_unit": "seeded random order of eligible sample-100B English shards; document splits use stable hashes", "shards": shards})
    write_json(p["run"] / "approved_config.json", {"approved_at_utc": utc_now(), "config": cfg})
    commit = cfg["fasttext"]["upstream_commit"]
    archive = fetch(f"https://codeload.github.com/facebookresearch/fastText/tar.gz/{commit}", p["root"] / "cache/source/fasttext.tar.gz")
    target = Path(__file__).resolve().parents[1] / "vendor/fasttext"
    if not target.exists():
        target.mkdir(parents=True)
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts[1:]
                if not parts:
                    continue
                member.name = str(Path(*parts))
                tar.extract(member, target, filter="data")
    write_json(target / "PINNED_COMMIT.json", {"commit": commit, "source_archive": str(archive)})
    write_json(p["run"] / "bootstrap_complete.json", {"shards": len(shards), "time": utc_now()})
    print(json.dumps({"event": "bootstrap_complete", "available_english_shards": len(shards)}), flush=True)


if __name__ == "__main__":
    main()
