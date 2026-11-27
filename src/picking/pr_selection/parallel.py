"""本地多 GPU：每卡一个纯前向副本，按固定行号分片并自动恢复。"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time

import pandas as pd
import torch

from picking.common import REPO
from .config import atomic_json, freeze_json, identity, load_config, run_dir
from .imports import import_probe, install_probe
from .modeling.qwen35 import ensure_assets, load_engine, weight_mirror
from .scoring import combine, score_model


def partition(rows, index, workers):
    if workers < 1 or not 0 <= index < workers:
        raise ValueError('Invalid worker partition')
    return rows[index::workers]


def worker(cfg, index, workers):
    root = run_dir(cfg)
    expected = identity(cfg)
    plan = json.loads((root / 'parallel.json').read_text())
    if plan['identity'] != expected or plan['workers'] != workers:
        raise ValueError('Worker configuration/partition differs from frozen plan')
    folder = root / 'workers' / f'{index:02d}'
    done = folder / 'COMPLETED.json'
    completion = {'identity': expected, 'worker': index, 'workers': workers}
    if done.exists():
        freeze_json(done, completion)
        return
    rows = pd.read_parquet(root / 'data/candidates.parquet').to_dict('records')
    assigned = partition(rows, index, workers)
    ordered = sorted(rows, key=lambda row: row['input_tokens'])
    checks = [ordered[i] for i in sorted({0, len(ordered)//2, len(ordered)-1})]
    engine = load_engine(cfg, [0])
    parity = {'gpu': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0))}
    for model in ('M0', 'M1'):
        if model == 'M1':
            install_probe(engine, cfg)
            gc.collect()
        parity[model] = [{'id': row['id'], **engine.losses(row, cfg['objectives'])} for row in checks]
        atomic_json(folder / 'parity.json', parity)
        score_model(engine, assigned, cfg, model, folder=folder, partition={'worker': index, 'workers': workers})
    if identity(cfg) != expected:
        raise ValueError('Scoring implementation changed while a worker was running')
    atomic_json(done, completion)


def run(cfg, gpu_ids):
    root = run_dir(cfg)
    if (root / 'SCORING_COMPLETED.json').exists():
        saved = json.loads((root / 'SCORING_COMPLETED.json').read_text())
        if saved['identity'] != identity(cfg):
            raise ValueError('Completed scores belong to another implementation')
        return
    if cfg.get('probe_from_run'):
        import_probe(cfg)
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(physical) != len(gpu_ids) or not physical or not physical[0]:
        raise ValueError('Explicit physical GPU allocation is required')
    freeze_json(root / 'parallel.json', {'identity': identity(cfg), 'workers': len(physical),
                                       'physical_gpus': physical, 'partition': 'row_index_modulo_workers'})
    # One coordinator prepares shared mmap files before the workers are spawned.
    assets = ensure_assets(cfg)
    index = json.loads((assets / 'model.safetensors.index.json').read_text())['weight_map']
    for filename in sorted(set(index.values())):
        weight_mirror(cfg, filename)
    children = []
    try:
        for index, gpu in enumerate(physical):
            folder = root / 'workers' / f'{index:02d}'
            folder.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'DATA_PICKING_GPU_IDS': gpu,
                   'PYTHONPATH': str(REPO / 'src'), 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'}
            command = [sys.executable, '-u', '-m', 'picking.pr_selection.parallel',
                       '--config', str(root / 'effective_config.json'), '--index', str(index), '--workers', str(len(physical))]
            with (folder / 'worker.log').open('a') as log:
                children.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError('Forward scoring worker failed; inspect workers/*/worker.log')
            time.sleep(2)
        if any(p.returncode for p in children):
            raise RuntimeError('Forward scoring worker failed')
    finally:
        for process in children:
            if process.poll() is None: process.terminate()
        for process in children:
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
    reference = None
    for index in range(len(physical)):
        check = json.loads((root / 'workers' / f'{index:02d}' / 'parity.json').read_text())
        if reference is None:
            reference = check
            continue
        if check['gpu'] != reference['gpu'] or check['capability'] != reference['capability']:
            raise ValueError('Mixed GPU architectures in one scoring run')
        for model in ('M0', 'M1'):
            expected_rows = {row['id']: row for row in reference[model]}
            for row in check[model]:
                for objective in cfg['objectives']:
                    expected = expected_rows[row['id']][objective]
                    if abs(row[objective] - expected) > 2e-5 * max(1., abs(expected)):
                        raise ValueError('Common-example loss differs across workers')
    candidates = pd.read_parquet(root / 'data/candidates.parquet').to_dict('records')
    combine(cfg, candidates, workers=len(physical), parallel=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()
    worker(load_config(args.config), args.index, args.workers)
