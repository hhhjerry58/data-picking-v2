from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml

from picking.common import REPO, digest, file_hash
from picking.runtime import free_gpu_ids
from ..artifacts import freeze_implementation, run_lock
from ..config import atomic_json, freeze_json, identity, run_dir
from ..imports import frozen_identity
from ..modeling.qwen35 import ensure_assets, load_engine, weight_mirror
from .data import prepare
from .evaluation import evaluate
from .training import ArmStore, train


def load(path):
    path = Path(path)
    cfg = json.loads(path.read_text()) if path.suffix == '.json' else yaml.safe_load(path.read_text())
    if not re.fullmatch(r'[A-Za-z0-9_-]+', cfg['run_id']): raise ValueError('Invalid run_id')
    if cfg['analysis_root'] != '/home/v-guanhechen/experiments/data-picking': raise ValueError('Wrong analysis root')
    if cfg['objectives'] != ['answer_only', 'full_text'] or cfg['arms'] != ['top_absolute', 'top_relative', 'random', 'bottom_absolute']:
        raise ValueError('This protocol requires the eight predefined arms')
    train = cfg['training']
    if (train['steps'] != 32 or cfg['selection']['input_token_budget'] != 1000000
            or not train['save_optimizer'] or not train['auto_resume'] or train['keep_checkpoints'] < 2
            or train['normalization'] != 'global_arm_target_tokens_per_step'):
        raise ValueError('Fixed budget, updates, normalization and resume settings must be preserved')
    source = Path(cfg['artifact_root'])/'runs'/cfg['source_run']
    if 'source_identity' not in cfg:
        source_cfg, source_id = frozen_identity(source)
        completed = json.loads((source/'SCORING_COMPLETED.json').read_text())
        if completed['identity'] != source_id or completed['candidates'] != 20000:
            raise ValueError('Source scoring run is incomplete or has a different identity')
        cfg.update(model=source_cfg['model'], data=source_cfg['data'], methods=source_cfg['methods'],
                   scoring={'attention_backend': 'default'}, probe={'keep_checkpoints': train['keep_checkpoints']},
                   source_identity=source_id, entrypoint_sha256=file_hash(REPO/'scripts/pr_usefulness.py'))
    if cfg['entrypoint_sha256'] != file_hash(REPO/'scripts/pr_usefulness.py'):
        raise ValueError('Launch implementation changed')
    return cfg


def preflight(cfg, lane):
    root = run_dir(cfg)
    engine = load_engine(cfg, [0, 1])
    test = pd.read_parquet(root/'data/test.parquet').to_dict('records')
    checks = []
    longest = max(test, key=lambda row: row['input_tokens'])
    value = engine.losses(longest, ['guidance'])['guidance']
    checks.append({'kind': 'longest_T_forward', 'id': longest['id'], 'tokens': longest['input_tokens'], 'loss': value})
    arms = json.loads((root/'data/arms.json').read_text())
    for objective in cfg['objectives']:
        records = []
        for arm in arms:
            if arm['objective'] == objective:
                records.extend(pd.read_parquet(root/'data/arms'/arm['name']/'records.parquet').to_dict('records'))
        row = max(records, key=lambda r: r['input_tokens'])
        for parameter in engine.parameters.values(): parameter.grad = torch.zeros_like(parameter)
        value = engine.backward(row, objective, accumulate=True)
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in engine.parameters.values()):
            raise ValueError('Full-parameter PR backward failed')
        checks.append({'kind': 'longest_PR_backward', 'objective': objective, 'tokens': row['input_tokens'], 'loss': value})
        engine.zero_grad()
    atomic_json(root/f'PREFLIGHT_LANE_{lane}.json', {'identity': identity(cfg), 'checks': checks,
                'gpu': torch.cuda.get_device_name(0), 'peak_bytes': [torch.cuda.max_memory_allocated(i) for i in range(2)]})
    return engine


def worker(cfg, lane, lanes):
    root = run_dir(cfg)
    current = identity(cfg)
    plan = json.loads((root/'parallel.json').read_text())
    if plan['identity'] != current or plan['lanes'] != lanes: raise ValueError('Worker plan differs')
    with open(Path(cfg['runtime_root'])/cfg['run_id']/f'lane-{lane}.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        engine = preflight(cfg, lane)
        test = pd.read_parquet(root/'data/test.parquet').to_dict('records')
        test_id = file_hash(root/'data/test.parquet')
        runtime = Path(cfg['runtime_root'])/cfg['run_id']/'evaluation'
        baseline_id = {'experiment': current, 'model': 'M0', 'test': test_id, 'placement': 'two_gpu'}
        evaluate(engine, test, root/'evaluation'/f'M0-lane-{lane}', baseline_id,
                 runtime/f'M0-lane-{lane}', cfg['evaluation']['persist_rows'])
        arms = json.loads((root/'data/arms.json').read_text())[lane::lanes]
        for arm in arms:
            folder = root/'data/arms'/arm['name']
            completed = root/'arms'/arm['name']/'COMPLETED.json'
            if completed.exists():
                if json.loads(completed.read_text())['identity'] != current: raise ValueError('Completed arm identity changed')
                continue
            del engine; gc.collect(); torch.cuda.empty_cache()
            engine = load_engine(cfg, [0, 1])
            records = pd.read_parquet(folder/'records.parquet').to_dict('records')
            schedule = json.loads((folder/'batches.json').read_text())
            data_id = digest({'records': file_hash(folder/'records.parquet'), 'batches': file_hash(folder/'batches.json'), 'objective': arm['objective']})
            store = ArmStore(cfg, arm['name'], data_id)
            atomic_json(root/'arms'/arm['name']/'status.json', {'stage': 'train', 'lane': lane, 'updated_at_unix': time.time()})
            trainer = train(engine, records, schedule, store, cfg, arm['objective'])
            if trainer['step'] != cfg['training']['steps'] or trainer['records'] != arm['records']:
                raise ValueError('Arm did not complete its one-pass 32-step budget')
            atomic_json(root/'arms'/arm['name']/'status.json', {'stage': 'evaluate', 'lane': lane, 'updated_at_unix': time.time()})
            model_id = store.latest()[1]
            evaluation = evaluate(engine, test, root/'evaluation'/arm['name'],
                {'experiment': current, 'model': model_id, 'test': test_id, 'placement': 'two_gpu'},
                runtime/arm['name'], cfg['evaluation']['persist_rows'])
            if identity(cfg) != current: raise ValueError('Implementation changed during the experiment')
            atomic_json(completed, {'identity': current, 'arm': arm, 'trainer': trainer, 'evaluation': evaluation})
            store.release_local()


def launch(cfg, physical):
    root = run_dir(cfg)
    width = cfg['runtime']['gpus_per_worker']
    if width != 2 or len(physical) % width: raise ValueError('This run uses pairs of GPUs')
    lanes = min(len(physical)//width, cfg['runtime']['max_workers'])
    if lanes < 1: raise ValueError('No complete GPU pair available')
    freeze_json(root/'parallel.json', {'identity': identity(cfg), 'lanes': lanes, 'gpus_per_lane': width, 'physical_gpus': physical})
    assets = ensure_assets(cfg)
    index = json.loads((assets/'model.safetensors.index.json').read_text())['weight_map']
    for filename in sorted(set(index.values())): weight_mirror(cfg, filename)
    (root/'logs').mkdir(exist_ok=True)
    children = []
    try:
        for lane in range(lanes):
            allocated = physical[lane*width:(lane+1)*width]
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(map(str, allocated)), 'DATA_PICKING_GPU_IDS': ','.join(map(str, allocated)),
                   'PYTHONPATH': str(REPO/'src'), 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false',
                   'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
            command = [sys.executable, '-u', str(REPO/'scripts/pr_usefulness.py'), 'worker',
                       '--config', str(root/'effective_config.json'), '--lane', str(lane), '--lanes', str(lanes)]
            with (root/'logs'/f'lane-{lane}.log').open('a') as log:
                children.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children): raise RuntimeError('Continuation worker failed; inspect lane logs')
            time.sleep(2)
        if any(p.returncode for p in children): raise RuntimeError('Continuation worker failed')
    finally:
        for process in children:
            if process.poll() is None: process.terminate()
        for process in children:
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
    from .report import report
    report(cfg)


def main():
    parser = argparse.ArgumentParser(description='固定八组 PR 继续训练与 T 评测')
    parser.add_argument('stage', choices=['prepare', 'run', 'worker', 'report'])
    parser.add_argument('--config', default=str(REPO/'configs/pr_usefulness.yaml'))
    parser.add_argument('--lane', type=int, default=0); parser.add_argument('--lanes', type=int, default=2)
    args = parser.parse_args(); cfg = load(args.config)
    if args.stage == 'worker':
        worker(cfg, args.lane, args.lanes); return
    locks = []
    physical = []
    if args.stage == 'run':
        physical = [int(x) for x in os.environ.get('DATA_PICKING_GPU_IDS', '').split(',') if x.strip()]
        if len(set(physical)) != len(physical) or not physical or not set(physical) <= set(free_gpu_ids()):
            raise ValueError('Explicitly select free distinct GPUs; other jobs will not be stopped')
        for gpu in physical:
            handle = open(f'/tmp/data-picking-domain-gpu-{gpu}.lock', 'a')
            fcntl.flock(handle, fcntl.LOCK_EX|fcntl.LOCK_NB); locks.append(handle)
    with run_lock(cfg):
        freeze_implementation(cfg)
        entry = run_dir(cfg)/'source/scripts/pr_usefulness.py'
        entry.parent.mkdir(parents=True, exist_ok=True)
        if not entry.exists(): shutil.copyfile(REPO/'scripts/pr_usefulness.py', entry)
        if file_hash(entry) != cfg['entrypoint_sha256']: raise ValueError('Frozen entrypoint differs')
        status = {'stage': args.stage, 'status': 'running', 'pid': os.getpid(), 'hostname': socket.gethostname(), 'started_at_unix': time.time()}
        atomic_json(run_dir(cfg)/'status.json', status)
        try:
            if args.stage in ('prepare', 'run'): prepare(cfg)
            if args.stage == 'run': launch(cfg, physical)
            elif args.stage == 'report':
                from .report import report
                report(cfg)
        except BaseException as error:
            atomic_json(run_dir(cfg)/'status.json', {**status, 'status': 'failed', 'error': str(error), 'error_type': type(error).__name__})
            raise
        atomic_json(run_dir(cfg)/'status.json', {**status, 'status': 'completed', 'finished_at_unix': time.time()})
