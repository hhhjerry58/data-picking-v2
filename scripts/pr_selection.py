#!/usr/bin/env python
"""PR 评分第一阶段：prepare → preflight → probe → score → analyze。"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from picking.pr_selection.config import load_config, run_dir, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'prepare', 'preflight', 'probe', 'score', 'analyze'])
    parser.add_argument('--config', default='configs/pr_selection.yaml')
    parser.add_argument('--run-id')
    parser.add_argument('--prepared-from-run', help='Reuse an identical immutable prepared dataset from another run')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.run_id:
        import re
        if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_id):
            parser.error('Invalid run-id')
        cfg['run_id'] = args.run_id
    if args.prepared_from_run:
        import re
        if not re.fullmatch(r'[A-Za-z0-9_-]+', args.prepared_from_run) or args.prepared_from_run == cfg['run_id']:
            parser.error('Invalid prepared-data source run')
        cfg['prepared_from_run'] = args.prepared_from_run
    if args.stage == 'check':
        print(json.dumps({'run_directory': str(run_dir(cfg)), 'model': cfg['model'],
                          'q_tasks': cfg['data']['q_tasks'], 'test_tasks': cfg['data']['expected_tasks']-cfg['data']['q_tasks'],
                          'candidates': cfg['data']['candidates'], 'pr_subsets': cfg['data']['pr_subsets'],
                          'probe': cfg['probe'], 'distributions': len(cfg['methods'])*len(cfg['objectives']), 'test_losses': False}, indent=2))
        return
    locks, gpu_ids = [], []
    if args.stage in ('preflight', 'score') or (args.stage == 'probe' and not cfg.get('probe_from_run')):
        raw = os.environ.get('DATA_PICKING_GPU_IDS', '')
        if not raw.strip():
            parser.error('Set DATA_PICKING_GPU_IDS explicitly')
        allowed = [int(x) for x in raw.split(',')]
        if len(set(allowed)) != len(allowed):
            parser.error('Repeated GPU ID')
        from picking.runtime import free_gpu_ids
        available = set(free_gpu_ids())
        if not set(allowed) <= available:
            parser.error('Some requested GPUs are occupied; no other task was stopped')
        for gpu in allowed:
            lock = open(f'/tmp/data-picking-domain-gpu-{gpu}.lock', 'a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, allowed))
        gpu_ids = list(range(len(allowed)))
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('CUDA_HOME', '/usr/local/cuda-12.8')
    os.environ['PATH'] = os.environ['CUDA_HOME'] + '/bin:' + os.environ.get('PATH', '')
    from picking.pr_selection.artifacts import run_lock, freeze_implementation
    with run_lock(cfg):
        status = {'stage': args.stage, 'status': 'running', 'pid': os.getpid(),
                  'started_at': datetime.now(timezone.utc).isoformat()}
        atomic_json(run_dir(cfg)/'status.json', status)
        try:
            if args.stage == 'prepare':
                from picking.pr_selection.data.prepare import prepare
                from picking.pr_selection.modeling.qwen35 import load_tokenizer
                prepare(cfg, load_tokenizer(cfg))
            elif args.stage == 'analyze':
                from picking.pr_selection.diagnostics import analyze
                print(analyze(cfg))
            else:
                freeze_implementation(cfg)
                from picking.pr_selection import workflow
                getattr(workflow, args.stage)(cfg, gpu_ids)
        except BaseException as error:
            atomic_json(run_dir(cfg)/'status.json', {**status, 'status': 'failed',
                        'error_type': type(error).__name__, 'error': str(error), 'resumable': True})
            raise
        atomic_json(run_dir(cfg)/'status.json', {**status, 'status': 'completed',
                    'finished_at': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
