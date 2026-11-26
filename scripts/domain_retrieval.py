#!/usr/bin/env python
"""带类别语料的领域检索与等 token 继续训练实验。"""
from pathlib import Path
import argparse
import fcntl
import os
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from picking.domain_retrieval.config import load_config, run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'prepare', 'run', 'report'])
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1] / 'configs/domain_retrieval.yaml'))
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.stage == 'check':
        import json
        print(json.dumps({'run_directory': str(run_dir(cfg)), 'model': cfg['model'],
                          'target_category': cfg['data']['target_category'],
                          'save_optimizer': cfg['training']['save_optimizer'], 'auto_resume': cfg['training']['auto_resume'],
                          'training_dtype': cfg['training']['autocast_dtype'],
                          'continuation_initialization': cfg['continuation']['initialization'],
                          'prediction_tokens': cfg['continuation']['prediction_tokens'],
                          'seeds': cfg['continuation']['seeds'], 'arms': cfg['continuation']['arms']}, indent=2))
        return
    gpu_locks, gpu_ids = [], []
    if args.stage == 'run' and args.device == 'cuda':
        if not os.environ.get('DATA_PICKING_GPU_IDS', '').strip():
            parser.error('Explicitly set DATA_PICKING_GPU_IDS to allowed physical GPUs')
        from picking.runtime import free_gpu_ids
        for gpu in free_gpu_ids():
            candidate = open(f'/tmp/data-picking-domain-gpu-{gpu}.lock', 'a')
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                candidate.close()
                continue
            gpu_locks.append(candidate)
            gpu_ids.append(gpu)
        if not gpu_ids:
            parser.error('No idle GPU among DATA_PICKING_GPU_IDS; no other jobs were stopped')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0])
        print(f'Using explicitly allowed physical GPUs {gpu_ids}; probe/scoring use {gpu_ids[0]}', flush=True)
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    from picking.domain_retrieval.experiment import progress, run, run_lock
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Experiment interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    with run_lock(cfg):
        try:
            if args.stage == 'prepare':
                from picking.domain_retrieval.data import download_assets, prepare_data
                download_assets(cfg)
                prepare_data(cfg)
            elif args.stage == 'report':
                from picking.domain_retrieval.reporting import report
                print(report(cfg), flush=True)
            else:
                run(cfg, device=args.device, gpu_ids=gpu_ids)
        except KeyboardInterrupt:
            progress(cfg, 'interrupted', resumable=True)
            raise
        except Exception as error:
            progress(cfg, 'failed', error_type=type(error).__name__, error=str(error))
            raise
    for lock in gpu_locks:
        lock.close()


if __name__ == '__main__':
    main()
