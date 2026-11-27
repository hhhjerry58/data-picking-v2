from __future__ import annotations

import gc
import json

import pandas as pd
import torch

from picking.common import file_hash
from .artifacts import CheckpointStore
from .config import atomic_json, identity, run_dir
from .modeling.qwen35 import load_engine
from .training import train_probe


def data(cfg):
    folder = run_dir(cfg) / 'data'
    if not (folder / 'COMPLETED.json').exists():
        raise ValueError('Run prepare first')
    return (pd.read_parquet(folder / 'guidance.parquet').to_dict('records'),
            pd.read_parquet(folder / 'candidates.parquet').to_dict('records'))


def probe_store(cfg):
    return CheckpointStore(cfg, 'probe', file_hash(run_dir(cfg) / 'data/guidance.parquet'))


def preflight(cfg, gpu_ids):
    starting_identity = identity(cfg)
    guidance, candidates = data(cfg)
    engine = load_engine(cfg, gpu_ids)
    checks = []
    # Reserve both FP32 Adam moment tensors, without updating M0. A backward-only
    # check without optimizer memory would understate the actual probe peak.
    optimizer_moments = [torch.zeros_like(p, device=cfg['probe']['optimizer_device'])
                         for p in engine.parameters.values() for _ in range(2)]
    # A complete, long Q example is essential: a short synthetic test cannot
    # establish full-context backward feasibility.
    samples = [('guidance', min(guidance, key=lambda x: x['input_tokens'])),
               ('guidance', max(guidance, key=lambda x: x['input_tokens']))]
    for objective, row in samples:
        # Later microbatches see all accumulated gradients already resident.
        for parameter in engine.parameters.values():
            parameter.grad = torch.zeros_like(parameter)
        before = engine.losses(row, [objective])[objective]
        backward = engine.backward(row, objective, accumulate=True)
        if abs(before-backward) > 2e-4 * max(1., abs(before)):
            raise RuntimeError('Inference and differentiable NLL disagree')
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in engine.parameters.values()):
            raise FloatingPointError('Full-model gradient failed preflight')
        checks.append({'id': row['id'], 'objective': objective, 'input_tokens': int(row['input_tokens']),
                       'forward_nll': before, 'backward_nll': backward,
                       'gpu_peak_bytes': {str(g): torch.cuda.max_memory_allocated(g) for g in gpu_ids}})
        print('PREFLIGHT', checks[-1], flush=True)
        engine.zero_grad()
    if identity(cfg) != starting_identity:
        raise RuntimeError('Implementation/environment changed during preflight')
    result = {'identity': starting_identity, 'gpu_ids': gpu_ids, 'parameter_count': sum(p.numel() for p in engine.parameters.values()),
              'device_map': engine.device_map, 'checks': checks, 'full_context_backward': True,
              'optimizer_device': cfg['probe']['optimizer_device'],
              'optimizer_moment_bytes_reserved': sum(t.numel()*t.element_size() for t in optimizer_moments)}
    atomic_json(run_dir(cfg) / 'PREFLIGHT_PASSED.json', result)
    return result


def require_preflight(cfg, gpu_ids):
    path = run_dir(cfg) / 'PREFLIGHT_PASSED.json'
    if not path.exists() or json.loads(path.read_text())['identity'] != identity(cfg):
        raise RuntimeError('This exact implementation/config must pass real-model preflight first')
    if json.loads(path.read_text())['gpu_ids'] != gpu_ids:
        raise RuntimeError('GPU topology differs from the validated preflight')


def probe(cfg, gpu_ids):
    if cfg.get('probe_from_run'):
        from .imports import import_probe
        return import_probe(cfg)
    require_preflight(cfg, gpu_ids)
    guidance, _ = data(cfg)
    store = probe_store(cfg)
    if (store.folder / 'COMPLETED.json').exists():
        return
    engine = load_engine(cfg, gpu_ids)
    train_probe(engine, guidance, store, cfg)


def score(cfg, gpu_ids):
    from .parallel import run
    return run(cfg, gpu_ids)
