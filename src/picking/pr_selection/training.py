from __future__ import annotations

import json
import math

import numpy as np
import torch

from picking.common import seed_all
from .artifacts import rng_state, restore_rng
from .config import atomic_json


def train_probe(engine, guidance, store, cfg, *, stop_after_steps=None):
    """One fixed probe recipe; no development or test loss is consulted."""
    spec = cfg['probe']
    seed_all(spec['seed'])
    torch.set_num_threads(spec['optimizer_cpu_threads'])
    # Full FP32 master parameters and moments on CPU; every language-model
    # parameter is updated. This is optimizer offload, not frozen layers/LoRA.
    masters = {name: torch.nn.Parameter(p.detach().cpu().clone()) for name, p in engine.parameters.items()}
    optimizer = torch.optim.AdamW(masters.values(), lr=spec['learning_rate'],
                                 betas=tuple(spec['betas']), eps=spec['adam_epsilon'],
                                 weight_decay=spec['weight_decay'], foreach=False)
    warmup = spec['warmup_steps']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1., (step+1)/max(1, warmup)))
    order = []
    for epoch in range(spec['epochs']):
        order.extend(np.random.default_rng(np.random.SeedSequence([spec['seed'], epoch])).permutation(len(guidance)).tolist())
    trainer = {'step': 0, 'cursor': 0, 'seen_prediction_tokens': 0}
    state = store.load()
    if state is not None:
        if state['order'] != order:
            raise RuntimeError('Probe sampling order changed')
        engine.model.load_state_dict(state['model'], strict=True)
        with torch.no_grad():
            for name, master in masters.items():
                master.copy_(engine.parameters[name].detach().cpu())
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        trainer = state['trainer']
        restore_rng(state['rng'])
        del state
    log_path = store.folder / 'train.jsonl'
    # A kill between a log append and checkpoint can leave uncommitted rows.
    if log_path.exists():
        rows = []
        with log_path.open('rb') as stream:
            for line in stream:
                if not line.endswith(b'\n'):
                    break  # Only an uncommitted final append may be incomplete.
                if line.strip():
                    rows.append(json.loads(line))
        rows = [row for row in rows if row['step'] <= trainer['step']]
        log_path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    while trainer['cursor'] < len(order):
        if stop_after_steps is not None and trainer['step'] >= stop_after_steps:
            break
        indexes = order[trainer['cursor']:trainer['cursor']+spec['documents_per_step']]
        batch = [guidance[i] for i in indexes]
        denominator = sum(int(row['prediction_tokens']) for row in batch)
        # With uniformly sampled records, a fixed expected token denominator
        # makes each minibatch an unbiased estimator of the SAME global token
        # objective used for gQ. Variable batch lengths must not reweight Q.
        gradient_denominator = sum(int(row['prediction_tokens']) for row in guidance) * len(batch) / len(guidance)
        engine.zero_grad()
        total_loss = 0.
        for row in batch:
            loss = engine.backward(row, 'guidance', denominator=gradient_denominator, accumulate=True)
            total_loss += loss * int(row['prediction_tokens'])
        norm = torch.nn.utils.clip_grad_norm_(list(engine.parameters.values()), spec['max_grad_norm'],
                                              error_if_nonfinite=True, foreach=False)
        for name, master in masters.items():
            if engine.parameters[name].grad is None:
                raise RuntimeError(f'Missing full-model gradient: {name}')
            master.grad = engine.parameters[name].grad.detach().cpu()
        learning_rate_used = optimizer.param_groups[0]['lr']
        optimizer.step(); scheduler.step()
        with torch.no_grad():
            for name, master in masters.items():
                engine.parameters[name].copy_(master)
        optimizer.zero_grad(set_to_none=True)
        trainer['step'] += 1
        trainer['cursor'] += len(batch)
        trainer['seen_prediction_tokens'] += denominator
        row = {**trainer, 'loss': total_loss/denominator, 'grad_norm': float(norm),
               'objective_estimate': total_loss/gradient_denominator,
               'learning_rate': learning_rate_used, 'next_learning_rate': optimizer.param_groups[0]['lr']}
        with log_path.open('a') as stream:
            stream.write(json.dumps(row)+'\n'); stream.flush()
        print('PROBE_STEP', json.dumps(row), flush=True)
        done = trainer['cursor'] == len(order)
        stop = stop_after_steps is not None and trainer['step'] >= stop_after_steps
        if done or stop or trainer['step'] % spec['save_steps'] == 0:
            store.save({'model': engine.model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'rng': rng_state(), 'trainer': dict(trainer),
                        'order': order, 'completed': done}, trainer['step'])
    if trainer['cursor'] == len(order):
        atomic_json(store.folder / 'COMPLETED.json', {'fingerprint': store.fingerprint, **trainer})
    engine.zero_grad()
    return trainer
