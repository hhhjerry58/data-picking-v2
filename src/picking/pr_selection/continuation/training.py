from __future__ import annotations

import fcntl
import gc
import json
import time
from pathlib import Path

import torch

from picking.common import seed_all
from ..artifacts import CheckpointStore, restore_rng, rng_state
from ..config import atomic_json


class ArmStore(CheckpointStore):
    """Keep two persistent checkpoints, one local mirror; serialize large writes."""
    def save(self, payload, step):
        with open('/tmp/data-picking-pr-continuation-checkpoint.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            meta = super().save(payload, step)
            for path in self.transport.scratch.glob('step-*/state.pt'):
                if path.parent.name != meta['name']:
                    path.unlink()
            return meta

    def release_local(self):
        # Only mirrors belonging to this completed arm; persistent state remains.
        for path in self.transport.scratch.glob('step-*/state.pt'):
            path.unlink()


def train(engine, records, schedule, store, cfg, objective, stop_after_steps=None):
    spec = cfg['training']
    if not spec['save_optimizer'] or not spec['auto_resume']:
        raise ValueError('Optimizer state and automatic resume are mandatory')
    seed_all(spec['seed'])
    torch.set_num_threads(spec['optimizer_cpu_threads'])
    masters = {name: torch.nn.Parameter(p.detach().cpu().clone()) for name, p in engine.parameters.items()}
    optimizer = torch.optim.AdamW(masters.values(), lr=spec['learning_rate'], betas=tuple(spec['betas']),
                                 eps=spec['adam_epsilon'], weight_decay=spec['weight_decay'], foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1., (s+1)/max(1, spec['warmup_steps'])))
    cursor = {'step': 0, 'records': 0, 'input_tokens': 0, 'target_tokens': 0}
    restored = store.load()
    if restored is not None:
        if restored['schedule'] != schedule or restored['objective'] != objective:
            raise ValueError('Training schedule/objective changed on resume')
        engine.model.load_state_dict(restored['model'], strict=True)
        with torch.no_grad():
            for name, master in masters.items(): master.copy_(engine.parameters[name].detach().cpu())
        optimizer.load_state_dict(restored['optimizer']); scheduler.load_state_dict(restored['scheduler'])
        cursor = restored['cursor']; restore_rng(restored['rng']); del restored
    log = store.folder / 'train.jsonl'
    if log.exists():
        committed = []
        for line in log.read_bytes().splitlines(keepends=True):
            if not line.endswith(b'\n'): break
            row = json.loads(line)
            if row['step'] <= cursor['step']: committed.append(row)
        log.write_text(''.join(json.dumps(row)+'\n' for row in committed))
    denominator = sum(int(row[objective+'_tokens']) for row in records) / len(schedule)
    while cursor['step'] < len(schedule):
        if stop_after_steps is not None and cursor['step'] >= stop_after_steps: break
        start = time.monotonic()
        batch = [records[i] for i in schedule[cursor['step']]]
        engine.zero_grad(); total_loss = 0.
        for row in batch:
            loss = engine.backward(row, objective, denominator=denominator, accumulate=True)
            total_loss += loss * int(row[objective+'_tokens'])
        norm = torch.nn.utils.clip_grad_norm_(list(engine.parameters.values()), spec['max_grad_norm'],
                                              foreach=False, error_if_nonfinite=True)
        for name, master in masters.items():
            gradient = engine.parameters[name].grad
            if gradient is None: raise RuntimeError('Missing full-language-model parameter gradient')
            master.grad = gradient.detach().cpu()
        lr = optimizer.param_groups[0]['lr']
        optimizer.step(); scheduler.step()
        with torch.no_grad():
            for name, master in masters.items(): engine.parameters[name].copy_(master)
        optimizer.zero_grad(set_to_none=True)
        n_targets = sum(int(row[objective+'_tokens']) for row in batch)
        cursor = {'step': cursor['step']+1, 'records': cursor['records']+len(batch),
                  'input_tokens': cursor['input_tokens']+sum(int(row['input_tokens']) for row in batch),
                  'target_tokens': cursor['target_tokens']+n_targets}
        result = {**cursor, 'training_nll': total_loss/n_targets, 'objective_estimate': total_loss/denominator,
                  'learning_rate': lr, 'grad_norm': float(norm), 'seconds': time.monotonic()-start}
        with log.open('a') as stream: stream.write(json.dumps(result)+'\n'); stream.flush()
        print('TRAIN', objective, json.dumps(result), flush=True)
        done = cursor['step'] == len(schedule)
        stopped = stop_after_steps is not None and cursor['step'] >= stop_after_steps
        if done or stopped or cursor['step'] % spec['save_steps'] == 0:
            store.save({'model': engine.model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'rng': rng_state(), 'cursor': dict(cursor),
                        'schedule': schedule, 'objective': objective, 'completed': done}, cursor['step'])
    if cursor['step'] == len(schedule):
        atomic_json(store.folder / 'COMPLETED.json', {'fingerprint': store.fingerprint, **cursor})
    engine.zero_grad()
    del masters, optimizer, scheduler
    gc.collect()
    return cursor
