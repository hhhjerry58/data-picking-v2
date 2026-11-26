from __future__ import annotations

import gc
import json
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM

from picking.common import digest, emit, file_hash, read_json, seed_all, utc_now
from picking.storage import checkpoint_transport, verified_checkpoint_mirror
from picking.training_log import TrainingLog
from .config import freeze_json, run_dir, runtime_cfg, write_json
from .data import model_assets


def load_model(cfg, device):
    # All experiment models start with actual pretrained weights. There is no
    # from_config/random initialization path in this experiment.
    model = AutoModelForCausalLM.from_pretrained(model_assets(cfg), local_files_only=True,
                                                dtype=torch.float32, attn_implementation='sdpa')
    model.config.use_cache = False
    return model.to(device)


def model_identity(cfg):
    path = model_assets(cfg)
    files = sorted([*path.glob('*.safetensors'), *path.glob('pytorch_model*.bin')])
    if not files:
        raise ValueError('Pretrained model weights are missing')
    # One targeted identity check on M0, used by all continuation arms.
    return {'model': cfg['model'], 'weights': {p.name: file_hash(p) for p in files},
            'config': file_hash(path / 'config.json'), 'tokenizer': file_hash(path / 'tokenizer.json')}


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(item) for item in value)
    return value


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


class TokenStream:
    """Shuffle documents by epoch; carry partial documents across updates.

    Earlier tokens remain context when a document spans a step boundary, but
    their targets are masked. No prediction token is silently discarded.
    """

    def __init__(self, rows, seed, state=None):
        self.rows = sorted(rows, key=lambda row: row['id'])
        if not self.rows or any(len(row['input_ids']) < 2 for row in self.rows):
            raise ValueError('Training needs nonempty documents with causal targets')
        self.seed = seed
        self.state = dict(state or {'epoch': 0, 'document': 0, 'offset': 0})
        self._epoch = None
        self._order = None

    def next_batch(self, budget, micro_batch_size):
        result, count = [], 0
        while len(result) < micro_batch_size and count < budget:
            if self._epoch != self.state['epoch']:
                self._epoch = self.state['epoch']
                self._order = np.random.default_rng(np.random.SeedSequence([self.seed, self._epoch])).permutation(len(self.rows))
            row = self.rows[int(self._order[self.state['document']])]
            tokens = [int(value) for value in row['input_ids']]
            offset = self.state['offset']
            take = min(len(tokens)-1-offset, budget-count)
            result.append({'id': row['id'], 'input_ids': tokens[:offset+take+1], 'loss_start': offset+1})
            count += take
            offset += take
            if offset == len(tokens)-1:
                self.state['offset'] = 0
                self.state['document'] += 1
                if self.state['document'] == len(self.rows):
                    self.state['epoch'] += 1
                    self.state['document'] = 0
            else:
                self.state['offset'] = offset
        return result, count


def batch_tensors(rows, pad_token_id, device):
    longest = max(len(row['input_ids']) for row in rows)
    ids = torch.full((len(rows), longest), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, row in enumerate(rows):
        tokens = torch.tensor(row['input_ids'], dtype=torch.long, device=device)
        length = len(tokens)
        ids[i, :length] = tokens
        attention[i, :length] = 1
        start = row.get('loss_start', 1)
        labels[i, start:length] = tokens[start:]
    return ids, attention, labels


def token_nll(logits, labels):
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                           labels[:, 1:].reshape(-1), ignore_index=-100,
                           reduction='none').reshape(labels.shape[0], -1)


def latest_checkpoint(folder, fingerprint):
    completed = []
    for path in (folder / 'checkpoints').glob('step-*'):
        marker = path / 'COMPLETED.json'
        if not marker.exists():
            continue
        manifest = read_json(path / 'manifest.json')
        if read_json(marker)['manifest_hash'] != digest(manifest) or manifest['fingerprint'] != fingerprint:
            raise RuntimeError('Checkpoint identity mismatch')
        if not manifest['save_optimizer'] or not manifest['auto_resume']:
            raise RuntimeError('Checkpoint lacks required optimizer/resume state')
        completed.append((manifest['global_step'], path, manifest))
    return max(completed, key=lambda item: item[0], default=None)


def checkpoint_payload(cfg, folder, checkpoint, manifest):
    sha = manifest['files']['state.pt']['sha256']
    transport_cfg = runtime_cfg(cfg)
    local = verified_checkpoint_mirror(folder, checkpoint, sha, transport_cfg)
    if local is None:
        local = Path(transport_cfg['runtime_root']) / 'resume' / folder.name / 'state.pt'
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkpoint / 'state.pt', local)
        if file_hash(local) != sha:
            raise RuntimeError('Persistent checkpoint differs from its manifest')
    payload = torch.load(local, map_location='cpu', weights_only=False)
    if payload['fingerprint'] != manifest['fingerprint'] or payload['trainer']['step'] != manifest['global_step']:
        raise RuntimeError('Checkpoint payload identity mismatch')
    for key in ('model', 'optimizer', 'scheduler', 'rng', 'trainer'):
        if key not in payload:
            raise RuntimeError(f'Checkpoint missing {key}')
    return payload


def save_checkpoint(cfg, folder, model, optimizer, scheduler, trainer, fingerprint, log):
    checkpoint = folder / 'checkpoints' / f'step-{trainer["step"]:06d}'
    checkpoint.mkdir(parents=True, exist_ok=True)
    transport = checkpoint_transport(runtime_cfg(cfg), folder)
    local = transport.scratch / checkpoint.name / 'state.pt'
    local.parent.mkdir(parents=True, exist_ok=True)
    payload = {'model': cpu_copy(model.state_dict()), 'optimizer': cpu_copy(optimizer.state_dict()),
               'scheduler': scheduler.state_dict(), 'trainer': dict(trainer), 'rng': rng_state(),
               'fingerprint': fingerprint}
    with local.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    del payload
    record = transport.persist(local, checkpoint / 'state.pt',
                               full_readback=not any((folder / 'checkpoint_read_records').glob('*.json')))
    log.flush()
    shutil.copyfile(log.local, checkpoint / 'train.jsonl')
    manifest = {'fingerprint': fingerprint, 'global_step': trainer['step'], 'seen_tokens': trainer['seen_tokens'],
                'save_optimizer': True, 'auto_resume': True, 'cursor': trainer['cursor'],
                'files': {'state.pt': {'size': record['bytes'], 'sha256': record['sha256']},
                          'train.jsonl': {'sha256': file_hash(checkpoint / 'train.jsonl')}}, 'time': utc_now()}
    write_json(checkpoint / 'manifest.json', manifest)
    write_json(checkpoint / 'COMPLETED.json', {'manifest_hash': digest(manifest)})
    log.persist()
    completed = sorted(path for path in (folder / 'checkpoints').glob('step-*') if (path / 'COMPLETED.json').exists())
    for old in completed[:-cfg['training']['keep_checkpoints']]:
        shutil.rmtree(old)
        mirror = transport.scratch / old.name
        if mirror.exists():
            shutil.rmtree(mirror)
    emit('domain_checkpoint', stage=folder.name, step=trainer['step'], seen_tokens=trainer['seen_tokens'])
    return checkpoint


def train(cfg, stage, documents, seed, token_budget, base_identity, device='cuda', stop_after_steps=None):
    spec = cfg['training']
    if spec['save_optimizer'] is not True or spec['auto_resume'] is not True:
        raise ValueError('Full optimizer state and automatic resume are required')
    folder = run_dir(cfg) / 'pretraining' / stage
    folder.mkdir(parents=True, exist_ok=True)
    rows = documents[['id', 'input_ids']].to_dict('records')
    for row in rows:
        row['input_ids'] = [int(value) for value in row['input_ids']]
    rows.sort(key=lambda row: row['id'])
    identity = {'initialization': 'pretrained_M0', 'base': base_identity, 'stage': stage, 'seed': seed,
                'training': spec, 'prediction_tokens': token_budget, 'data': digest(rows),
                'implementation': file_hash(__file__), 'torch': torch.__version__, 'device_type': torch.device(device).type}
    fingerprint = digest(identity)
    freeze_json(folder / 'effective_config.json', {'identity': identity, 'fingerprint': fingerprint})
    latest = latest_checkpoint(folder, fingerprint)
    if latest and latest[2]['seen_tokens'] == token_budget:
        checkpoint = latest[1]
        write_json(folder / 'TRAINING_COMPLETED.json', {'checkpoint': str(checkpoint), 'fingerprint': fingerprint,
                                                      'prediction_tokens': token_budget, 'steps': latest[0]})
        return checkpoint
    seed_all(seed)
    model = load_model(cfg, device).train()
    pad = model.config.eos_token_id
    if pad is None:
        pad = 0
    decay = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
    no_decay = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': spec['weight_decay']},
                                   {'params': no_decay, 'weight_decay': 0.}],
                                  lr=spec['learning_rate'], betas=tuple(spec['betas']),
                                  eps=spec['adam_epsilon'], foreach=False)
    steps = math.ceil(token_budget / spec['prediction_tokens_per_step'])
    def multiplier(step):
        if step < spec['warmup_steps']:
            return (step+1) / max(1, spec['warmup_steps'])
        progress = min(1., (step-spec['warmup_steps']) / max(1, steps-spec['warmup_steps']-1))
        minimum = spec['min_learning_rate'] / spec['learning_rate']
        return minimum + (1-minimum) * (1+math.cos(math.pi*progress))/2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    trainer = {'step': 0, 'seen_tokens': 0, 'cursor': {'epoch': 0, 'document': 0, 'offset': 0}}
    if latest:
        payload = checkpoint_payload(cfg, folder, latest[1], latest[2])
        model.load_state_dict(payload['model'], strict=True)
        optimizer.load_state_dict(payload['optimizer'])
        scheduler.load_state_dict(payload['scheduler'])
        trainer = payload['trainer']
        restore_rng(payload['rng'])
        del payload
        emit('domain_training_resumed', stage=stage, step=trainer['step'], seen_tokens=trainer['seen_tokens'])
    stream = TokenStream(rows, seed, trainer['cursor'])
    log = TrainingLog(runtime_cfg(cfg), folder, trainer['step'], latest[1] if latest else None)
    dtype = {'bfloat16': torch.bfloat16, 'float32': torch.float32}[spec['autocast_dtype']]
    checkpoint = latest[1] if latest else None
    try:
        while trainer['seen_tokens'] < token_budget:
            optimizer.zero_grad(set_to_none=True)
            step_tokens = min(spec['prediction_tokens_per_step'], token_budget-trainer['seen_tokens'])
            remaining, step_loss = step_tokens, 0.
            while remaining:
                batch, count = stream.next_batch(remaining, spec['micro_batch_size'])
                ids, attention, labels = batch_tensors(batch, pad, device)
                if int(labels[:, 1:].ne(-100).sum()) != count:
                    raise RuntimeError('Effective prediction-token count differs from the cursor')
                context = torch.autocast(device_type='cuda', dtype=dtype) if torch.device(device).type == 'cuda' and dtype != torch.float32 else nullcontext()
                with context:
                    logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
                    loss_sum = token_nll(logits, labels).sum()
                if not torch.isfinite(loss_sum):
                    raise RuntimeError('Non-finite training loss')
                (loss_sum / step_tokens).backward()
                step_loss += float(loss_sum.detach())
                remaining -= count
                del logits, loss_sum
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec['max_grad_norm'], error_if_nonfinite=True)
            learning_rate = optimizer.param_groups[0]['lr']
            optimizer.step()
            scheduler.step()
            trainer = {'step': trainer['step']+1, 'seen_tokens': trainer['seen_tokens']+step_tokens,
                       'cursor': dict(stream.state)}
            record = {'step': trainer['step'], 'seen_tokens': trainer['seen_tokens'], 'step_tokens': step_tokens,
                      'loss': step_loss / step_tokens, 'lr': learning_rate, 'grad_norm': float(grad_norm)}
            log.write(record)
            if trainer['step'] % 8 == 0:
                emit('domain_training_progress', stage=stage, **record)
            stopping = stop_after_steps is not None and trainer['step'] >= stop_after_steps
            if trainer['step'] % spec['save_steps'] == 0 or trainer['seen_tokens'] == token_budget or stopping:
                checkpoint = save_checkpoint(cfg, folder, model, optimizer, scheduler, trainer, fingerprint, log)
            if stopping:
                break
        if trainer['seen_tokens'] == token_budget:
            write_json(folder / 'TRAINING_COMPLETED.json', {'checkpoint': str(checkpoint), 'fingerprint': fingerprint,
                                                          'prediction_tokens': token_budget, 'steps': trainer['step']})
    finally:
        log.close()
        del model, optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return checkpoint


def restore_model(cfg, checkpoint, device='cuda'):
    checkpoint = Path(checkpoint)
    manifest = read_json(checkpoint / 'manifest.json')
    if read_json(checkpoint / 'COMPLETED.json')['manifest_hash'] != digest(manifest):
        raise RuntimeError('Incomplete model checkpoint')
    payload = checkpoint_payload(cfg, checkpoint.parent.parent, checkpoint, manifest)
    model = load_model(cfg, device)
    model.load_state_dict(payload['model'], strict=True)
    del payload
    return model.eval()


@torch.inference_mode()
def score_documents(cfg, documents, output, identity, model=None, checkpoint=None, device='cuda'):
    output = Path(output)
    marker = output.with_suffix('.complete.json')
    inputs = [{'id': row.id, 'input_ids': [int(value) for value in row.input_ids]}
              for row in documents.sort_values('id').itertuples()]
    fingerprint = digest({'model': identity, 'data': inputs, 'implementation': file_hash(__file__),
                          'torch': torch.__version__, 'numerics': 'fp32 sdpa; float64 per-document accumulation'})
    if marker.exists():
        previous = read_json(marker)
        if previous['fingerprint'] != fingerprint or previous['sha256'] != file_hash(output):
            raise RuntimeError('Cached evaluation identity mismatch')
        return pd.read_parquet(output)
    own_model = model is None
    if model is None:
        model = restore_model(cfg, checkpoint, device) if checkpoint else load_model(cfg, device)
    model.eval()
    rows = []
    ordered = sorted(inputs, key=lambda row: (-len(row['input_ids']), row['id']))
    batch_size = cfg['scoring_batch_size']
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start:start+batch_size]
        ids, attention, labels = batch_tensors(batch, model.config.eos_token_id or 0, device)
        logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
        losses = token_nll(logits, labels).sum(dim=1, dtype=torch.float64).cpu().tolist()
        counts = labels[:, 1:].ne(-100).sum(dim=1).cpu().tolist()
        for row, total, count in zip(batch, losses, counts):
            if count != len(row['input_ids'])-1 or not np.isfinite(total):
                raise RuntimeError('Scoring token count/numerics failed')
            rows.append({'id': row['id'], 'sum_nll': total, 'prediction_tokens': count, 'loss': total / count})
        if start % (batch_size*64) == 0:
            emit('domain_scoring_progress', output=output.name, documents=min(start+batch_size, len(ordered)), total=len(ordered))
    result = pd.DataFrame(rows).sort_values('id')
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    write_json(marker, {'fingerprint': fingerprint, 'sha256': file_hash(output), 'documents': len(rows)})
    if own_model:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result
