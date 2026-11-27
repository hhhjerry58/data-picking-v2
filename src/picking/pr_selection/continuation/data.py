from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from picking.common import file_hash
from ..config import atomic_json
from ..data.splits import task_id
from ..modeling.masks import render_trajectory
from ..modeling.qwen35 import load_tokenizer


def select(frame, arm, budget, seed):
    if arm == 'random':
        ordered = frame.iloc[np.random.default_rng(seed).permutation(len(frame))]
    else:
        column = 'relative_loss_drop' if arm == 'top_relative' else 'absolute_loss_drop'
        ordered = frame.sort_values([column, 'id'], ascending=[arm == 'bottom_absolute', True])
    count = int(np.searchsorted(ordered.input_tokens.cumsum().to_numpy(), budget, side='right'))
    if count == 0:
        raise ValueError('Token budget cannot hold a complete PR')
    return ordered.iloc[:count].copy()


def batches(records, steps, seed):
    """Shuffle once, then cut near equal cumulative input-token quantiles."""
    if len(records) < steps or steps < 1:
        raise ValueError('Need at least one complete PR per optimizer update')
    order = np.random.default_rng(seed).permutation(len(records)).tolist()
    totals = np.cumsum([int(records[i]['input_tokens']) for i in order])
    cuts = [0]
    for step in range(1, steps):
        goal = totals[-1] * step / steps
        lo, hi = cuts[-1] + 1, len(order) - (steps - step)
        closest = int(np.searchsorted(totals, goal)) + 1
        options = {max(lo, min(hi, closest)), max(lo, min(hi, closest - 1))}
        cuts.append(min(options, key=lambda i: (abs(totals[i-1] - goal), i)))
    cuts.append(len(order))
    return [order[a:b] for a, b in zip(cuts, cuts[1:])]


def prepare(cfg):
    root = Path(cfg['artifact_root']) / 'runs' / cfg['run_id']
    source = Path(cfg['artifact_root']) / 'runs' / cfg['source_run']
    target = root / 'data'
    target.mkdir(parents=True, exist_ok=True)
    source_data = json.loads((source / 'data/COMPLETED.json').read_text())
    data_identity = {'scores': file_hash(source / 'scores/paired_scores.parquet'),
                     'candidates': file_hash(source / 'data/candidates.parquet'),
                     'split_manifest': file_hash(source / 'data/split_manifest.csv'),
                     'trajectory_sha256': file_hash(cfg['data']['trajectory_file'])}
    if data_identity['trajectory_sha256'] != source_data['trajectory_sha256']:
        raise ValueError('Original trajectory dataset changed')
    marker = target / 'COMPLETED.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['source_files'] != data_identity:
            raise ValueError('Prepared experiment inputs changed')
        for item in saved['files']:
            if file_hash(root / item['path']) != item['sha256']:
                raise ValueError('Prepared arm/test data changed')
        return saved
    frame = pd.read_parquet(source / 'scores/paired_scores.parquet')
    candidates = pd.read_parquet(source / 'data/candidates.parquet').set_index('id', drop=False)
    splits = pd.read_csv(source / 'data/split_manifest.csv')
    wanted = splits[splits.split == 'T'].set_index('id')
    if len(wanted) != cfg['evaluation']['expected_records'] or wanted.task_id.nunique() != cfg['evaluation']['expected_tasks']:
        raise ValueError('Frozen T record/task counts differ')
    arms = []
    for objective in cfg['objectives']:
        group = frame[frame.objective == objective]
        if group.id.duplicated().any() or set(group.id) != set(candidates.index):
            raise ValueError('Scoring candidate coverage differs')
        for arm in cfg['arms']:
            chosen = select(group, arm, cfg['selection']['input_token_budget'], cfg['selection']['random_seed'])
            rows = candidates.loc[chosen.id].reset_index(drop=True)
            records = rows.to_dict('records')
            schedule = batches(records, cfg['training']['steps'], cfg['training']['seed'])
            name = objective + '-' + arm
            path = target / 'arms' / name
            path.mkdir(parents=True, exist_ok=True)
            rows.to_parquet(path / 'records.parquet', index=False)
            atomic_json(path / 'batches.json', schedule)
            supervised = int(rows[objective + '_tokens'].sum())
            batch_inputs = [sum(records[i]['input_tokens'] for i in batch) for batch in schedule]
            arms.append({'name': name, 'objective': objective, 'selection': arm, 'records': len(rows),
                         'input_tokens': int(rows.input_tokens.sum()), 'target_tokens': supervised,
                         'steps': len(schedule), 'min_step_input_tokens': min(batch_inputs),
                         'max_step_input_tokens': max(batch_inputs), 'epochs': 1})
    atomic_json(target / 'arms.json', arms)
    tokenizer = load_tokenizer(cfg)
    test = []
    with open(cfg['data']['trajectory_file']) as stream:
        for line in stream:
            if not line.strip(): continue
            row = json.loads(line)
            if row['id'] not in wanted.index:
                continue
            task = task_id(row)
            if task != wanted.loc[row['id'], 'task_id']:
                raise ValueError('Test task identity differs from frozen split')
            encoded = render_trajectory(tokenizer, row)
            if encoded['input_tokens'] > cfg['model']['max_length']:
                raise ValueError('T exceeds the native context; do not truncate')
            test.append({**encoded, 'task_id': task})
    if len(test) != len(wanted) or {row['id'] for row in test} != set(wanted.index):
        raise ValueError('Incomplete T rendering')
    pd.DataFrame(test).sort_values('id').to_parquet(target / 'test.parquet', index=False)
    paths = [target/'test.parquet', target/'arms.json', *sorted((target/'arms').glob('*/*'))]
    result = {'source_files': data_identity, 'arms': arms, 'test_records': len(test),
              'test_tasks': int(wanted.task_id.nunique()), 'test_max_input_tokens': max(r['input_tokens'] for r in test),
              'test_input_tokens': sum(r['input_tokens'] for r in test),
              'test_target_tokens': sum(r['prediction_tokens'] for r in test),
              'files': [{'path': str(p.relative_to(root)), 'sha256': file_hash(p)} for p in paths]}
    atomic_json(marker, result)
    return result
