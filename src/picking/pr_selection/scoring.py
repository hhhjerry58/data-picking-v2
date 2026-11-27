"""M0/M1 纯前向评分，两个监督区域共享每个模型的主体前向。"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from picking.common import REPO, file_hash
from .artifacts import ScoreJournal
from .config import OBJECTIVES, atomic_json, identity, run_dir
from .imports import model_identity


def method(name):
    path = REPO / 'methods' / name / 'method.py'
    spec = importlib.util.spec_from_file_location('pr_method_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_identity(cfg, model, objective, partition):
    return {'identity': identity(cfg), 'objective': objective, 'model': model,
            'checkpoint': model_identity(cfg, model), 'partition': partition,
            'candidates': file_hash(run_dir(cfg) / 'data/candidates.parquet')}


def score_model(engine, candidates, cfg, model, *, folder=None, partition=None):
    folder = Path(folder) if folder else run_dir(cfg)
    part = partition or {'worker': 0, 'workers': 1}
    label = 'loss_before' if model == 'M0' else 'loss_after'
    journals = {obj: ScoreJournal(folder / 'scores' / obj / (model.lower() + '.jsonl'),
        score_identity(cfg, model, obj, part),
        mirror_path=Path(cfg['runtime_root']) / cfg['run_id'] / 'score_journals' / str(part['worker']) / obj / (model.lower() + '.jsonl'),
        persist_every=cfg['scoring']['journal_persist_rows']) for obj in OBJECTIVES}
    try:
        for i, row in enumerate(candidates):
            missing = [obj for obj in OBJECTIVES if row['id'] not in journals[obj].rows]
            if not missing:
                continue
            start = time.monotonic()
            losses = engine.losses(row, missing)
            if set(losses) != set(missing) or not np.isfinite(list(losses.values())).all():
                raise FloatingPointError('Invalid forward loss')
            for obj in missing:
                journals[obj].append({'id': row['id'], label: losses[obj],
                                     'prediction_tokens': int(row[obj + '_tokens'])})
            elapsed = time.monotonic() - start
            print('SCORE', model, part['worker'], i + 1, len(candidates), elapsed, flush=True)
            if (i + 1) % 16 == 0 or i + 1 == len(candidates):
                atomic_json(folder / 'progress.json', {'model': model, 'completed': i + 1,
                            'records': len(candidates), 'last_seconds': elapsed, 'updated_at_unix': time.time()})
    finally:
        for journal in journals.values():
            journal.persist()


def score_m0(engine, candidates, cfg):
    score_model(engine, candidates, cfg, 'M0')


def score_m1(engine, candidates, cfg):
    score_model(engine, candidates, cfg, 'M1')


def combine(cfg, candidates, workers=1, parallel=False):
    root = run_dir(cfg)
    metadata = pd.DataFrame([{key: row[key] for key in ('id', 'query_id', 'source_category', 'format',
            'input_tokens', 'full_text_tokens', 'answer_only_tokens')} for row in candidates])
    expected = set(metadata.id)
    if len(expected) != len(candidates):
        raise ValueError('Duplicate candidate IDs')
    frames = []
    for objective in OBJECTIVES:
        tables = {}
        for model in ('M0', 'M1'):
            records = []
            for index in range(workers):
                folder = root / 'workers' / f'{index:02d}' if parallel else root
                part = {'worker': index, 'workers': workers}
                path = folder / 'scores' / objective / (model.lower() + '.jsonl')
                if json.loads(path.with_suffix('.identity.json').read_text()) != score_identity(cfg, model, objective, part):
                    raise ValueError('Score journal source/partition identity differs')
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                assigned = {r['id'] for r in candidates[index::workers]}
                if {r['id'] for r in rows} != assigned:
                    raise ValueError('Scoring shard coverage differs')
                records.extend(rows)
            table = pd.DataFrame(records)
            if table.id.duplicated().any() or set(table.id) != expected:
                raise ValueError('Incomplete or repeated candidate scores')
            tables[model] = table
        paired = tables['M0'].merge(tables['M1'], on=['id', 'prediction_tokens'], validate='one_to_one')
        if len(paired) != len(candidates):
            raise ValueError('M0/M1 supervision tokens differ')
        for name in cfg['methods']:
            paired[name] = method(name).score(paired.to_dict('records'), seed=cfg['data']['sample_seed'])
            destination = root / 'methods' / name / objective
            destination.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'id': paired.id, 'score': paired[name]}).to_parquet(destination / 'scores.parquet', index=False)
        paired = paired.merge(metadata, on='id', validate='one_to_one')
        paired['objective'] = objective
        frames.append(paired)
    result = pd.concat(frames, ignore_index=True)
    if not np.isfinite(result[[*cfg['methods'], 'loss_before', 'loss_after']].to_numpy()).all():
        raise FloatingPointError('Nonfinite paired scores')
    (root / 'scores').mkdir(exist_ok=True)
    result.to_parquet(root / 'scores/paired_scores.parquet', index=False)
    atomic_json(root / 'SCORING_COMPLETED.json', {'identity': identity(cfg), 'candidates': len(candidates),
                'objectives': list(OBJECTIVES), 'methods': cfg['methods'], 'distributions': 4, 'test_losses_computed': False})
    return result
