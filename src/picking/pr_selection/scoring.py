from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from picking.common import REPO, file_hash
from .artifacts import CheckpointStore, ScoreJournal
from .config import METHODS, OBJECTIVES, atomic_json, identity, run_dir


def method(name):
    path = REPO / 'methods' / name / 'method.py'
    spec = importlib.util.spec_from_file_location('pr_method_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_m0(engine, guidance, candidates, cfg):
    folder = run_dir(cfg)
    data_id = file_hash(folder / 'data/guidance.parquet')
    candidate_identity = file_hash(folder / 'data/candidates.parquet')
    grad_method = method('grad_dot')
    journals = {objective: ScoreJournal(folder / 'scores' / objective / 'm0.jsonl',
                {'identity': identity(cfg), 'objective': objective, 'model': 'M0',
                 'guidance': data_id, 'candidates': candidate_identity},
                 mirror_path=Path(cfg['runtime_root'])/cfg['run_id']/'scores'/objective/'m0.jsonl',
                 persist_every=cfg['scoring']['journal_persist_rows']) for objective in OBJECTIVES}
    if all(len(j.rows) == len(candidates) for j in journals.values()):
        for journal in journals.values():
            journal.persist()
        return
    store = CheckpointStore(cfg, 'target-gradient', data_id)
    reference = grad_method.prepare(engine, guidance, store, cfg)
    for i, record in enumerate(candidates):
        for objective in OBJECTIVES:
            if record['id'] in journals[objective].rows:
                continue
            result = grad_method.score_example(engine, record, objective, reference)
            journals[objective].append({'id': record['id'], 'loss_before': result['loss'],
                'gradient_dot': result['gradient_dot'], 'gradient_norm': result['gradient_norm'],
                'prediction_tokens': int(record[objective+'_tokens'])})
            print('M0_SCORE', i+1, len(candidates), objective, record['id'], result, flush=True)
    for journal in journals.values():
        journal.persist()
    engine.zero_grad()


def score_m1(engine, candidates, cfg):
    folder = run_dir(cfg)
    candidate_identity = file_hash(folder / 'data/candidates.parquet')
    checkpoint_identity = json.loads((folder / 'pretraining/probe/latest.json').read_text())
    journals = {objective: ScoreJournal(folder / 'scores' / objective / 'm1.jsonl',
                {'identity': identity(cfg), 'objective': objective, 'model': 'M1',
                 'checkpoint': checkpoint_identity, 'candidates': candidate_identity},
                 mirror_path=Path(cfg['runtime_root'])/cfg['run_id']/'scores'/objective/'m1.jsonl',
                 persist_every=cfg['scoring']['journal_persist_rows']) for objective in OBJECTIVES}
    for i, record in enumerate(candidates):
        missing = [objective for objective in OBJECTIVES if record['id'] not in journals[objective].rows]
        if not missing:
            continue
        losses = engine.losses(record, missing)
        for objective in missing:
            journals[objective].append({'id': record['id'], 'loss_after': losses[objective],
                                       'prediction_tokens': int(record[objective+'_tokens'])})
        print('M1_SCORE', i+1, len(candidates), record['id'], losses, flush=True)
    for journal in journals.values():
        journal.persist()


def combine(cfg, candidates):
    folder = run_dir(cfg)
    starting_identity = identity(cfg)
    candidate_identity = file_hash(folder / 'data/candidates.parquet')
    expected = {r['id'] for r in candidates}
    metadata = pd.DataFrame([{k: row[k] for k in ('id', 'query_id', 'source_category', 'format',
                            'input_tokens', 'full_text_tokens', 'answer_only_tokens')} for row in candidates])
    frames = []
    for objective in OBJECTIVES:
        for model in ('m0', 'm1'):
            metadata_path = folder/'scores'/objective/(model+'.identity.json')
            provenance = json.loads(metadata_path.read_text())
            if provenance['identity'] != starting_identity or provenance['candidates'] != candidate_identity:
                raise RuntimeError('Score cache provenance differs from current implementation/data')
        before = pd.read_json(folder / 'scores' / objective / 'm0.jsonl', lines=True)
        after = pd.read_json(folder / 'scores' / objective / 'm1.jsonl', lines=True)
        for table in (before, after):
            if table.id.duplicated().any() or set(table.id) != expected:
                raise ValueError('Incomplete or inconsistent candidate score coverage')
        paired = before.merge(after, on=['id', 'prediction_tokens'], validate='one_to_one')
        if len(paired) != len(expected):
            raise ValueError('Before/after prediction masks changed')
        records = paired.to_dict('records')
        for name in cfg['methods']:
            scores = np.asarray(method(name).score(records, seed=cfg['data']['sample_seed']), dtype=float)
            if len(scores) != len(records) or not np.isfinite(scores).all():
                raise FloatingPointError(f'Invalid scores: {name}/{objective}')
            paired[name] = scores
            dest = folder / 'methods' / name / objective
            dest.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'id': paired.id, 'score': scores}).to_parquet(dest / 'scores.parquet', index=False)
        paired = paired.merge(metadata, on='id', validate='one_to_one')
        paired['objective'] = objective
        frames.append(paired)
    result = pd.concat(frames, ignore_index=True)
    if identity(cfg) != starting_identity:
        raise RuntimeError('Implementation/environment changed while publishing scores')
    result.to_parquet(folder / 'scores/paired_scores.parquet', index=False)
    atomic_json(folder / 'SCORING_COMPLETED.json', {'identity': identity(cfg), 'candidates': len(expected),
                                                  'objectives': list(OBJECTIVES), 'methods': cfg['methods'],
                                                  'distributions': len(OBJECTIVES)*len(cfg['methods']), 'test_losses_computed': False})
    return result
