from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml

from picking.common import REPO, digest


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    if not re.fullmatch(r'[A-Za-z0-9_-]+', cfg['run_id']):
        raise ValueError('run_id must be a single directory name')
    for key in ('artifact_root', 'runtime_root', 'analysis_root'):
        if not Path(cfg[key]).is_absolute():
            raise ValueError(f'{key} must be absolute')
    if Path(cfg['analysis_root']) != Path('/home/v-guanhechen/experiments'):
        raise ValueError('Analyses belong in /home/v-guanhechen/experiments')
    spec = cfg['training']
    if spec['save_optimizer'] is not True or spec['auto_resume'] is not True:
        raise ValueError('save_optimizer and auto_resume must both be true')
    if cfg['continuation']['initialization'] != 'M0':
        raise ValueError('The paired continuation experiment starts from M0')
    data = cfg['data']
    if not 1 <= data['min_prediction_tokens'] < data['max_length']:
        raise ValueError('Invalid token length bounds')
    for field in ('probe_documents', 'positive_documents', 'negative_documents',
                  'target_test_documents', 'other_test_documents'):
        if data[field] < 1:
            raise ValueError(f'{field} must be positive')
    if not 0 < data['near_duplicate_jaccard'] <= 1:
        raise ValueError('Invalid near duplicate threshold')
    if not 0 < cfg['continuation']['selected_documents'] <= min(
            data['positive_documents'], data['negative_documents']):
        raise ValueError('Both controls must contain the requested document count')
    for field in ('prediction_tokens_per_step', 'micro_batch_size', 'save_steps', 'keep_checkpoints'):
        if spec[field] < 1:
            raise ValueError(f'{field} must be positive')
    for phase in ('probe', 'continuation'):
        if cfg[phase]['prediction_tokens'] < 1:
            raise ValueError('Prediction-token budgets must be positive')
    if not cfg['continuation']['seeds'] or len(set(cfg['continuation']['seeds'])) != len(cfg['continuation']['seeds']):
        raise ValueError('Use distinct continuation seeds')
    if not {'oracle_target', 'other'} <= set(cfg['continuation']['arms']):
        raise ValueError('Both domain and other-data controls are required')
    if len(cfg['continuation']['arms']) != len(set(cfg['continuation']['arms'])):
        raise ValueError('Duplicate continuation arms could create concurrent writers')
    if not set(cfg['continuation']['arms']) <= set(cfg['methods']) | {'oracle_target', 'other'}:
        raise ValueError('Continuation arm has no configured selection method')
    if len(cfg['methods']) != len(set(cfg['methods'])):
        raise ValueError('Duplicate methods')
    for name in cfg['methods']:
        method_file(name)
    return cfg


def method_file(name):
    if not re.fullmatch(r'[a-z][a-z0-9_]*', name):
        raise ValueError('Invalid method name')
    path = REPO / 'methods' / name / 'method.py'
    if not path.is_file():
        raise ValueError(f'Missing method: {path}')
    return path


def run_dir(cfg):
    return Path(cfg['artifact_root']) / 'runs' / cfg['run_id']


def runtime_cfg(cfg):
    # Existing checkpoint transport only needs these three keys. Each run gets
    # its own scratch namespace, including when stage/arm names coincide.
    return {**cfg, 'runtime_root': str(Path(cfg['runtime_root']) / cfg['run_id'])}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if digest(json.loads(path.read_text())) != digest(value):
            raise RuntimeError(f'Existing experiment identity differs: {path}')
    else:
        write_json(path, value)
