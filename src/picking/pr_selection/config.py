from __future__ import annotations

import json
import os
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

from picking.common import REPO, digest, file_hash

OBJECTIVES = ('full_text', 'answer_only')
METHODS = ('absolute_loss_drop', 'relative_loss_drop', 'grad_dot')


def load_config(path):
    path = Path(path)
    # YAML 1.1 treats JSON scientific notation such as 2e-06 as a string.
    # Frozen JSON must round-trip with the JSON parser, including tiny eps/LR.
    cfg = json.loads(path.read_text()) if path.suffix.lower() == '.json' else yaml.safe_load(path.read_text())
    if not re.fullmatch(r'[A-Za-z0-9_-]+', cfg['run_id']):
        raise ValueError('run_id must be a single directory name')
    for key in ('artifact_root', 'runtime_root', 'analysis_root'):
        if not Path(cfg[key]).is_absolute():
            raise ValueError(f'{key} must be absolute')
    if cfg['analysis_root'] != '/home/v-guanhechen/experiments':
        raise ValueError('Training analyses belong in /home/v-guanhechen/experiments')
    if cfg['objectives'] != list(OBJECTIVES) or not set(METHODS) <= set(cfg['methods']):
        raise ValueError('Phase one requires the paired objectives and all three core scores')
    if len(cfg['methods']) != len(set(cfg['methods'])):
        raise ValueError('Repeated scoring method')
    for name in cfg['methods']:
        if not re.fullmatch(r'[a-z][a-z0-9_]*', name) or not (REPO/'methods'/name/'method.py').is_file():
            raise ValueError(f'No valid method directory: {name}')
    if cfg['probe']['save_optimizer'] is not True or cfg['probe']['auto_resume'] is not True:
        raise ValueError('Optimizer state and automatic resume are mandatory')
    if cfg['probe']['optimizer_device'] != 'cpu' or cfg['probe']['gradient_normalization'] != 'global_q_token_mean':
        raise ValueError('The validated full-parameter recipe uses CPU Adam and global-Q normalization')
    if cfg['probe']['epochs'] < 1 or cfg['probe']['documents_per_step'] < 1:
        raise ValueError('Invalid probe budget')
    if not 0 < cfg['data']['q_tasks'] < cfg['data']['expected_tasks']:
        raise ValueError('Q and T must both contain tasks')
    if cfg['data']['candidates'] < 1 or cfg['data']['max_source_shards'] < 1:
        raise ValueError('Candidate pool must be nonempty and bounded')
    if cfg['model']['max_length'] > 262144:
        raise ValueError('This protocol does not enable context extension')
    if cfg['model']['parameter_dtype'] != 'float32':
        raise ValueError('Full-parameter probe uses FP32 parameters/optimizer and BF16 compute')
    if cfg['model']['compute_dtype'] != 'bfloat16' or cfg['model']['gradient_checkpointing'] is not True:
        raise ValueError('Production backend requires BF16 compute and activation checkpointing')
    for key in ('lm_head_chunk_tokens', 'activation_offload_min_tokens'):
        if cfg['model'][key] < 1:
            raise ValueError(f'Invalid model.{key}')
    if cfg['probe']['save_steps'] < 1 or cfg['probe']['keep_checkpoints'] < 2:
        raise ValueError('Require positive checkpoint interval and at least two retained checkpoints')
    if cfg['scoring']['gradient_save_documents'] < 1 or cfg['scoring']['journal_persist_rows'] < 1:
        raise ValueError('Scoring persistence intervals must be positive')
    if cfg.get('prepared_from_run') and not re.fullmatch(r'[A-Za-z0-9_-]+', cfg['prepared_from_run']):
        raise ValueError('Invalid prepared-data source run')
    return cfg


def run_dir(cfg):
    return Path(cfg['artifact_root']) / 'runs' / cfg['run_id']


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f'Identity changed at {path}; use a new run_id')
    else:
        atomic_json(path, value)


def implementation_identity(methods=METHODS):
    # These hashes protect numerical/cache/checkpoint identity, not a file inventory.
    files = sorted(Path(__file__).parent.rglob('*.py'))
    files += [REPO / 'methods' / name / 'method.py' for name in methods]
    files += [REPO/'scripts/pr_selection.py', REPO/'requirements-pr.txt', REPO/'src/picking/common.py']
    return {str(p.relative_to(REPO)): file_hash(p) for p in files}


def identity(cfg):
    versions = {}
    for package in ('torch', 'transformers', 'tokenizers', 'flash-linear-attention', 'accelerate',
                    'triton', 'tilelang', 'causal-conv1d'):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return digest({'config': cfg, 'implementation': implementation_identity(cfg['methods']), 'versions': versions})


def verify_artifact_mount(cfg):
    root = Path(cfg['artifact_root'])
    if str(root).startswith('/mnt/tianzhu_hot/') and not os.path.ismount('/mnt/tianzhu_hot'):
        raise RuntimeError('tianzhu_hot is not mounted; refusing to write into the local mount directory')
