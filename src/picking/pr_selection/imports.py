"""按历史源码快照和训练指纹导入 M1，不改历史身份。"""
from __future__ import annotations

import json
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from picking.common import digest, file_hash
from .config import atomic_json, identity, run_dir

PACKAGES = ('torch', 'transformers', 'tokenizers', 'flash-linear-attention',
            'accelerate', 'triton', 'tilelang', 'causal-conv1d')


def frozen_identity(source):
    source = Path(source)
    cfg = json.loads((source / 'effective_config.json').read_text())
    recorded = json.loads((source / 'source_identity.json').read_text())
    for relative, expected in recorded['files'].items():
        path = source / 'source' / relative
        if not path.resolve().is_relative_to((source / 'source').resolve()) or file_hash(path) != expected:
            raise ValueError('Historical numerical source snapshot differs')
    versions = {}
    for name in PACKAGES:
        try: versions[name] = version(name)
        except PackageNotFoundError: versions[name] = None
    actual = digest({'config': cfg, 'implementation': recorded['files'], 'versions': versions})
    if actual != recorded['identity']:
        raise ValueError('Historical config/source/runtime identity differs')
    return cfg, actual


def verified_payload(descriptor):
    path = Path(descriptor['path'])
    stat = path.stat()
    if stat.st_size != descriptor['bytes'] or stat.st_mtime_ns != descriptor['mtime_ns']:
        raise ValueError('Imported checkpoint mirror changed after verification')
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    if state['fingerprint'] != descriptor['fingerprint'] or not state.get('completed'):
        raise ValueError('Imported M1 checkpoint payload identity/completion differs')
    if not all(key in state for key in ('optimizer', 'scheduler', 'rng')):
        raise ValueError('Source training checkpoint lacks complete optimizer/resume state')
    return state


def import_probe(cfg):
    receipt = run_dir(cfg) / 'imported_probe.json'
    current_identity = identity(cfg)
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['destination_identity'] != current_identity:
            raise ValueError('Imported probe belongs to another scoring identity')
        if file_hash(run_dir(cfg) / 'data/guidance.parquet') != saved['guidance_sha256']:
            raise ValueError('Imported M1 used a different Q dataset')
        descriptor = saved['checkpoint']
        path = Path(descriptor['path'])
        if path.exists() and path.stat().st_size == descriptor['bytes'] and path.stat().st_mtime_ns == descriptor['mtime_ns']:
            return saved
    source = Path(cfg['artifact_root']) / 'runs' / cfg['probe_from_run']
    original, source_identity = frozen_identity(source)
    if original['probe'] != cfg['probe'] or original['model'] != cfg['model']:
        raise ValueError('Imported M1 training recipe/model differs from requested probe')
    guidance = file_hash(source / 'data/guidance.parquet')
    if file_hash(run_dir(cfg) / 'data/guidance.parquet') != guidance:
        raise ValueError('Imported M1 used a different Q dataset')
    stage = source / 'pretraining/probe'
    meta = json.loads((stage / 'latest.json').read_text())
    completed = json.loads((stage / 'COMPLETED.json').read_text())
    expected = digest({'experiment': source_identity, 'stage': 'probe', 'data': guidance})
    checkpoint = stage / 'checkpoints' / meta['name']
    if (meta['fingerprint'] != expected or completed['fingerprint'] != expected
            or completed['step'] != meta['step']
            or json.loads((checkpoint / 'manifest.json').read_text()) != meta):
        raise ValueError('Historical probe checkpoint fingerprint/manifest differs')
    record = stage / 'checkpoint_read_records' / (meta['name'] + '.json')
    mirror = None
    if record.exists():
        previous = json.loads(record.read_text())
        candidate = Path(previous['runtime_copy'])
        if previous['sha256'] == meta['sha256'] and candidate.is_file():
            mirror = candidate
    if mirror is None:
        mirror = Path(cfg['runtime_root']) / cfg['run_id'] / 'imports' / 'm1-state.pt'
        mirror.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkpoint / 'state.pt', mirror)
    if mirror.stat().st_size != meta['bytes'] or file_hash(mirror) != meta['sha256']:
        raise ValueError('Historical M1 bytes differ from committed checkpoint')
    descriptor = {**meta, 'path': str(mirror), 'mtime_ns': mirror.stat().st_mtime_ns}
    state = verified_payload(descriptor)
    if state['trainer']['step'] != meta['step']:
        raise ValueError('Historical M1 trainer state differs')
    del state
    saved = {'destination_identity': current_identity, 'source_run': str(source),
             'source_identity': source_identity, 'guidance_sha256': guidance, 'checkpoint': descriptor}
    atomic_json(receipt, saved)
    return saved


def model_identity(cfg, model):
    if model == 'M0':
        return {'name': cfg['model']['name'], 'revision': cfg['model']['revision']}
    if cfg.get('probe_from_run'):
        saved = json.loads((run_dir(cfg) / 'imported_probe.json').read_text())
        if saved['destination_identity'] != identity(cfg):
            raise ValueError('Imported M1 scoring identity changed')
        return {key: saved['checkpoint'][key] for key in ('sha256', 'fingerprint', 'step')}
    return json.loads((run_dir(cfg) / 'pretraining/probe/latest.json').read_text())


def install_probe(engine, cfg):
    if cfg.get('probe_from_run'):
        model_identity(cfg, 'M1')
        saved = json.loads((run_dir(cfg) / 'imported_probe.json').read_text())
        state = verified_payload(saved['checkpoint'])
    else:
        from .artifacts import CheckpointStore
        store = CheckpointStore(cfg, 'probe', file_hash(run_dir(cfg) / 'data/guidance.parquet'))
        state = store.load()
        if not state or not state.get('completed'):
            raise ValueError('Complete the fixed probe before scoring M1')
    engine.model.load_state_dict(state['model'], strict=True)
