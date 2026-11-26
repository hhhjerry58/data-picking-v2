from __future__ import annotations

import json
from pathlib import Path

import torch

from picking.common import file_hash
from picking.pr_selection.artifacts import CheckpointStore
from picking.pr_selection.config import identity, load_config


def parent_config(run):
    run = Path(run)
    cfg = load_config(run/'effective_config.json')
    expected = json.loads((run/'source_identity.json').read_text())['identity']
    if identity(cfg) != expected:
        raise ValueError('Parent source/environment identity differs; do not bypass its checkpoint checks')
    return cfg


def checkpoint_descriptor(run, stage):
    """Validate the original checkpoint under its ORIGINAL, unchanged config."""
    run = Path(run)
    cfg = parent_config(run)
    store = CheckpointStore(cfg, stage, file_hash(run/'data/guidance.parquet'))
    latest = store.latest()
    if latest is None:
        raise ValueError(f'Parent checkpoint missing: {stage}')
    _, metadata = latest
    local = store.transport.scratch/metadata['name']/'state.pt'
    if not local.exists():
        state = store.load()  # Original checked transport restores the local copy.
        del state
    if local.stat().st_size != metadata['bytes'] or file_hash(local) != metadata['sha256']:
        raise ValueError('Parent checkpoint bytes differ from its published manifest')
    return {**metadata, 'path': str(local), 'mtime_ns': local.stat().st_mtime_ns,
            'parent_run': str(run), 'stage': stage}


def open_verified_checkpoint(descriptor):
    """Workers reuse a checksum-verified immutable local file through a private mmap."""
    path = Path(descriptor['path'])
    stat = path.stat()
    if stat.st_size != descriptor['bytes'] or stat.st_mtime_ns != descriptor['mtime_ns']:
        raise ValueError('Verified parent checkpoint changed after validation')
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if state['fingerprint'] != descriptor['fingerprint']:
        raise ValueError('Parent checkpoint payload fingerprint differs')
    return state
