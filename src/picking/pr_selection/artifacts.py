from __future__ import annotations

import fcntl
import json
import os
import random
import socket
import shutil
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from picking.common import digest, file_hash
from picking.storage import checkpoint_transport, verified_checkpoint_mirror
from .config import atomic_json, freeze_json, identity, implementation_identity, run_dir, verify_artifact_mount


@contextmanager
def run_lock(cfg):
    verify_artifact_mount(cfg)
    folder = run_dir(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    scratch = Path(cfg['runtime_root']) / cfg['run_id']
    scratch.mkdir(parents=True, exist_ok=True)
    with (scratch / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = {'hostname': socket.gethostname()}
        freeze_json(folder / 'OWNER.json', owner)
        freeze_json(folder / 'effective_config.json', cfg)
        yield


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu(v) for v in value)
    return value


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def freeze_implementation(cfg):
    from picking.common import REPO
    files = implementation_identity(cfg['methods'])
    folder = run_dir(cfg)
    freeze_json(folder/'source_identity.json', {'identity': identity(cfg), 'files': files})
    for relative, expected in files.items():
        target = folder/'source'/relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO/relative, target)
        if file_hash(target) != expected:
            raise RuntimeError('Frozen source differs from the numerical implementation')


class CheckpointStore:
    def __init__(self, cfg, stage, data_identity):
        self.cfg = cfg
        self.folder = run_dir(cfg) / 'pretraining' / stage
        self.folder.mkdir(parents=True, exist_ok=True)
        self.fingerprint = digest({'experiment': identity(cfg), 'stage': stage, 'data': data_identity})
        freeze_json(self.folder / 'identity.json', {'fingerprint': self.fingerprint})
        runtime = {**cfg, 'runtime_root': str(Path(cfg['runtime_root']) / cfg['run_id'])}
        self.transport = checkpoint_transport(runtime, self.folder)
        self.transport.scratch.mkdir(parents=True, exist_ok=True)
        self.runtime_cfg = runtime

    def latest(self):
        pointer = self.folder / 'latest.json'
        if not pointer.exists():
            return None
        meta = json.loads(pointer.read_text())
        if meta['fingerprint'] != self.fingerprint:
            raise RuntimeError('Checkpoint identity mismatch')
        checkpoint = self.folder / 'checkpoints' / meta['name']
        manifest = json.loads((checkpoint / 'manifest.json').read_text())
        if manifest != meta:
            raise RuntimeError('Incomplete checkpoint manifest')
        return checkpoint, meta

    def load(self):
        latest = self.latest()
        if latest is None:
            return None
        path, meta = latest
        mirror = verified_checkpoint_mirror(self.folder, path, meta['sha256'], self.runtime_cfg)
        if mirror is None:
            directory = self.transport.scratch / meta['name']
            directory.mkdir(parents=True, exist_ok=True)
            temporary = directory / 'download.tmp'
            if hasattr(self.transport, 'storage'):
                import requests
                try:
                    with requests.get(self.transport._url(path / 'state.pt'), stream=True, timeout=(15, 120)) as response:
                        response.raise_for_status()
                        with temporary.open('wb') as output:
                            for block in response.iter_content(8*1024**2):
                                output.write(block)
                except Exception:
                    raise RuntimeError('Checkpoint download failed; credential URL omitted') from None
            else:
                shutil.copyfile(path / 'state.pt', temporary)
            if temporary.stat().st_size != meta['bytes'] or file_hash(temporary) != meta['sha256']:
                raise RuntimeError('Downloaded checkpoint differs from its manifest')
            mirror = directory / 'state.pt'
            os.replace(temporary, mirror)
        state_path = mirror
        state = torch.load(state_path, map_location='cpu', weights_only=False)
        if state['fingerprint'] != self.fingerprint:
            raise RuntimeError('Checkpoint payload identity mismatch')
        return state

    def save(self, payload, step):
        name = f'step-{step:06d}'
        destination = self.folder / 'checkpoints' / name
        if destination.exists() and (destination / 'manifest.json').exists():
            raise RuntimeError('Refusing to overwrite a complete checkpoint')
        destination.mkdir(parents=True, exist_ok=True)
        local_dir = self.transport.scratch / name
        local_dir.mkdir(parents=True, exist_ok=True)
        temporary = local_dir / 'state.tmp'
        torch.save({'fingerprint': self.fingerprint, **cpu(payload)}, temporary)
        local = local_dir / 'state.pt'
        os.replace(temporary, local)
        receipt = self.transport.persist(local, destination / 'state.pt')
        meta = {'name': name, 'fingerprint': self.fingerprint, 'step': step,
                'sha256': receipt['sha256'], 'bytes': receipt['bytes']}
        atomic_json(destination / 'manifest.json', meta)
        atomic_json(self.folder / 'latest.json', meta)
        complete = sorted((self.folder / 'checkpoints').glob('step-*/manifest.json'))
        keep = self.cfg['probe'].get('keep_checkpoints', 2)
        for old in complete[:-keep]:
            # Only completed checkpoints of this new run, after publishing the
            # new latest pointer. Never touch another experiment's artifacts.
            old_meta = json.loads(old.read_text())
            if old_meta['fingerprint'] != self.fingerprint:
                raise RuntimeError('Refusing to prune a foreign checkpoint')
            if hasattr(self.transport, 'storage'):
                import requests
                try:
                    response = requests.delete(self.transport._url(old.parent/'state.pt'), timeout=(15, 120))
                    if response.status_code not in (202, 404):
                        raise RuntimeError('Checkpoint object deletion rejected')
                except Exception:
                    raise RuntimeError('Old checkpoint deletion failed; credential URL omitted') from None
                old.unlink()
                try:
                    old.parent.rmdir()
                except OSError:
                    pass  # Blobfuse can retain a stale empty-directory entry.
            else:
                shutil.rmtree(old.parent)
            scratch = self.transport.scratch / old.parent.name
            if scratch.exists():
                shutil.rmtree(scratch)
        return meta


class ScoreJournal:
    """Fsync one complete JSON row; recover only an incomplete final append."""
    def __init__(self, path, expected_identity, *, mirror_path=None, persist_every=32):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        freeze_json(self.path.with_suffix('.identity.json'), expected_identity)
        self.local = Path(mirror_path) if mirror_path is not None else self.path
        self.persist_every = persist_every
        if self.local != self.path:
            self.local.parent.mkdir(parents=True, exist_ok=True)
            freeze_json(self.local.with_suffix('.identity.json'), expected_identity)
            if not self.local.exists() and self.path.exists():
                shutil.copyfile(self.path, self.local)
        self.rows = {}
        if self.local.exists():
            with self.local.open('r+b') as stream:
                while True:
                    start = stream.tell(); line = stream.readline()
                    if not line:
                        break
                    if not line.endswith(b'\n'):
                        stream.truncate(start)
                        break
                    row = json.loads(line)
                    if row['id'] in self.rows:
                        raise RuntimeError('Duplicate score journal row')
                    self.rows[row['id']] = row
        committed = {}
        if self.local != self.path and self.path.exists():
            with self.path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    committed[row['id']] = row
            for key in set(committed) & set(self.rows):
                if committed[key] != self.rows[key]:
                    raise RuntimeError('Persistent and local score caches disagree')
            if not set(committed) <= set(self.rows):
                if not set(self.rows) <= set(committed):
                    raise RuntimeError('Divergent score journals')
                shutil.copyfile(self.path, self.local)
                self.rows = committed
        self.committed = len(committed)

    def append(self, row):
        if row['id'] in self.rows:
            raise ValueError('Duplicate score ID')
        with self.local.open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush(); os.fsync(stream.fileno())
        self.rows[row['id']] = row
        if self.local != self.path and len(self.rows)-self.committed >= self.persist_every:
            self.persist()

    def persist(self):
        if self.local == self.path or not self.local.exists() or len(self.rows) == self.committed:
            return
        temporary = self.path.with_name(self.path.name+'.upload.tmp')
        shutil.copyfile(self.local, temporary)
        os.replace(temporary, self.path)
        self.committed = len(self.rows)
