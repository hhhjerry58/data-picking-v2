"""Append on POSIX scratch; persist bounded snapshots and checkpoint histories."""
import json
import shutil
from pathlib import Path
from .common import paths, read_json, file_hash


class TrainingLog:
    def __init__(self, cfg, folder, completed_step, checkpoint=None):
        p = paths(cfg)
        self.persistent = folder / 'train.jsonl'
        self.local = p['scratch'] / 'training_logs' / folder.name / 'train.jsonl'
        self.local.parent.mkdir(parents=True, exist_ok=True)
        source = None
        if checkpoint is not None and (checkpoint / 'train.jsonl').exists():
            manifest = read_json(checkpoint / 'manifest.json')
            expected = manifest['files']['train.jsonl']['sha256']
            if file_hash(checkpoint / 'train.jsonl') != expected:
                raise RuntimeError('Checkpoint training history checksum mismatch')
            source = checkpoint / 'train.jsonl'
        elif self.local.exists():
            source = self.local
        elif self.persistent.exists():
            source = self.persistent
        retained = []
        if source is not None:
            for line in source.read_text().splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # Only authoritative committed steps are retained below.
                if record['step'] <= completed_step:
                    retained.append(record)
        by_step = {r['step']: r for r in retained}
        with self.local.open('w') as stream:
            for step in sorted(by_step):
                stream.write(json.dumps(by_step[step])+'\n')
        self.stream = self.local.open('a', buffering=1)

    def write(self, record):
        self.stream.write(json.dumps(record)+'\n')

    def flush(self):
        self.stream.flush()

    def persist(self):
        self.flush()
        shutil.copyfile(self.local, self.persistent)

    def close(self):
        self.persist()
        self.stream.close()
