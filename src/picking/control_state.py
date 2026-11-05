"""Control-plane state must remain usable when the artifact mount fails.

Only status/stop/watchdog records use this spool. Dataset and checkpoint
completion markers still require the normal persistent-write guarantees.
"""
from __future__ import annotations

import json
import os
import socket
import uuid
from pathlib import Path

from .common import REPO, paths, utc_now


def control_directories(cfg):
    return (paths(cfg)['scratch']/'control_state'/cfg['run_id'],
            REPO/'.runtime'/cfg['run_id']/'control_state')


def atomic_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w') as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_control(cfg, name, record):
    if Path(name).name != name or not name.endswith('.json'):
        raise ValueError('A control record must be a JSON basename')
    payload = {**record, 'hostname':socket.gethostname(), 'control_written_at_utc': utc_now()}
    successes = []
    last_error = None
    for directory in control_directories(cfg):
        try:
            atomic_json(directory/name, payload)
            successes.append(directory/name)
        except OSError as error:
            last_error = error
    if not successes:
        raise last_error or RuntimeError('No writable control-state spool')
    return payload


def read_control(cfg, name):
    candidates = []
    for directory in control_directories(cfg):
        try:
            candidates.append(json.loads((directory/name).read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    if candidates:
        return max(candidates, key=lambda row: row.get('control_written_at_utc', row.get('time', '')))
    try:
        return json.loads((paths(cfg)['run']/name).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def stop_requested(cfg):
    for directory in control_directories(cfg):
        try:
            if (directory/'STOP_REQUESTED.json').exists():
                return True
        except OSError:
            pass
    try:
        return (paths(cfg)['run']/'STOP_REQUESTED.json').exists()
    except OSError:
        return False


def process_alive(pid, script_name, hostname=None):
    if hostname is not None and hostname != socket.gethostname():
        return False
    try:
        process = Path('/proc')/str(int(pid))
        args = (process/'cmdline').read_bytes().split(b'\0')
        stat = (process/'stat').read_text().rsplit(')', 1)[1].split()
        return (stat[0] != 'Z' and len(args) > 1 and Path(args[1].decode()).name == script_name
                and (process/'cwd').resolve() == REPO)
    except (OSError, ValueError, TypeError):
        return False


def mirror_control(cfg):
    """Best-effort persistent copies, called only by the separate mirror job."""
    names = set()
    for directory in control_directories(cfg):
        try:
            names.update(path.name for path in directory.glob('*.json'))
        except OSError:
            continue
    copied, failed = [], []
    for name in sorted(names):
        record = read_control(cfg, name)
        if record is None:
            continue
        try:
            target = paths(cfg)['run']/name
            try:
                if json.loads(target.read_text()) == record:
                    continue
            except (OSError, json.JSONDecodeError):
                pass
            with target.open('w') as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
            copied.append(name)
        except OSError:
            failed.append(name)
    return {'copied': copied, 'failed': failed}
