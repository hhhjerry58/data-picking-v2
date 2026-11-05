"""Self-contained checkpoint transport and optional node-local weight mirrors."""
from pathlib import Path
import os
import shutil
import yaml
from .common import paths, read_json, file_hash, write_json, utc_now
from .blob_transport import DirectCheckpointReader


def mount_configuration(mount):
    explicit = os.environ.get('DATA_PICKING_BLOB_CONFIG')
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise RuntimeError('DATA_PICKING_BLOB_CONFIG does not point to a readable file')
        return path
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            args = (process / 'cmdline').read_bytes().split(b'\0')
            if not args or b'blobfuse' not in args[0] or str(mount).encode() not in args:
                continue
            for index, arg in enumerate(args):
                value = arg.split(b'=', 1)[1] if arg.startswith(b'--config-file=') else (
                    args[index+1] if arg == b'--config-file' and index+1 < len(args) else None)
                if value:
                    path = Path(value.decode())
                    if not path.is_absolute():
                        path = (process / 'cwd').resolve() / path
                    return path
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    raise RuntimeError('Cannot find configuration for the authorized persistent mount')


class FilesystemCheckpointTransport:
    def persist(self, local, destination, full_readback=False):
        local, destination = Path(local), Path(destination)
        if not destination.is_relative_to(self.root) or destination.name != 'state.pt':
            raise ValueError('Checkpoint scope violation')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with local.open('rb') as source, destination.open('wb') as output:
            shutil.copyfileobj(source, output, length=8*1024**2)
            output.flush()
            os.fsync(output.fileno())
        sha = file_hash(local)
        if local.stat().st_size != destination.stat().st_size or file_hash(destination) != sha:
            raise RuntimeError('Persistent checkpoint readback differs')
        record = {'source':str(destination),'runtime_copy':str(local),'sha256':sha,'bytes':local.stat().st_size,
                  'verified_at_utc':utc_now(),'full_remote_readback':False,
                  'transport':'Filesystem copy, fsync, close, size and full SHA-256 readback through the persistent filesystem'}
        write_json(local.with_suffix('.verified.json'),record)
        write_json(self.run/'checkpoint_read_records'/(destination.parent.name+'.json'),record)
        return record


def checkpoint_transport(cfg, folder):
    p = paths(cfg)
    folder = Path(folder)
    if not folder.is_relative_to(p['run'] / 'pretraining'):
        raise ValueError('Checkpoint transport is restricted to this experiment')
    mode = os.environ.get('DATA_PICKING_CHECKPOINT_IO', 'blob')
    if mode not in ('blob','filesystem'):
        raise ValueError('DATA_PICKING_CHECKPOINT_IO must be blob or filesystem')
    reader = DirectCheckpointReader() if mode == 'blob' else FilesystemCheckpointTransport()
    reader.root = folder / 'checkpoints'
    reader.run = folder
    reader.mount = p['root'].parent
    if mode == 'blob':
        settings = yaml.safe_load(mount_configuration(reader.mount).read_text())
        reader.storage = settings['azstorage']
        if not all(reader.storage.get(key) for key in ('endpoint','container','sas')):
            raise RuntimeError('Direct blob transport requires endpoint/container/SAS in the external mount config; use DATA_PICKING_CHECKPOINT_IO=filesystem for other mounts')
    reader.scratch = p['scratch'] / 'checkpoint_mirrors' / folder.name
    reader.workers = 32
    reader.written_copies = {}
    reader.cfg = cfg
    return reader


def verified_checkpoint_mirror(folder, checkpoint, expected_sha256, cfg):
    record = Path(folder) / 'checkpoint_read_records' / (Path(checkpoint).name + '.json')
    if not record.exists():
        return None
    saved = read_json(record)
    path = Path(saved['runtime_copy'])
    if not path.is_relative_to(paths(cfg)['scratch']):
        return None  # A mirror recorded on another node is only an optional cache.
    if saved['sha256'] == expected_sha256 and path.exists() and file_hash(path) == expected_sha256:
        return path
    return None
