"""Direct Azure blob transfers with retries, checksums and renewable run leases."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import requests


def sha256(path):
    value = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''): value.update(chunk)
    return value.hexdigest()


class BlobStore:
    def __init__(self, account='msranlpinternhot', container='tianzhu'):
        self.endpoint = f'https://{account}.blob.core.windows.net'
        self.container = container
        self.sas = os.environ.get('DATA_PICKING_BLOB_SAS', '').lstrip('?')
        configuration = os.environ.get('DATA_PICKING_BLOB_CONFIG')
        if configuration:
            import yaml
            try: storage = yaml.safe_load(Path(configuration).read_text())['azstorage']
            except Exception: raise RuntimeError('Cannot read the external storage credential configuration; contents omitted') from None
            if storage['container'] != container: raise ValueError('Storage container differs')
            self.endpoint = storage['endpoint'].rstrip('/')
            self.sas = storage['sas'].lstrip('?')
        self.credential = None
        self.token = None
        self.token_lock = threading.Lock()
        self.local = threading.local()

    def headers(self):
        headers = {'x-ms-version': '2021-12-02'}
        if not self.sas:
            with self.token_lock:
                if self.token is None or self.token.expires_on < time.time()+300:
                    try:
                        if self.credential is None:
                            from azure.identity import DefaultAzureCredential
                            self.credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)
                        self.token = self.credential.get_token('https://storage.azure.com/.default')
                    except Exception:
                        raise RuntimeError('Azure storage authentication failed; configure the job managed identity or DATA_PICKING_BLOB_SAS outside the code bundle') from None
                headers['Authorization'] = 'Bearer ' + self.token.token
        return headers

    def request(self, method, key, *, query='', data=None, headers=None, accepted=(200,), attempts=7):
        if not key.startswith('data-picking-v2/') or '..' in Path(key).parts:
            raise ValueError('Blob request outside the authorized project prefix')
        url = self.endpoint+'/'+quote(self.container)+'/'+quote(key, safe='/')
        params = '&'.join(part for part in [self.sas, query] if part)
        if params: url += '?' + params
        if not hasattr(self.local, 'session'): self.local.session = requests.Session()
        reason = 'unknown'
        for attempt in range(attempts):
            try:
                response = self.local.session.request(method, url, data=data,
                    headers={**self.headers(), **(headers or {})}, timeout=(15, 60))
                if response.status_code in accepted: return response
                reason = f'HTTP {response.status_code}'
                if response.status_code in (401, 403, 404, 409, 412):
                    raise ValueError(f'Blob operation failed for {key}: {reason}; credentials and URL omitted')
            except ValueError as error:
                if str(error).startswith('Blob operation failed for '): raise
                reason = type(error).__name__
            except Exception as error:
                reason = type(error).__name__
            if attempt+1 < attempts:
                print('BLOB_RETRY', method, key, attempt+1, reason, flush=True)
                time.sleep(min(2**attempt, 15))
        raise RuntimeError(f'Blob operation failed for {key}: {reason}; credentials and URL omitted')

    def get_json(self, key):
        return self.request('GET', key).json()

    def put_json(self, key, value, *, lease=None, create_only=False):
        headers = {'x-ms-blob-type': 'BlockBlob', 'Content-Type': 'application/json'}
        if lease: headers['x-ms-lease-id'] = lease
        if create_only: headers['If-None-Match'] = '*'
        return self.request('PUT', key, data=(json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode(),
                            headers=headers, accepted=(201,))

    def download(self, key, destination, checksum, size=None):
        destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and (size is None or destination.stat().st_size == size) and sha256(destination) == checksum:
            return
        head = self.request('HEAD', key)
        actual_size = int(head.headers['Content-Length']); etag = head.headers['ETag']
        if size is not None and actual_size != size: raise ValueError('Blob size differs from manifest')
        temporary = destination.with_name(destination.name+'.partial')
        block = 8*1024**2
        fd = os.open(temporary, os.O_CREAT|os.O_TRUNC|os.O_WRONLY, 0o600)
        os.ftruncate(fd, actual_size)
        def part(start):
            end = min(actual_size, start+block)-1
            response = self.request('GET', key, headers={'Range': f'bytes={start}-{end}', 'If-Match': etag}, accepted=(206,))
            if len(response.content) != end-start+1: raise ValueError('Incomplete blob range')
            os.pwrite(fd, response.content, start)
        try:
            with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(part, range(0, actual_size, block)))
            os.fsync(fd)
        finally: os.close(fd)
        if sha256(temporary) != checksum: raise ValueError('Downloaded blob checksum differs')
        os.replace(temporary, destination)

    def upload(self, path, key, *, assert_owner=lambda: None):
        path = Path(path); size = path.stat().st_size; checksum = sha256(path)
        block = 8*1024**2; starts = list(range(0, size, block))
        prefix = uuid.uuid4().hex[:12]
        ids = [base64.b64encode(f'{prefix}-{i:08d}'.encode()).decode() for i in range(len(starts))]
        fd = os.open(path, os.O_RDONLY)
        def part(i):
            assert_owner()
            data = os.pread(fd, min(block, size-starts[i]), starts[i])
            md5 = base64.b64encode(hashlib.md5(data).digest()).decode()
            self.request('PUT', key, query='comp=block&blockid='+quote(ids[i], safe=''), data=data,
                         headers={'Content-MD5': md5}, accepted=(201,))
        try:
            with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(part, range(len(starts))))
        finally: os.close(fd)
        assert_owner()
        listing = ('<BlockList>'+''.join('<Latest>'+identifier+'</Latest>' for identifier in ids)+'</BlockList>').encode()
        self.request('PUT', key, query='comp=blocklist', data=listing,
                     headers={'x-ms-meta-sha256': checksum}, accepted=(201,))
        head = self.request('HEAD', key)
        if int(head.headers['Content-Length']) != size or head.headers.get('x-ms-meta-sha256') != checksum:
            raise ValueError('Committed checkpoint size/hash metadata differs')
        return {'key': key, 'bytes': size, 'sha256': checksum, 'etag': head.headers['ETag']}


class RunLease:
    """Only the current lease holder can publish STATE.json / checkpoint pointers."""
    def __init__(self, store, key, identity, heartbeat=None):
        self.store, self.key, self.identity = store, key, identity
        self.identifier = str(uuid.uuid4()); self.attempt = uuid.uuid4().hex
        self.updated = 0.; self.failure = None; self.stop = threading.Event()
        self.heartbeat = Path(heartbeat) if heartbeat else None

    def _mark(self):
        if self.heartbeat:
            self.heartbeat.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.heartbeat.with_suffix('.tmp')
            temporary.write_text(json.dumps({'attempt': self.attempt, 'updated': time.time()}))
            os.replace(temporary, self.heartbeat)

    def __enter__(self):
        try: self.store.put_json(self.key, {'identity': self.identity, 'latest': None, 'checkpoints': []}, create_only=True)
        except ValueError as error:
            if 'HTTP 412' not in str(error): raise
        state = self.store.get_json(self.key)
        if state['identity'] != self.identity: raise ValueError('Run identity changed; use a new run')
        self.store.request('PUT', self.key, query='comp=lease', headers={'x-ms-lease-action': 'acquire',
            'x-ms-proposed-lease-id': self.identifier, 'x-ms-lease-duration': '60'}, accepted=(201,))
        self.updated = time.monotonic(); self.state = self.store.get_json(self.key)
        self._mark()
        self.thread = threading.Thread(target=self._renew, daemon=True); self.thread.start()
        return self

    def _renew(self):
        while not self.stop.wait(10):
            try:
                self.store.request('PUT', self.key, query='comp=lease', headers={'x-ms-lease-action': 'renew',
                                   'x-ms-lease-id': self.identifier}, accepted=(200,), attempts=1)
                self.updated = time.monotonic()
                self._mark()
            except Exception as error:
                self.failure = type(error).__name__; return

    def check(self):
        if self.failure or time.monotonic()-self.updated > 35:
            raise RuntimeError('Run lease is no longer fresh; refusing to publish')

    def publish(self, state):
        self.check()
        self.store.put_json(self.key, state, lease=self.identifier)
        self.state = state

    def __exit__(self, *args):
        self.stop.set(); self.thread.join(timeout=2)
        try: self.store.request('PUT', self.key, query='comp=lease', headers={'x-ms-lease-action': 'release',
                               'x-ms-lease-id': self.identifier}, accepted=(200,), attempts=1)
        except Exception: pass


def checked_mount_copy(source, destination, checksum, size, attempts=5):
    """Retry transient FUSE EIO; publish only a complete checksum-verified copy."""
    import shutil
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == size and sha256(destination) == checksum: return
    temporary = destination.with_name(destination.name+'.copy.tmp')
    for attempt in range(attempts):
        try:
            with source.open('rb') as src, temporary.open('wb') as out: shutil.copyfileobj(src, out, length=8*1024**2)
            if temporary.stat().st_size != size or sha256(temporary) != checksum:
                raise OSError('Mounted input is incomplete or differs from its manifest')
            os.replace(temporary, destination); return
        except OSError:
            temporary.unlink(missing_ok=True)
            if attempt+1 == attempts: raise
            print('MOUNT_COPY_RETRY', source.name, attempt+1, flush=True)
            time.sleep(min(2**attempt, 15))
