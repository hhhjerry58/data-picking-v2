"""Self-contained copy of the previously validated checkpoint blob transport.

Only transport was moved into this repository. Model/optimizer serialization and
the frozen pretraining implementation are unchanged. Credentials are supplied by
the active mount configuration, never by checked-in files.
"""
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import requests

from .common import file_hash, write_json


class DirectCheckpointReader:
    def _url(self, path):
        storage = self.storage
        return (storage['endpoint'].rstrip('/') + '/' +
                urllib.parse.quote(storage['container']) + '/' +
                urllib.parse.quote(str(Path(path).relative_to(self.mount)), safe='/') +
                '?' + storage['sas'].lstrip('?'))

    def _head(self, url):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=45) as response:
                return {'etag': response.headers['ETag'], 'bytes': int(response.headers['Content-Length'])}
        except Exception:
            raise RuntimeError('Checkpoint blob HEAD failed; URL omitted') from None

    def persist(self,local,destination,verify_only=False,full_readback=False):
        local,destination=Path(local),Path(destination)
        if not destination.is_relative_to(self.root) or destination.name!='state.pt':raise ValueError('Checkpoint scope violation')
        size=local.stat().st_size;sha=file_hash(local);block=8*1024**2
        indexes=list(range(0,size,block));url=self._url(destination)
        ids=[base64.b64encode(f'{i:08d}'.encode()).decode() for i in range(len(indexes))]
        sessions=threading.local()
        def request(target,data=None,headers=None,method=None):
            if not hasattr(sessions,'http'):sessions.http=requests.Session()
            reason='unknown'
            for attempt in range(8):
                try:
                    with sessions.http.request(method or 'GET',target,data=data,headers=headers or {},timeout=(15,90)) as r:
                        if r.status_code>=400:
                            try:azure_code=ET.fromstring(r.content).findtext('Code') or 'unknown'
                            except Exception:azure_code='unknown'
                            reason=f'HTTP {r.status_code} / {azure_code}'
                            raise RuntimeError('Blob operation failed')
                        return r.status_code,r.content,dict(r.headers)
                except Exception as exc:
                    if not isinstance(exc,RuntimeError):reason=type(exc).__name__
                    print('CHECKPOINT_TRANSPORT_RETRY',destination.parent.name,attempt+1,reason,flush=True)
                    if attempt==7:raise RuntimeError(f'Checkpoint blob transport failed ({reason}); URL omitted') from None
                    time.sleep(min(2**attempt,15))
        expected=[]
        fd=os.open(local,os.O_RDONLY)
        def upload(i):
            data=os.pread(fd,min(block,size-indexes[i]),indexes[i])
            content_sha=hashlib.sha256(data).hexdigest()
            if verify_only:return content_sha
            md5=base64.b64encode(hashlib.md5(data).digest()).decode()
            status,_,headers=request(url+'&comp=block&blockid='+urllib.parse.quote(ids[i],safe=''),data,
                {'x-ms-version':'2021-12-02','Content-MD5':md5},'PUT')
            if status!=201:raise RuntimeError('Block upload was not accepted')
            returned=headers.get('Content-MD5',headers.get('content-md5'))
            if returned is not None and returned!=md5:raise RuntimeError('Server block MD5 differs')
            return content_sha
        try:
            with ThreadPoolExecutor(max_workers=64) as pool:
                for i,value in enumerate(pool.map(upload,range(len(indexes))),1):
                    expected.append(value)
                    if i%512==0:print('CHECKPOINT_LOCAL_BLOCK_HASHES' if verify_only else 'CHECKPOINT_BLOCKS_UPLOADED',destination.parent.name,i,len(indexes),flush=True)
        finally:os.close(fd)
        if not verify_only:
            body=('<BlockList>'+''.join('<Latest>'+i+'</Latest>' for i in ids)+'</BlockList>').encode()
            status,_,_=request(url+'&comp=blocklist',body,{'x-ms-version':'2021-12-02','x-ms-meta-sha256':sha},'PUT')
            if status!=201:raise RuntimeError('Checkpoint block-list commit failed')
        head=self._head(url)
        if head['bytes']!=size:raise RuntimeError('Committed checkpoint size mismatch')
        if not verify_only:
            _,listing,_=request(url+'&comp=blocklist&blocklisttype=committed')
            actual=[(b.findtext('Name'),int(b.findtext('Size'))) for b in ET.fromstring(listing).findall('./CommittedBlocks/Block')]
            wanted=[(ids[i],min(block,size-start)) for i,start in enumerate(indexes)]
            if actual!=wanted:raise RuntimeError('Committed checkpoint block order/size mismatch')
        def verify(i):
            start=indexes[i];end=min(size,start+block)-1
            status,data,_=request(url,headers={'Range':f'bytes={start}-{end}','If-Match':head['etag']})
            if status!=206 or len(data)!=end-start+1 or hashlib.sha256(data).hexdigest()!=expected[i]:
                raise RuntimeError('Checkpoint range SHA-256 verification failed')
        if verify_only or full_readback:
            with ThreadPoolExecutor(max_workers=32) as pool:
                for i,_ in enumerate(pool.map(verify,range(len(indexes))),1):
                    if i%512==0:print('CHECKPOINT_BLOCKS_VERIFIED',destination.parent.name,i,len(indexes),flush=True)
        if self._head(url)!=head:raise RuntimeError('Checkpoint changed during readback')
        if file_hash(local)!=sha:raise RuntimeError('Local checkpoint changed during persistence')
        record={'source':str(destination),'runtime_copy':str(local),'sha256':sha,'bytes':size,'etag':head['etag'],
            'adapter_sha256':file_hash(__file__),
            'verified_at_utc':datetime.now(timezone.utc).isoformat(),
            'full_remote_readback':bool(verify_only or full_readback),
            'transport':'Server-validated Content-MD5 per uploaded block; committed block order and sizes; stable ETag; local full-file SHA-256. Full If-Match range readback when full_remote_readback is true.'}
        write_json(local.with_suffix('.verified.json'),record)
        self.written_copies[destination]=local
        write_json(self.run/'checkpoint_read_records'/f'{destination.parent.name}.json',record)
        return record
