#!/usr/bin/env python3
"""AMLT 初始 Python 只负责重试下载并校验便携环境，再进入固定运行时。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def retry_read(path):
    for attempt in range(6):
        try:return Path(path).read_bytes()
        except OSError:
            if attempt==5:raise
            print('BOOTSTRAP_READ_RETRY',Path(path).name,attempt+1,flush=True)
            time.sleep(min(2**attempt,15))


def checksum(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for data in iter(lambda:stream.read(8*1024**2),b''):h.update(data)
    return h.hexdigest()


def copy_archive(source,target,expected):
    for attempt in range(6):
        try:
            with open(source,'rb') as src,open(target,'wb') as out:shutil.copyfileobj(src,out,8*1024**2)
            if target.stat().st_size!=expected['bytes'] or checksum(target)!=expected['sha256']:
                raise OSError('Archive size/checksum mismatch')
            return
        except OSError:
            target.unlink(missing_ok=True)
            if attempt==5:raise
            print('BOOTSTRAP_COPY_RETRY',source.name,attempt+1,flush=True)
            time.sleep(min(2**attempt,15))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True);parser.add_argument('--job',required=True)
    parser.add_argument('--mode',choices=['smoke','train'],default='train');args=parser.parse_args()
    bundle=Path(args.bundle)
    ready=json.loads(retry_read(bundle/'READY.json'));manifest=json.loads(retry_read(bundle/'manifest.json'))
    if bool(manifest.get('smoke_only')) != (args.mode == 'smoke'):
        raise ValueError('Smoke and 100k training bundles must use their matching launch mode')
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    if ready['manifest_identity']!=identity:raise ValueError('Incomplete or mismatched bundle publication')
    roots=[Path('/dev/shm'),Path('/tmp')]
    parent=next((root for root in roots if shutil.disk_usage(root).free>160*1024**3),None)
    if parent is None:raise RuntimeError('Need at least 160 GiB of local staging/checkpoint space')
    local=Path(tempfile.mkdtemp(prefix='pr-training-',dir=parent))
    for name,target in [('runtime.tar.gz','runtime'),('code.tar.gz','code')]:
        archive=local/name;copy_archive(bundle/name,archive,manifest['archives'][name])
        directory=local/target;directory.mkdir()
        subprocess.run(['tar','-xzf',str(archive),'-C',str(directory)],check=True);archive.unlink()
        print('ARCHIVE_VERIFIED',name,flush=True)
    runtime=local/'runtime';code=local/'code';python=runtime/'python/bin/python3.12'
    env={**os.environ,'PYTHONPATH':':'.join(str(runtime/p) for p in ['extra','overlay','base'])+':'+str(code/'src'),
         'PYTHONNOUSERSITE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'2',
         'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','DATA_PICKING_GPU_IDS':'0,1,2,3,4,5,6,7'}
    env.setdefault('CUDA_HOME','/usr/local/cuda');env['PATH']=env['CUDA_HOME']+'/bin:'+env.get('PATH','')
    command=[str(python),'-u','-m','picking.pr_cluster.node','--bundle-prefix',manifest['protocol']['bundle_prefix'],
             '--local',str(local/'assets'),'--job',args.job,'--mode',args.mode]
    raise SystemExit(subprocess.call(command,env=env))


if __name__=='__main__':main()
