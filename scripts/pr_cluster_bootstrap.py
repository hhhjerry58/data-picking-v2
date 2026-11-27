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


def local_directory(roots, *, prefix, min_free_gib, executable=False):
    """分别选择可执行 runtime 和大容量数据目录；不能只检查剩余空间。"""
    failures=[]
    for root in map(Path,roots):
        directory=None
        try:
            if executable and os.statvfs(root).f_flag & os.ST_NOEXEC:
                raise PermissionError('filesystem is mounted noexec')
            free=shutil.disk_usage(root).free
            if free < min_free_gib*1024**3:
                raise OSError(f'need {min_free_gib} GiB free, found {free/1024**3:.1f} GiB')
            directory=Path(tempfile.mkdtemp(prefix=prefix,dir=root)).resolve()
            if executable:
                # 实际 exec 可同时发现挂载标志之外的执行限制；不能只依赖 chmod/access。
                probe=directory/'exec-probe'
                try:
                    shutil.copyfile('/bin/true',probe)
                    probe.chmod(0o700)
                    subprocess.run([str(probe)],check=True,capture_output=True,timeout=10)
                finally:
                    probe.unlink(missing_ok=True)
            return directory
        except (OSError,subprocess.SubprocessError) as error:
            if directory is not None:shutil.rmtree(directory)
            failures.append(f'{root}: {error}')
            print('BOOTSTRAP_ROOT_SKIPPED',root,str(error),flush=True)
    purpose='executable runtime' if executable else 'staging/checkpoint'
    raise RuntimeError(f'No usable {purpose} directory: '+ '; '.join(failures))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True);parser.add_argument('--job',required=True)
    parser.add_argument('--runtime-root',type=Path,help='Existing executable local filesystem with at least 32 GiB free')
    parser.add_argument('--staging-root',type=Path,help='Existing local filesystem with at least 160 GiB free; noexec is allowed')
    parser.add_argument('--mode',choices=['smoke','train'],default='train');args=parser.parse_args()
    bundle=Path(args.bundle)
    ready=json.loads(retry_read(bundle/'READY.json'));manifest=json.loads(retry_read(bundle/'manifest.json'))
    if bool(manifest.get('smoke_only')) != (args.mode == 'smoke'):
        raise ValueError('Smoke and 100k training bundles must use their matching launch mode')
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    if ready['manifest_identity']!=identity:raise ValueError('Incomplete or mismatched bundle publication')
    runtime_roots=[args.runtime_root] if args.runtime_root else ['/scratch','/tmp','/dev/shm']
    execution=local_directory(runtime_roots,prefix='pr-runtime-',min_free_gib=32,executable=True)
    print('BOOTSTRAP_EXECUTION_DIR',execution,flush=True)
    for name,target in [('runtime.tar.gz','runtime'),('code.tar.gz','code')]:
        archive=execution/name;copy_archive(bundle/name,archive,manifest['archives'][name])
        directory=execution/target;directory.mkdir()
        subprocess.run(['tar','-xzf',str(archive),'-C',str(directory)],check=True);archive.unlink()
        print('ARCHIVE_VERIFIED',name,flush=True)
    runtime=execution/'runtime';code=execution/'code';python=runtime/'python/bin/python3.12'
    temporary=execution/'tmp';temporary.mkdir()
    env={**os.environ,'PYTHONPATH':':'.join(str(runtime/p) for p in ['extra','overlay','base'])+':'+str(code/'src'),
         'PYTHONNOUSERSITE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'2',
         'TMPDIR':str(temporary),'TRITON_CACHE_DIR':str(execution/'triton-cache'),
         'TORCH_EXTENSIONS_DIR':str(execution/'torch-extensions'),
         'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','DATA_PICKING_GPU_IDS':'0,1,2,3,4,5,6,7'}
    env.setdefault('CUDA_HOME','/usr/local/cuda');env['PATH']=env['CUDA_HOME']+'/bin:'+env.get('PATH','')
    subprocess.run([str(python),'-c','import ctypes, ssl; print("BOOTSTRAP_RUNTIME_READY", flush=True)'],env=env,check=True)
    # runtime 解压后再检查容量，两个目录落在同一文件系统时也保留 160 GiB。
    staging_roots=[args.staging_root] if args.staging_root else ['/dev/shm','/scratch','/tmp']
    local=local_directory(staging_roots,prefix='pr-training-',min_free_gib=160)
    print('BOOTSTRAP_STAGING_DIR',local,flush=True)
    command=[str(python),'-u','-m','picking.pr_cluster.node','--bundle-prefix',manifest['protocol']['bundle_prefix'],
             '--local',str(local/'assets'),'--job',args.job,'--mode',args.mode]
    raise SystemExit(subprocess.call(command,env=env))


if __name__=='__main__':main()
