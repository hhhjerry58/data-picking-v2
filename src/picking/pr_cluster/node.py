"""Stage verified local inputs, run torchrun, and expose rank errors in AMLT logs."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from picking.common import REPO,digest
from .storage import BlobStore,sha256
from .protocol import recipe_identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-prefix',required=True);parser.add_argument('--local',required=True)
    parser.add_argument('--job',required=True);parser.add_argument('--mode',choices=['smoke','train'],default='train')
    args=parser.parse_args();local=Path(args.local);local.mkdir(parents=True,exist_ok=True)
    store=BlobStore();store.headers()
    manifest=store.get_json(args.bundle_prefix+'/manifest.json')
    ready=store.get_json(args.bundle_prefix+'/READY.json')
    if ready['manifest_identity']!=digest(manifest):raise ValueError('Bundle ready/manifest identities differ')
    for name,checksum in manifest['source_files'].items():
        if sha256(REPO/name)!=checksum:raise ValueError(f'Execution source differs: {name}')
    for package,expected in manifest['packages'].items():
        if version(package)!=expected:raise ValueError(f'Runtime version differs: {package}')
    cfg=manifest['protocol']
    if (cfg['world_size']!=8 or cfg['training']['steps']!=100 or cfg['training']['global_input_tokens_per_step']!=256000
            or cfg['checkpoint']['every_steps']!=5 or not cfg['checkpoint']['keep_all']
            or not cfg['checkpoint']['save_optimizer'] or not cfg['checkpoint']['auto_resume']
            or cfg['training']['gradient_normalization']!='global_step_target_token_mean'
            or cfg['evaluation']['steps']!=[0,100]):
        raise ValueError('The frozen 100-step/256k/5-step-checkpoint protocol differs')
    if torch.cuda.device_count()!=8 or any('A100' not in torch.cuda.get_device_name(i) or torch.cuda.get_device_properties(i).total_memory<75*1024**3 for i in range(8)):
        raise ValueError('This deployment requires eight A100 80GB GPUs')
    if args.job not in {job['name'] for job in manifest['jobs']}:raise ValueError('Unknown experiment')
    if args.mode=='train':
        first=manifest['jobs'][0]['name']
        smoke=store.get_json(cfg['output_prefix']+'/'+first+'-smoke/STATE.json')
        if not smoke.get('completed') or smoke.get('recipe_identity')!=recipe_identity(manifest) or smoke.get('world_size')!=8:
            raise ValueError('Run the matching eight-A100 smoke before submitting training jobs')
    (local/'manifest.json').write_text(json.dumps(manifest))
    for item in manifest['assets']:
        if item['local_path'].startswith('plans/') and item['local_path']!=f'plans/{args.job}.json':continue
        destination=local/item['local_path']
        if not destination.resolve().is_relative_to(local.resolve()):raise ValueError('Invalid staged asset path')
        print('STAGING',item['key'],item['bytes'],flush=True)
        store.download(item['key'],destination,item['sha256'],item['bytes'])
    logs=local.parent/'rank-logs';logs.mkdir(exist_ok=True)
    env={**os.environ,'DATA_PICKING_GPU_IDS':'0,1,2,3,4,5,6,7','OMP_NUM_THREADS':'2',
         'TOKENIZERS_PARALLELISM':'false','TORCH_NCCL_ASYNC_ERROR_HANDLING':'1','NCCL_DEBUG':'WARN'}
    command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=8',
             '--log-dir',str(logs),'--tee','3','-m','picking.pr_cluster.run',
             '--assets',str(local),'--job',args.job,'--mode',args.mode]
    code=subprocess.call(command,env=env)
    if code:
        for path in sorted(logs.rglob('stderr.log')):
            print('RANK_ERROR_LOG',path,flush=True)
            print('\n'.join(path.read_text(errors='replace').splitlines()[-60:]),flush=True)
        raise SystemExit(code)


if __name__=='__main__':main()
