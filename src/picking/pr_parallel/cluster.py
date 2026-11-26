from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from safetensors import safe_open

from picking.common import digest, file_hash
from picking.pr_selection.artifacts import ScoreJournal
from picking.pr_selection.config import atomic_json, freeze_json
from picking.pr_selection.modeling.qwen35 import load_engine, weight_mirror
from picking.pr_selection.scoring import method
from .backend import gradient_statistics
from .deployment import read_deployment, verify_execution


def partition_indices(records, worker, workers):
    if workers<1 or not 0<=worker<workers:raise ValueError('Invalid shard assignment')
    return range(worker,records,workers)


def checked_copy(source, target, expected):
    source,target=Path(source),Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    receipt=target.with_name(target.name+'.verified.json')
    if target.exists() and receipt.exists():
        record=json.loads(receipt.read_text())
        if record=={'sha256':expected,'bytes':target.stat().st_size,'mtime_ns':target.stat().st_mtime_ns}:return
    temporary=target.with_name(target.name+'.copy.tmp')
    shutil.copyfile(source,temporary)
    if file_hash(temporary)!=expected:raise ValueError(f'Copied input checksum differs: {source.name}')
    os.replace(temporary,target)
    atomic_json(receipt,{'sha256':expected,'bytes':target.stat().st_size,'mtime_ns':target.stat().st_mtime_ns})


def stage(bundle, storage, local):
    bundle,storage,local=Path(bundle),Path(storage),Path(local)
    manifest,deployment=read_deployment(bundle)
    verify_execution(deployment)
    local.mkdir(parents=True,exist_ok=True)
    for item in [manifest['candidate_file'],manifest['validation_file'],*manifest['m1_files'],*manifest['gq_files']]:
        checked_copy(bundle/item['file'],local/'bundle'/item['file'],item['sha256'])
    model_root=storage/manifest['m0_asset_path_relative']
    for item in manifest['m0_files']:
        checked_copy(model_root/item['file'],local/manifest['m0_asset_path_relative']/item['file'],item['sha256'])
    cfg=copy.deepcopy(manifest['parent_config'])
    cfg['artifact_root']=str(local);cfg['runtime_root']=str(local/'runtime')
    # Exactly one coordinator prepares shared mirrors before child processes.
    for item in manifest['m0_files']:
        if item['file'].endswith('.safetensors'):weight_mirror(cfg,item['file'])
    atomic_json(local/'manifest.json',manifest)
    atomic_json(local/'deployment.json',deployment)
    atomic_json(local/'engine_config.json',cfg)
    return manifest


def load_flat(local, descriptions, parameters, *, reference):
    seen=set();result={}
    for item in descriptions:
        path=Path(local)/'bundle'/item['file']
        with safe_open(path,framework='pt',device='cpu') as reader:
            if set(reader.keys())!=set(item['names']):raise ValueError('Tensor shard names changed')
            for name in reader.keys():
                tensor=reader.get_tensor(name)
                if name in seen or name not in parameters or tensor.shape!=parameters[name].shape or tensor.dtype!=torch.float32:
                    raise ValueError(f'Tensor contract mismatch: {name}')
                seen.add(name)
                if reference:result[name]=tensor.to(parameters[name].device)
                else:
                    with torch.no_grad():parameters[name].copy_(tensor)
    if seen!=set(parameters):raise ValueError('Incomplete full-language-model tensor set')
    return result


def validate_numbers(actual, expected, gq_norm):
    """Cross-device BF16 checks; strict FP64-reduction equality is tested separately.

    These bounds are declared before the A100 run. Failure stops that worker;
    no automatic tolerance relaxation or mixing with parent H100 scores.
    """
    if not all(math.isfinite(x) for x in [gq_norm,*actual.values(),expected['loss_before'],expected['grad_dot'],expected['gradient_norm']]):
        raise FloatingPointError('Nonfinite numerical validation input')
    loss_error=abs(actual['loss_before']-expected['loss_before'])
    dot_error=abs(actual['gradient_dot']-expected['grad_dot'])/max(1e-12,gq_norm*expected['gradient_norm'])
    norm_error=abs(actual['gradient_norm']-expected['gradient_norm'])/max(1e-12,expected['gradient_norm'])
    passed=loss_error<=5e-4*max(1.,abs(expected['loss_before'])) and dot_error<=1e-3 and norm_error<=.02
    return {'passed':bool(passed),'loss_absolute_difference':loss_error,'dot_error_over_norm_product':dot_error,'gradient_norm_relative_difference':norm_error}


def worker(local, output, worker_id, workers, validate_only=False):
    local,output=Path(local),Path(output)
    manifest=json.loads((local/'manifest.json').read_text())
    deployment=json.loads((local/'deployment.json').read_text())
    verify_execution(deployment)
    cfg=json.loads((local/'engine_config.json').read_text())
    source_id=digest(manifest)
    expected_workers=deployment['nodes']*deployment['workers_per_node']
    if workers!=expected_workers:raise ValueError('Worker partition differs from deployment')
    hardware={'gpu':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0))}
    if deployment['expected_gpu'] not in hardware['gpu']:raise ValueError('Unexpected GPU architecture for this run')
    worker_identity={'bundle':source_id,'deployment':digest(deployment),'worker':worker_id,'workers':workers,
                     'hardware':hardware,'reduction':'gpu-fp64-v1'}
    freeze_json(output/'identity.json',worker_identity)
    all_rows=pd.read_parquet(local/'bundle/candidates.parquet').to_dict('records')
    if len(all_rows)!=manifest['records'] or len({row['id'] for row in all_rows})!=len(all_rows):
        raise ValueError('Candidate count/uniqueness differs from manifest')
    rows=[all_rows[i] for i in partition_indices(len(all_rows),worker_id,workers)]
    gold=json.loads((local/'bundle/validation.json').read_text())
    checks={row['id']:row for row in all_rows if row['id'] in {item['id'] for item in gold}}
    expected={(item['id'],item['objective']):item for item in gold}
    if not checks or len(expected)!=len(gold) or set(expected)!={(identifier,obj) for identifier in checks for obj in cfg['objectives']}:
        raise ValueError('Incomplete paired numerical validation examples')
    engine=load_engine(cfg,[0,1])
    reference=load_flat(local,manifest['gq_files'],engine.parameters,reference=True)
    validation=[]
    for row in checks.values():
        for objective in cfg['objectives']:
            loss=engine.backward(row,objective)
            dot,norm=gradient_statistics(engine,reference)
            actual={'loss_before':loss,'gradient_dot':dot,'gradient_norm':norm}
            checked=validate_numbers(actual,expected[(row['id'],objective)],manifest['gq_norm'])
            validation.append({'id':row['id'],'objective':objective,**actual,**checked})
    atomic_json(output/'m0_validation.json',validation)
    if not all(row['passed'] for row in validation):raise RuntimeError('A100 numerical validation failed; see m0_validation.json')
    m0=ScoreJournal(output/'m0.jsonl',{**worker_identity,'stage':'M0'},
                    mirror_path=local/'journals'/f'{worker_id:03d}-m0.jsonl',persist_every=32)
    if not validate_only:
        for i,row in enumerate(rows):
            if row['id'] in m0.rows:continue
            start=time.monotonic();record={'id':row['id']}
            for objective in cfg['objectives']:
                loss=engine.backward(row,objective);dot,norm=gradient_statistics(engine,reference)
                record[objective]={'loss_before':loss,'gradient_dot':dot,'gradient_norm':norm,
                                   'prediction_tokens':int(row[objective+'_tokens'])}
            m0.append(record)
            print('M0',worker_id,i+1,len(rows),time.monotonic()-start,flush=True)
        m0.persist()
    engine.zero_grad();del reference;gc.collect();torch.cuda.empty_cache()
    load_flat(local,manifest['m1_files'],engine.parameters,reference=False)
    after_check=[]
    for row in checks.values():
        for objective,value in engine.losses(row,cfg['objectives']).items():
            before=expected[(row['id'],objective)]['loss_after']
            after_check.append({'id':row['id'],'objective':objective,'loss_after':value,'reference':before,
                                'passed':abs(value-before)<=5e-4*max(1.,abs(before))})
    atomic_json(output/'m1_validation.json',after_check)
    if not all(row['passed'] for row in after_check):raise RuntimeError('M1 numerical validation failed')
    if validate_only:
        atomic_json(output/'VALIDATED.json',worker_identity);return
    m1=ScoreJournal(output/'m1.jsonl',{**worker_identity,'stage':'M1'},
                    mirror_path=local/'journals'/f'{worker_id:03d}-m1.jsonl',persist_every=32)
    for i,row in enumerate(rows):
        if row['id'] in m1.rows:continue
        start=time.monotonic();m1.append({'id':row['id'],**engine.losses(row,cfg['objectives'])})
        print('M1',worker_id,i+1,len(rows),time.monotonic()-start,flush=True)
    m1.persist()
    result=[]
    for row in rows:
        for objective in cfg['objectives']:
            scored={'id':row['id'],'objective':objective,**m0.rows[row['id']][objective],
                    'loss_after':m1.rows[row['id']][objective], 'input_tokens':int(row['input_tokens'])}
            for name in cfg['methods']:scored[name]=float(method(name).score([scored],seed=cfg['data']['sample_seed'])[0])
            result.append(scored)
    frame=pd.DataFrame(result)
    if not np.isfinite(frame[[*cfg['methods'],'loss_before','loss_after','gradient_norm']].to_numpy()).all():
        raise FloatingPointError('Nonfinite worker scores')
    frame.to_parquet(output/'scores.parquet',index=False)
    atomic_json(output/'COMPLETED.json',{**worker_identity,
                'records':len(rows),'scores_sha256':file_hash(output/'scores.parquet')})


def node(bundle, storage, local, output, node_id, node_count, validate_only=False):
    # A stable host-local lock also prevents duplicate starts with different scratch paths.
    lock_root=Path('/tmp/pr-scoring-locks');lock_root.mkdir(exist_ok=True)
    with (lock_root/(digest(str(Path(output).absolute()))+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return run_node(bundle,storage,local,output,node_id,node_count,validate_only)


def run_node(bundle, storage, local, output, node_id, node_count, validate_only=False):
    local,output=Path(local),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    freeze_json(output/'OWNER.json',{'hostname':socket.gethostname()})
    manifest=stage(bundle,storage,local)
    deployment=json.loads((local/'deployment.json').read_text())
    if node_count!=deployment['nodes'] or not 0<=node_id<node_count:raise ValueError('Node assignment differs from deployment')
    cfg=json.loads((local/'engine_config.json').read_text())
    cfg['run_id']=output.parent.name+'-'+output.name
    atomic_json(local/'engine_config.json',cfg)
    freeze_json(output/'run.json',{'run_id':cfg['run_id'],'node_id':node_id,'nodes':node_count,
                                 'deployment':digest(deployment),'bundle':digest(manifest)})
    total_workers=node_count*4
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    devices=visible.split(',') if visible else [str(i) for i in range(torch.cuda.device_count())]
    if len(devices)!=8:raise ValueError('Each node must expose exactly 8 allocated GPUs')
    children=[]
    try:
        for index in range(4):
            worker_id=node_id*4+index;folder=output/f'worker-{worker_id:03d}';folder.mkdir(exist_ok=True)
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=','.join(devices[index*2:index*2+2]);env['DATA_PICKING_GPU_IDS']='0,1'
            command=[sys.executable,'-u','-m','picking.pr_parallel.cluster','worker','--local',str(local),'--output',str(folder),
                     '--worker-id',str(worker_id),'--workers',str(total_workers)]
            if validate_only:command.append('--validate-only')
            with (folder/'worker.log').open('a') as log:children.append(subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Scoring worker failed; inspect per-worker logs')
            time.sleep(2)
        if any(p.returncode for p in children):raise RuntimeError('Scoring worker failed')
        atomic_json(output/('VALIDATED.json' if validate_only else 'COMPLETED.json'),{'node':node_id,'nodes':node_count,'bundle':digest(manifest),'deployment':digest(deployment)})
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait()


def merge(outputs, bundle, output):
    outputs,bundle,output=Path(outputs),Path(bundle),Path(output)
    manifest,deployment=read_deployment(bundle);frames=[];workers=set()
    if file_hash(bundle/'candidates.parquet')!=manifest['candidate_file']['sha256']:
        raise ValueError('Candidate file differs from bundle')
    candidates=pd.read_parquet(bundle/'candidates.parquet')
    total_workers=deployment['nodes']*deployment['workers_per_node']
    objectives=manifest['parent_config']['objectives']
    for completed in sorted(outputs.glob('node-*/worker-*/COMPLETED.json')):
        meta=json.loads(completed.read_text())
        if meta['bundle']!=digest(manifest) or meta['deployment']!=digest(deployment) or meta['worker'] in workers:
            raise ValueError('Mixed or repeated worker result')
        path=completed.parent/'scores.parquet'
        if file_hash(path)!=meta['scores_sha256']:raise ValueError('Worker scores changed')
        shard=pd.read_parquet(path)
        assigned={candidates.iloc[i].id for i in partition_indices(len(candidates),meta['worker'],total_workers)}
        if meta['workers']!=total_workers or meta['records']!=len(assigned) or set(shard.objective)!=set(objectives):
            raise ValueError('Worker partition/objectives differ')
        for objective in objectives:
            if set(shard[shard.objective==objective].id)!=assigned:raise ValueError('Worker scored the wrong partition')
        workers.add(meta['worker']);frames.append(shard)
    if not frames:raise ValueError('No completed scoring shards')
    if workers!=set(range(deployment['nodes']*deployment['workers_per_node'])):raise ValueError('Worker coverage is incomplete')
    frame=pd.concat(frames,ignore_index=True)
    if frame.duplicated(['id','objective']).any():raise ValueError('Repeated scored PR')
    if len(candidates)!=manifest['records'] or candidates.id.duplicated().any():raise ValueError('Invalid candidate pool')
    for objective in objectives:
        if set(frame[frame.objective==objective].id)!=set(candidates.id):raise ValueError('Full candidate coverage is incomplete')
    if not np.isfinite(frame[[*manifest['parent_config']['methods'],'loss_before','loss_after','gradient_norm']].to_numpy()).all():
        raise FloatingPointError('Nonfinite merged scores')
    output.mkdir(parents=True,exist_ok=True)
    freeze_json(output/'identity.json',{'bundle':digest(manifest),'deployment':digest(deployment)})
    frame.sort_values(['id','objective']).to_parquet(output/'scores.parquet',index=False)
    atomic_json(output/'COMPLETED.json',{'records':len(candidates),'workers':len(workers),'distributions':6,
                'bundle':digest(manifest),'deployment':digest(deployment),'scores_sha256':file_hash(output/'scores.parquet'),
                'test_losses_computed':False})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['node','worker','merge'])
    for name in ['bundle','storage','local','output','outputs']:parser.add_argument('--'+name)
    parser.add_argument('--node-id',type=int,default=0);parser.add_argument('--node-count',type=int,default=10)
    parser.add_argument('--worker-id',type=int,default=0);parser.add_argument('--workers',type=int,default=40)
    parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    if args.stage=='node':node(args.bundle,args.storage,args.local,args.output,args.node_id,args.node_count,args.validate_only)
    elif args.stage=='worker':worker(args.local,args.output,args.worker_id,args.workers,args.validate_only)
    else:merge(args.outputs,args.bundle,args.output)


if __name__=='__main__':main()
