"""One eight-GPU A100 training job, with renewable ownership and rank checkpoints."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from picking.common import digest
from .checkpoints import restore_local, save_local
from .model import example, load_backbone, wrap
from .sampling import rank_batches
from .storage import BlobStore, RunLease
from .validation import check_fsdp, check_linear_attention
from .protocol import recipe_identity


def broadcast(value):
    values=[value];dist.broadcast_object_list(values,src=0);return values[0]


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed(value)


def evaluate(model, rows, device, owned=lambda: None):
    rank,world=dist.get_rank(),dist.get_world_size()
    model.eval();local=[]
    with torch.no_grad():
        for start in range(0,len(rows),world):
            owned()
            index=start+rank;row=rows[index] if index<len(rows) else None
            ids,positions,count=example(row,'guidance',device)
            value=model(ids,positions)
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite T loss')
            if row is not None:
                local.append({'id':row['id'],'task_id':row['task_id'],'nll_sum':float(value),
                              'prediction_tokens':count,'input_tokens':int(row['input_tokens'])})
            if rank==0:print('EVAL',min(start+world,len(rows)),len(rows),flush=True)
            owned()
    all_rows=[None]*world;dist.all_gather_object(all_rows,local)
    if rank!=0:return None
    frame=pd.DataFrame([row for shard in all_rows for row in shard])
    if frame.id.duplicated().any() or set(frame.id)!={row['id'] for row in rows}:raise ValueError('T coverage differs')
    tasks=frame.groupby('task_id')[['nll_sum','prediction_tokens']].sum()
    return {'records':frame.to_dict('records'),'token_mean_nll':float(frame.nll_sum.sum()/frame.prediction_tokens.sum()),
            'task_mean_nll':float((tasks.nll_sum/tasks.prediction_tokens).mean()),'tasks':len(tasks)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',required=True);parser.add_argument('--job',required=True)
    parser.add_argument('--mode',choices=['train','smoke'],default='train')
    args=parser.parse_args();assets=Path(args.assets)
    warnings.filterwarnings('ignore',message='Please use DTensor instead.*',category=FutureWarning)
    manifest=json.loads((assets/'manifest.json').read_text());cfg=manifest['protocol']
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=cfg['world_size']:raise ValueError('Distributed world size differs from the frozen protocol')
    device=torch.device('cuda',local_rank);torch.cuda.set_device(device)
    dist.init_process_group('nccl',device_id=device,timeout=timedelta(minutes=30))
    torch.set_num_threads(2)
    job=json.loads((assets/'plans'/(args.job+'.json')).read_text())
    if job['name']!=args.job:raise ValueError('Wrong job plan')
    identity=digest({'manifest':digest(manifest),'job':job,'world_size':world,'mode':args.mode,
                     'gpu':torch.cuda.get_device_name(device),'capability':torch.cuda.get_device_capability(device)})
    name=job['name']+('-smoke' if args.mode=='smoke' else '')
    prefix=cfg['output_prefix']+'/'+name
    scratch=assets.parent/name;scratch.mkdir(parents=True,exist_ok=True)
    heartbeat=scratch/'lease.json'
    store=BlobStore(cfg['storage_account'],cfg['storage_container'])
    lease_context=RunLease(store,prefix+'/STATE.json',identity,heartbeat) if rank==0 else nullcontext()
    with lease_context as lease:
        state=broadcast(lease.state if rank==0 else None)
        attempt=broadcast(lease.attempt if rank==0 else None)
        if state.get('completed'):
            if rank==0:print('ALREADY_COMPLETED',name,flush=True)
            dist.destroy_process_group();return
        def owned():
            health=json.loads(heartbeat.read_text())
            if health['attempt']!=attempt or time.time()-health['updated']>35:
                raise RuntimeError('Ownership heartbeat expired')
            if rank==0:lease.check()
        if args.mode=='smoke':
            checks={'distributed':check_fsdp(device,scratch),'linear_attention':check_linear_attention(device)}
            if rank==0:store.put_json(prefix+'/kernel_checks.json',checks)
        seed(job['seed']+rank)
        model=wrap(load_backbone(assets/'m0'),device,cfg['training']['head_chunk_tokens'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],
            betas=tuple(cfg['training']['betas']),eps=cfg['training']['epsilon'],
            weight_decay=cfg['training']['weight_decay'],foreach=False)
        warmup=cfg['training']['warmup_steps']
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:min(1.,(s+1)/max(1,warmup)))
        candidates=pd.read_parquet(assets/'data/pool.parquet')
        test=pd.read_parquet(assets/'data/test.parquet').to_dict('records')
        plan=job['steps']
        if args.mode=='smoke':
            longest=int(candidates.input_tokens.to_numpy().argmax())
            plan=[{'step':i+1,'samples':[{'index':longest,'epoch':0} for _ in range(world)],
                   'input_tokens':int(candidates.iloc[longest].input_tokens)*world} for i in range(2)]
            ordered=sorted(test,key=lambda row:row['input_tokens'])
            test=[ordered[i] for i in sorted({0,len(ordered)//2,len(ordered)-1})]
        if not state.get('baseline'):
            if state.get('latest'):raise ValueError('Checkpoint exists without its M0 baseline')
            baseline=evaluate(model,test,device,owned)
            if rank==0:
                key=prefix+f'/evaluation/step-000000-{attempt}.json'
                owned();store.put_json(key,baseline)
                state={**state,'baseline':{'key':key,'sha256':digest(baseline),
                                          'token_mean_nll':baseline['token_mean_nll']}}
                lease.publish(state)
            state=broadcast(state if rank==0 else None)
        start=0
        if state.get('latest'):
            shard=state['latest']['shards'][rank];path=scratch/f'resume-rank-{rank}.pt'
            store.download(shard['key'],path,shard['sha256'],shard['bytes'])
            start=restore_local(model,optimizer,scheduler,identity,path);path.unlink()
            if start!=state['latest']['step']:raise ValueError('Checkpoint cursor differs from manifest')
        else:seed(job['seed']+rank)
        model.train()
        metrics=list(state.get('metrics',[]))
        for step in plan[start:]:
            owned();begin=time.monotonic();optimizer.zero_grad(set_to_none=True)
            target_column=job['objective']+'_tokens'
            denominator=sum(int(candidates.iloc[sample['index']][target_column]) for sample in step['samples'])
            if denominator<=0:raise ValueError('Empty global target region')
            loss_total=torch.zeros((),device=device,dtype=torch.float64)
            for index in rank_batches(step,candidates,rank,world):
                row=candidates.iloc[index].to_dict() if index is not None else None
                ids,positions,count=example(row,job['objective'],device)
                value=model(ids,positions)
                if not torch.isfinite(value):raise FloatingPointError('Nonfinite training loss')
                (value*(world/denominator if count else 0.)).backward()
                if count:loss_total.add_(value.detach().double())
            norm=model.clip_grad_norm_(cfg['training']['max_grad_norm'])
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite global gradient norm')
            lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
            dist.all_reduce(loss_total)
            result={'step':step['step'],'input_tokens':step['input_tokens'],'target_tokens':denominator,
                    'loss':float(loss_total)/denominator,'learning_rate':lr,'grad_norm':float(norm),
                    'seconds':time.monotonic()-begin,'max_epoch':max(sample['epoch'] for sample in step['samples'])}
            metrics.append(result)
            if rank==0:
                print('TRAIN',json.dumps(result),flush=True)
                owned();lease.publish({**lease.state,'progress':{'identity':identity,'job':name,**result}})
            every=1 if args.mode=='smoke' else cfg['checkpoint']['every_steps']
            if step['step']%every==0 or step['step']==len(plan):
                optimizer.zero_grad(set_to_none=True)
                path=scratch/f'checkpoint-rank-{rank}.pt'
                save_local(model,optimizer,scheduler,step['step'],identity,path)
                key=f'{prefix}/checkpoints/step-{step["step"]:06d}-{attempt}/rank-{rank:03d}/state.pt'
                receipt=store.upload(path,key,assert_owner=owned)
                receipts=[None]*world;dist.all_gather_object(receipts,receipt)
                checkpoint={'step':step['step'],'shards':receipts,'identity':identity,'world_size':world}
                if rank==0:
                    lease.publish({**lease.state,'latest':checkpoint,
                        'checkpoints':[*lease.state['checkpoints'],checkpoint],'metrics':metrics})
                    print('CHECKPOINT_COMMITTED',step['step'],sum(x['bytes'] for x in receipts),flush=True)
                dist.barrier();path.unlink()
                if args.mode=='smoke' and step['step']==1:
                    # Read all shards through the blob API before restoring them.
                    store.download(receipt['key'],path,receipt['sha256'],receipt['bytes'])
                    if restore_local(model,optimizer,scheduler,identity,path)!=1:raise ValueError('Smoke resume cursor differs')
                    path.unlink();dist.barrier()
        result=evaluate(model,test,device,owned)
        if rank==0:
            key=prefix+f'/evaluation/step-{len(plan):06d}-{attempt}.json'
            owned();store.put_json(key,result)
            lease.publish({**lease.state,'completed':True,'final_nll':result['token_mean_nll'],'metrics':metrics,
                           'final_evaluation':{'key':key,'sha256':digest(result)},
                           'recipe_identity':recipe_identity(manifest),'world_size':world,'mode':args.mode})
            marker='SMOKE_PASSED.json' if args.mode=='smoke' else 'COMPLETED.json'
            owned();store.put_json(prefix+'/'+attempt+'/'+marker,{'identity':identity,'job':name,'steps':len(plan),
                           'manifest_identity':digest(manifest),'recipe_identity':recipe_identity(manifest),'world_size':world,
                           'token_mean_nll':result['token_mean_nll'],'baseline_nll':lease.state['baseline']['token_mean_nll']})
            print(marker,json.dumps({'job':name,'token_mean_nll':result['token_mean_nll']}),flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':main()
