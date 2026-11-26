from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

from picking.common import file_hash
from picking.pr_selection.config import atomic_json
from picking.pr_selection.modeling.qwen35 import asset_dir
from .parent import checkpoint_descriptor, open_verified_checkpoint, parent_config


def shards(tensors, kind, staging, output):
    descriptions=[];group={};size=0;counter=0
    def flush():
        nonlocal counter,group,size
        if not group:return
        name=f'{kind}/part-{counter:03d}.safetensors'
        local=staging/name;local.parent.mkdir(parents=True,exist_ok=True)
        save_file(group,local)
        checksum=file_hash(local)
        destination=output/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(local,destination)
        descriptions.append({'file':name,'sha256':checksum,'bytes':local.stat().st_size,'names':list(group)})
        print('EXPORTED',name,local.stat().st_size,flush=True)
        counter+=1;group={};size=0
    for name,value in sorted(tensors.items()):
        if value.dtype!=torch.float32:raise ValueError('The scoring bundle preserves FP32 tensors')
        amount=value.numel()*value.element_size()
        if group and size+amount>2*1024**3:flush()
        group[name]=value.contiguous();size+=amount
    flush()
    return descriptions


def export(parent, preview, output):
    parent,preview,output=Path(parent),Path(preview),Path(output)
    cfg=parent_config(parent)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists():raise ValueError('Do not overwrite a published bundle')
    staging=Path('/dev/shm/pr-parallel-bundles')/output.name
    staging.mkdir(parents=True,exist_ok=True)
    preview_meta=json.loads((preview/'manifest.json').read_text())
    m1_desc=preview_meta['m1']
    m1=open_verified_checkpoint(m1_desc)
    if not m1['completed']:raise ValueError('M1 must be the completed probe')
    model_names=set(m1['model'])
    m1_files=shards(m1['model'],'m1',staging,output)
    del m1
    gq_desc=checkpoint_descriptor(parent,'target-gradient')
    gq=open_verified_checkpoint(gq_desc)
    if set(gq['gradients'])!=model_names:raise ValueError('M1 and gQ parameter sets differ')
    squared=0.
    torch.set_num_threads(16)
    for tensor in gq['gradients'].values():
        flat=tensor.reshape(-1)
        for start in range(0,flat.numel(),16*1024*1024):
            values=flat[start:start+16*1024*1024].double()
            squared+=float(torch.dot(values,values))
    if not math.isfinite(squared) or squared<=0:raise ValueError('Invalid guidance gradient norm')
    gq_files=shards(gq['gradients'],'gq',staging,output)
    denominator=gq['denominator'];q_records=gq['cursor']
    del gq
    shutil.copyfile(parent/'data/candidates.parquet',output/'candidates.parquet')
    import pandas as pd
    scores=pd.read_parquet(preview/'scores.parquet')
    check_ids=preview_meta['check_ids']
    validation=scores[scores.id.isin(check_ids)][['id','objective','loss_before','loss_after','grad_dot','gradient_norm']].to_dict('records')
    atomic_json(output/'validation.json',validation)
    assets=asset_dir(cfg)
    index=json.loads((assets/'model.safetensors.index.json').read_text())['weight_map']
    names=['config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','chat_template.jinja',*sorted(set(index.values()))]
    base_files=[]
    for name in names:
        p=assets/name
        base_files.append({'file':name,'bytes':p.stat().st_size,'sha256':file_hash(p)})
        meta=assets/'.cache/huggingface/download'/(name+'.metadata')
        if meta.exists():base_files.append({'file':str(meta.relative_to(assets)),'bytes':meta.stat().st_size,'sha256':file_hash(meta)})
    manifest={'schema_version':1,'parent_run':str(parent),'parent_config':cfg,
              'candidate_file':{'file':'candidates.parquet','sha256':file_hash(output/'candidates.parquet'),'bytes':(output/'candidates.parquet').stat().st_size},
              'records':20000,'q_records':q_records,'q_prediction_tokens':denominator,'gq_norm':math.sqrt(squared),
              'm1_origin':m1_desc,'gq_origin':gq_desc,'m1_files':m1_files,'gq_files':gq_files,
              'm0_asset_path_relative':str(assets.relative_to(Path(cfg['artifact_root']))),'m0_files':base_files,
              'validation_file':{'file':'validation.json','sha256':file_hash(output/'validation.json'),'bytes':(output/'validation.json').stat().st_size},
              'note':'M1 and gQ are read-only exports verified against original checkpoint fingerprints. No probe rerun or T losses.'}
    atomic_json(output/'manifest.json',manifest)
    print('BUNDLE_EXPORTED',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',required=True);parser.add_argument('--preview',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();export(args.parent,args.preview,args.output)
