from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from picking.common import REPO, digest, file_hash
from picking.pr_selection.artifacts import ScoreJournal
from picking.pr_selection.config import atomic_json, freeze_json, identity
from .parent import checkpoint_descriptor, open_verified_checkpoint, parent_config


def snapshot(parent, output, limit=0):
    parent, output = Path(parent), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = parent_config(parent)
    before = {}
    for objective in cfg['objectives']:
        path = parent/'scores'/objective/'m0.jsonl'
        source = json.loads(path.with_suffix('.identity.json').read_text())
        if source['identity'] != identity(cfg):
            raise ValueError('Parent score provenance differs')
        with path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if len({row['id'] for row in rows}) != len(rows):
            raise ValueError('Repeated parent score IDs')
        before[objective] = {row['id']: row for row in rows}
    common = set.intersection(*(set(rows) for rows in before.values()))
    candidates = pd.read_parquet(parent/'data/candidates.parquet')
    selected = candidates[candidates.id.isin(common)].sort_values('id')
    if limit:
        selected = selected.head(limit)
    if selected.empty:
        raise ValueError('No paired completed PR scores')
    selected.to_parquet(output/'candidates.parquet', index=False)
    rows = [{**before[obj][identifier], 'objective': obj} for obj in cfg['objectives'] for identifier in selected.id]
    pd.DataFrame(rows).to_parquet(output/'m0.parquet', index=False)
    # Metadata-only selection of validation examples, never selection by scores.
    ordered = selected.sort_values('input_tokens')
    check_ids = ordered.iloc[sorted({0, len(ordered)//2, len(ordered)-1})].id.tolist()
    descriptor = checkpoint_descriptor(parent, 'probe')
    manifest = {'parent_run': str(parent), 'parent_identity': identity(cfg), 'parent_config': cfg,
                'm1': descriptor, 'records': len(selected), 'parent_records': len(candidates),
                'scope': 'already-scored ID-sorted prefix; NOT a random sample of the full pool',
                'partial_preview': True, 'check_ids': check_ids,
                'candidate_sha256': file_hash(output/'candidates.parquet'),
                'm0_sha256': file_hash(output/'m0.parquet'),
                'created_at_utc': datetime.now(timezone.utc).isoformat()}
    freeze_json(output/'manifest.json', manifest)
    return manifest


def worker(run, index, count):
    # The numerical backend stays identical to the parent. Only independent
    # forward requests are assigned to separate GPUs.
    from picking.pr_selection.modeling.qwen35 import load_engine
    run = Path(run)
    manifest = json.loads((run/'manifest.json').read_text())
    parent_config(manifest['parent_run'])
    execution = json.loads((run/'execution_identity.json').read_text())
    for name, expected in execution['code'].items():
        if file_hash(Path(__file__).parent/name) != expected:
            raise ValueError('Preview execution code changed')
    if file_hash(run/'candidates.parquet') != manifest['candidate_sha256']:
        raise ValueError('Preview candidate snapshot changed')
    cfg = manifest['parent_config']
    all_rows = pd.read_parquet(run/'candidates.parquet').to_dict('records')
    rows = all_rows[index::count]
    checks = [row for row in all_rows if row['id'] in manifest['check_ids']]
    prior = pd.read_parquet(run/'m0.parquet').set_index(['id','objective'])
    engine = load_engine(cfg, [0])
    parity = []
    for row in checks:
        losses = engine.losses(row, cfg['objectives'])
        for objective, actual in losses.items():
            expected = float(prior.loc[(row['id'],objective), 'loss_before'])
            if abs(actual-expected) > 2e-5*max(1., abs(expected)):
                raise ValueError(f'M0 single-GPU parity failed: {row["id"]}/{objective}: {actual} vs {expected}')
            parity.append({'id':row['id'], 'objective':objective, 'parent_m0':expected, 'worker_m0':actual})
    state = open_verified_checkpoint(manifest['m1'])
    if not state.get('completed'):
        raise ValueError('M1 is not the completed fixed probe')
    engine.model.load_state_dict(state['model'], strict=True)
    del state
    gc.collect()
    m1_checks = [{'id': row['id'], **engine.losses(row,cfg['objectives'])} for row in checks]
    folder = run/'workers'/f'{index:02d}'
    atomic_json(folder/'parity.json', {'m0':parity, 'm1':m1_checks})
    journal = ScoreJournal(folder/'m1.jsonl', {'preview':digest(manifest), 'worker':index, 'workers':count},
                           mirror_path=Path(cfg['runtime_root'])/run.name/'workers'/f'{index:02d}.jsonl', persist_every=32)
    for i,row in enumerate(rows):
        if row['id'] in journal.rows:
            continue
        t = time.monotonic()
        values = engine.losses(row, cfg['objectives'])
        journal.append({'id':row['id'], **values})
        print('PREVIEW_M1',index,i+1,len(rows),row['id'],values,'seconds',time.monotonic()-t,flush=True)
    journal.persist()
    atomic_json(folder/'COMPLETED.json', {'records':len(rows), 'preview':digest(manifest)})


def run_workers(run, gpu_ids):
    from picking.runtime import gpu_inventory
    import fcntl
    run = Path(run)
    code = {name:file_hash(Path(__file__).parent/name) for name in ['preview.py','parent.py']}
    freeze_json(run/'execution_identity.json', {'code':code,'manifest':file_hash(run/'manifest.json'),'gpus':gpu_ids})
    for name in code:
        (run/'source').mkdir(exist_ok=True)
        shutil.copyfile(Path(__file__).parent/name,run/'source'/name)
    allowed = set(gpu_ids)
    if len(allowed) != len(gpu_ids) or not allowed:
        raise ValueError('Distinct GPU IDs are required')
    available = {r['index'] for r in gpu_inventory() if r['memory_used_mib'] < 768 and r['utilization'] < 5}
    if not allowed <= available:
        raise ValueError('Requested GPUs are occupied')
    locks, children = [], []
    try:
        for gpu in gpu_ids:
            lock = open(f'/tmp/data-picking-domain-gpu-{gpu}.lock','a')
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        for index,gpu in enumerate(gpu_ids):
            folder=run/'workers'/f'{index:02d}';folder.mkdir(parents=True,exist_ok=True)
            env=os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu),DATA_PICKING_GPU_IDS=str(gpu),PYTHONPATH=str(REPO/'src'),
                       CUDA_HOME='/usr/local/cuda-12.8',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
            env['PATH']=env['CUDA_HOME']+'/bin:'+env.get('PATH','')
            with (folder/'worker.log').open('a') as log:
                p=subprocess.Popen([sys.executable,'-u','-m','picking.pr_parallel.preview','worker','--run',str(run),
                                    '--worker',str(index),'--workers',str(len(gpu_ids))],env=env,stdout=log,stderr=subprocess.STDOUT)
            children.append(p)
        while any(p.poll() is None for p in children):
            failed=[p.returncode for p in children if p.poll() not in (None,0)]
            if failed:raise RuntimeError(f'Preview worker failed: {failed}')
            time.sleep(1)
        if any(p.returncode for p in children):raise RuntimeError('Preview worker failed')
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for lock in locks:lock.close()


def collect(run):
    from picking.pr_selection.scoring import method
    run=Path(run);manifest=json.loads((run/'manifest.json').read_text())
    records=[];checks=[]
    for folder in sorted((run/'workers').glob('*')):
        if not (folder/'COMPLETED.json').exists():raise ValueError('Incomplete preview partition')
        with (folder/'m1.jsonl').open() as f:records += [json.loads(line) for line in f if line.strip()]
        checks.append(json.loads((folder/'parity.json').read_text()))
    canonical={(r['id'],o):r[o] for r in checks[0]['m1'] for o in manifest['parent_config']['objectives']}
    for check in checks[1:]:
        for r in check['m1']:
            for o in manifest['parent_config']['objectives']:
                if abs(r[o]-canonical[(r['id'],o)])>2e-5*max(1,abs(r[o])):
                    raise ValueError('M1 worker numerical parity failed')
    after=pd.DataFrame(records)
    candidates=pd.read_parquet(run/'candidates.parquet')
    if after.id.duplicated().any() or set(after.id)!=set(candidates.id):raise ValueError('Preview coverage mismatch')
    before=pd.read_parquet(run/'m0.parquet')
    frames=[]
    for o in manifest['parent_config']['objectives']:
        joined=before[before.objective==o].merge(after[['id',o]].rename(columns={o:'loss_after'}),on='id',validate='one_to_one')
        records=joined.to_dict('records')
        for name in manifest['parent_config']['methods']:
            joined[name]=method(name).score(records,seed=manifest['parent_config']['data']['sample_seed'])
        cols=['id','query_id','source_category','format','input_tokens','full_text_tokens','answer_only_tokens']
        frames.append(joined.merge(candidates[cols],on='id',validate='one_to_one'))
    frame=pd.concat(frames,ignore_index=True)
    if not np.isfinite(frame[['loss_before','loss_after','absolute_loss_drop','relative_loss_drop','grad_dot']]).all().all():
        raise ValueError('Nonfinite preview scores')
    frame.to_parquet(run/'scores.parquet',index=False)
    atomic_json(run/'COMPLETED.json', {'records':len(candidates),'distributions':6,'partial_preview':True,'manifest':digest(manifest)})
    return frame


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['snapshot','worker','run','collect'])
    parser.add_argument('--run',required=True)
    parser.add_argument('--parent')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--worker',type=int,default=0)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--gpus',default='0,1,2,3')
    args=parser.parse_args()
    if args.stage=='snapshot':snapshot(args.parent,args.run,args.limit)
    elif args.stage=='worker':worker(args.run,args.worker,args.workers)
    elif args.stage=='run':run_workers(args.run,[int(x) for x in args.gpus.split(',')]);collect(args.run)
    else:collect(args.run)


if __name__=='__main__':main()
