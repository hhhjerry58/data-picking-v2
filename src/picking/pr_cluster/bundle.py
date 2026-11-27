"""Freeze the completed 100k pool, 32 step plans and an immutable node runtime."""
from __future__ import annotations

import json
import gzip
import shutil
import tarfile
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import yaml

from picking.common import REPO, digest
from picking.pr_selection.config import atomic_json
from .sampling import experiments, subset_indices, step_plan
from .storage import BlobStore, RunLease, sha256


def make_test(source, cfg, destination):
    from picking.pr_selection.modeling.qwen35 import load_tokenizer
    from picking.pr_selection.modeling.masks import render_trajectory
    from picking.pr_selection.data.splits import task_id
    wanted = {row['id']: row['task_id'] for row in json.loads((source/'data/test_manifest.json').read_text())}
    tokenizer = load_tokenizer(cfg); rows = []
    for line in Path(cfg['data']['trajectory_file']).open():
        if not line.strip(): continue
        row = json.loads(line)
        if row['id'] not in wanted: continue
        if task_id(row) != wanted[row['id']]: raise ValueError('T task mapping differs')
        rows.append({**render_trajectory(tokenizer, row), 'task_id': wanted[row['id']]})
    if len(rows) != 135 or len(set(wanted.values())) != 53 or {r['id'] for r in rows} != set(wanted):
        raise ValueError('Frozen T coverage differs')
    pd.DataFrame(rows).sort_values('id').to_parquet(destination, index=False)


def build(config_path, staging, publish=False, smoke_only=False):
    cfg = yaml.safe_load(Path(config_path).read_text()); staging = Path(staging)
    if smoke_only:
        cfg['pool_run']='pr-lossdrop-qwen35-9b-v1-20261126'
        cfg['bundle_prefix']+='-smoke'
    source = Path(cfg['artifact_root'])/'runs'/cfg['pool_run']
    if not (source/'SCORING_COMPLETED.json').exists(): raise ValueError('The 100k scoring pool is not complete yet')
    from picking.pr_selection.imports import frozen_identity
    pool_cfg, pool_identity = frozen_identity(source)
    completed = json.loads((source/'SCORING_COMPLETED.json').read_text())
    if completed['identity'] != pool_identity or completed['candidates'] != (20000 if smoke_only else 100000):
        raise ValueError('Completed 100k pool identity differs')
    if cfg['random_seeds'] != list(range(42,50)) or cfg['selection_seed'] != 42 or cfg['world_size'] != 8:
        raise ValueError('This deployment requires the agreed 32 jobs and eight ranks')
    staging.mkdir(parents=True, exist_ok=True)
    if (staging/'READY.json').exists(): raise ValueError('Do not overwrite a published bundle')
    (staging/'plans').mkdir(exist_ok=True); (staging/'data').mkdir(exist_ok=True)
    candidates = pd.read_parquet(source/'data/candidates.parquet')
    scores = pd.read_parquet(source/'scores/paired_scores.parquet')
    if len(candidates) != (20000 if smoke_only else 100000): raise ValueError('Wrong pool size')
    if smoke_only:
        ordered=candidates.sort_values('input_tokens')
        candidates=ordered.iloc[[0,len(ordered)//2,len(ordered)-1]].reset_index(drop=True)
        candidates.to_parquet(staging/'data/pool.parquet',index=False)
    else:shutil.copyfile(source/'data/candidates.parquet', staging/'data/pool.parquet')
    shutil.copyfile(source/'scores/paired_scores.parquet', staging/'data/scores.parquet')
    make_test(source, pool_cfg, staging/'data/test.parquet')
    jobs = []
    for job in (experiments()[:1] if smoke_only else experiments()):
        indexes = list(range(len(candidates))) if smoke_only else subset_indices(candidates, scores, job)
        plan = [] if smoke_only else step_plan(candidates, indexes, job['seed'], steps=cfg['training']['steps'],
                                              input_tokens_per_step=cfg['training']['global_input_tokens_per_step'])
        target_column = job['objective']+'_tokens'
        used = [sample['index'] for step in plan for sample in step['samples']]
        record = {**job, 'subset_records': len(indexes), 'subset_input_tokens': int(candidates.iloc[indexes].input_tokens.sum()),
                  'subset_target_tokens': int(candidates.iloc[indexes][target_column].sum()),
                  'training_input_tokens': sum(step['input_tokens'] for step in plan),
                  'training_target_tokens': int(candidates.iloc[used][target_column].sum()),
                  'unique_training_records': len(set(used)), 'sample_occurrences': len(used), 'steps': plan}
        atomic_json(staging/'plans'/(job['name']+'.json'), record)
        jobs.append({k:v for k,v in record.items() if k != 'steps'})
    atomic_json(staging/'jobs.json', jobs)
    atomic_json(staging/'protocol.json', cfg)
    files = sorted(set([*list((REPO/'src/picking/pr_cluster').rglob('*.py')),
        *list((REPO/'src/picking/pr_selection/modeling').glob('*.py')),
        REPO/'src/picking/pr_selection/__init__.py', REPO/'src/picking/pr_selection/config.py',
        REPO/'src/picking/__init__.py', REPO/'src/picking/common.py',
        *list((REPO/'methods').glob('*/method.py'))]))
    source_hashes = {str(p.relative_to(REPO)): sha256(p) for p in files}
    def stable_header(info):
        info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
        return info
    with (staging/'code.tar.gz').open('wb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as compressed:
            with tarfile.open(fileobj=compressed,mode='w') as archive:
                for path in files: archive.add(path,arcname=str(path.relative_to(REPO)),recursive=False,filter=stable_header)
    runtime = Path(cfg['runtime_archive'])
    if not runtime.is_file() or not Path(str(runtime)+'.complete').exists():
        raise ValueError('Portable runtime archive has not completed')
    local_files = [staging/'data/pool.parquet', staging/'data/test.parquet', staging/'jobs.json',
                   staging/'protocol.json', *sorted((staging/'plans').glob('*.json'))]
    assets = [{'key': cfg['bundle_prefix']+'/'+str(p.relative_to(staging)),
               'local_path': str(p.relative_to(staging)), 'sha256': sha256(p), 'bytes': p.stat().st_size} for p in local_files]
    previous = json.loads((Path(cfg['artifact_root'])/'runs/pr-a100-bundle-v1-20261125/manifest.json').read_text())
    if previous['parent_config']['model']['revision'] != pool_cfg['model']['revision']:
        raise ValueError('M0 revision differs from the fixed input weights')
    for item in previous['m0_files']:
        if item['file'].endswith('.safetensors') or item['file'] in ['config.json','model.safetensors.index.json']:
            assets.append({'key': 'data-picking-v2/'+previous['m0_asset_path_relative']+'/'+item['file'],
                           'local_path': 'm0/'+item['file'], 'sha256': item['sha256'], 'bytes': item['bytes']})
    archives = {'code.tar.gz': {'sha256': sha256(staging/'code.tar.gz'), 'bytes': (staging/'code.tar.gz').stat().st_size},
                'runtime.tar.gz': {'sha256': sha256(runtime), 'bytes': runtime.stat().st_size}}
    manifest = {'schema': 1, 'campaign': cfg['campaign'], 'pool_identity': pool_identity,
                'pool_run': str(source), 'model': pool_cfg['model'], 'protocol': cfg,
                'source_files': source_hashes, 'assets': assets, 'archives': archives, 'jobs': jobs,
                'packages': {name: version(name) for name in ['torch','transformers','tokenizers','flash-linear-attention','tilelang','triton','causal-conv1d']},
                'scoring_hardware': 'H100', 'training_hardware': 'A100',
                'checkpoints': {'every_steps':5,'retained_per_job':20,'optimizer_saved':True},
                'smoke_only':smoke_only, 'a100_smoke_required': True}
    atomic_json(staging/'manifest.json', manifest)
    ready = {'manifest_identity': digest(manifest), 'status': 'ready_for_user_submitted_smoke', 'jobs': len(jobs)}
    if publish:
        store = BlobStore(cfg['storage_account'], cfg['storage_container'])
        with RunLease(store,cfg['bundle_prefix']+'/_publication.json',digest(manifest)) as lease:
            if store.request('HEAD',cfg['bundle_prefix']+'/READY.json',accepted=(200,404)).status_code==200:
                raise ValueError('The remote bundle is already published; use a new prefix')
            for item in assets:
                if item['key'].startswith(cfg['bundle_prefix']+'/'):
                    store.upload(staging/item['local_path'], item['key'],assert_owner=lease.check)
                else:
                    head=store.request('HEAD',item['key'])
                    if int(head.headers['Content-Length'])!=item['bytes']:raise ValueError('M0 asset size differs')
            store.upload(staging/'code.tar.gz', cfg['bundle_prefix']+'/code.tar.gz',assert_owner=lease.check)
            store.upload(runtime, cfg['bundle_prefix']+'/runtime.tar.gz',assert_owner=lease.check)
            lease.check()
            store.put_json(cfg['bundle_prefix']+'/manifest.json', manifest)
            store.put_json(cfg['bundle_prefix']+'/READY.json', ready, create_only=True)
    atomic_json(staging/'READY.json', ready)
    print('BUNDLE_READY', staging, 'published', publish, flush=True)
    return manifest
