"""候选纯前向、分片配对、缓存恢复及历史 M1 身份检查。"""
import copy
import json
from importlib.metadata import PackageNotFoundError, version

import pandas as pd
import pytest
import torch

from picking.common import digest, file_hash
from picking.pr_selection.config import atomic_json, load_config, run_dir
from picking.pr_selection.imports import PACKAGES, frozen_identity, import_probe, verified_payload
from picking.pr_selection.parallel import partition
from picking.pr_selection.scoring import combine, score_model


def test_partition_exactly_covers_pool():
    rows=list(range(20000))
    shards=[partition(rows,i,4) for i in range(4)]
    assert all(len(s)==5000 for s in shards)
    assert sorted(x for s in shards for x in s)==rows
    with pytest.raises(ValueError):partition(rows,4,4)


def test_forward_only_pairing_and_resume(tmp_path):
    cfg=load_config('configs/pr_selection.yaml')
    cfg.update(run_id='forward-test',artifact_root=str(tmp_path),runtime_root=str(tmp_path/'scratch'))
    cfg.pop('probe_from_run')
    root=run_dir(cfg);(root/'data').mkdir(parents=True)
    candidates=[{'id':str(i),'input_tokens':5,'full_text_tokens':4,'answer_only_tokens':2,
                 'query_id':str(i),'source_category':'test','format':'git_diff'} for i in range(5)]
    pd.DataFrame(candidates).to_parquet(root/'data/candidates.parquet')
    atomic_json(root/'pretraining/probe/latest.json',{'fingerprint':'test','sha256':'fixed-model'})
    class ForwardOnly:
        model='M0'
        calls=[]
        def losses(self,row,objectives):
            self.calls.append((self.model,row['id'],objectives))
            return {obj:(2. if self.model=='M0' else 1.) for obj in objectives}
        def backward(self,*args,**kwargs):raise AssertionError('Candidate scoring must never backpropagate')
    engine=ForwardOnly()
    for index in range(2):
        for model in ['M0','M1']:
            engine.model=model
            score_model(engine,partition(candidates,index,2),cfg,model,folder=root/'workers'/f'{index:02d}',partition={'worker':index,'workers':2})
    assert len(engine.calls)==10 and all(len(c[2])==2 for c in engine.calls)
    result=combine(cfg,candidates,workers=2,parallel=True)
    assert len(result)==10 and result.absolute_loss_drop.eq(1.).all() and result.relative_loss_drop.eq(.5).all()
    assert json.loads((root/'SCORING_COMPLETED.json').read_text())['distributions']==4
    score_model(engine,partition(candidates,0,2),cfg,'M1',folder=root/'workers/00',partition={'worker':0,'workers':2})
    assert len(engine.calls)==10
    path=root/'workers/00/scores/full_text/m0.jsonl'
    row=json.loads(path.read_text().splitlines()[0]);row['id']='other'
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='coverage'):combine(cfg,candidates,workers=2,parallel=True)


def make_history(tmp_path):
    cfg=load_config('configs/pr_selection.yaml')
    cfg.update(artifact_root=str(tmp_path),runtime_root=str(tmp_path/'scratch'),run_id='destination',probe_from_run='source')
    original={**cfg,'run_id':'source'};original.pop('probe_from_run')
    source=tmp_path/'runs/source';destination=run_dir(cfg)
    for folder in [source,destination]:
        (folder/'data').mkdir(parents=True);(folder/'data/guidance.parquet').write_bytes(b'fixed-Q')
    snapshot=source/'source/numerics.py';snapshot.parent.mkdir();snapshot.write_text('fixed numerical implementation')
    files={'numerics.py':file_hash(snapshot)}
    versions={}
    for name in PACKAGES:
        try:versions[name]=version(name)
        except PackageNotFoundError:versions[name]=None
    historical=digest({'config':original,'implementation':files,'versions':versions})
    atomic_json(source/'effective_config.json',original)
    atomic_json(source/'source_identity.json',{'identity':historical,'files':files})
    fingerprint=digest({'experiment':historical,'stage':'probe','data':file_hash(source/'data/guidance.parquet')})
    checkpoint=source/'pretraining/probe/checkpoints/step-000001';checkpoint.mkdir(parents=True)
    payload={'fingerprint':fingerprint,'completed':True,'trainer':{'step':1},'optimizer':{},'scheduler':{},'rng':{},'model':{'weight':torch.ones(3)}}
    torch.save(payload,checkpoint/'state.pt')
    meta={'name':'step-000001','fingerprint':fingerprint,'step':1,'sha256':file_hash(checkpoint/'state.pt'),'bytes':(checkpoint/'state.pt').stat().st_size}
    atomic_json(checkpoint/'manifest.json',meta)
    atomic_json(source/'pretraining/probe/latest.json',meta)
    atomic_json(source/'pretraining/probe/COMPLETED.json',{'fingerprint':fingerprint,'step':1})
    return cfg,source


def test_historical_import_preserves_source_identity(tmp_path):
    cfg,source=make_history(tmp_path)
    expected=json.loads((source/'source_identity.json').read_text())['identity']
    imported=import_probe(cfg)
    assert imported['source_identity']==expected
    assert verified_payload(imported['checkpoint'])['model']['weight'].eq(1).all()
    assert import_probe(cfg)==imported
    changed=copy.deepcopy(cfg);changed['scoring']['attention_backend']='flash'
    with pytest.raises(ValueError,match='another scoring identity'):import_probe(changed)
    (run_dir(cfg)/'data/guidance.parquet').write_bytes(b'changed after import')
    with pytest.raises(ValueError,match='different Q'):import_probe(cfg)
    (source/'source/numerics.py').write_text('changed')
    with pytest.raises(ValueError,match='source snapshot'):frozen_identity(source)


def test_import_rejects_different_q(tmp_path):
    cfg,source=make_history(tmp_path)
    (run_dir(cfg)/'data/guidance.parquet').write_bytes(b'other Q')
    with pytest.raises(ValueError,match='different Q'):import_probe(cfg)
