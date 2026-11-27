import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from picking.pr_selection.config import load_config
from picking.pr_selection.modeling.qwen35 import Engine
from picking.pr_selection.continuation.data import batches, select
from picking.pr_selection.continuation.evaluation import evaluate
from picking.pr_selection.continuation.report import task_bootstrap
from picking.pr_selection.continuation.training import train
from picking.pr_selection.artifacts import CheckpointStore


def test_selection_budget_is_whole_record_prefix():
    frame=pd.DataFrame({'id':['a','b','c'],'input_tokens':[6,4,3],
                        'absolute_loss_drop':[2.,3.,-1.],'relative_loss_drop':[.2,.1,-.3]})
    assert select(frame,'top_absolute',10,42).id.tolist()==['b','a']
    assert select(frame,'top_relative',8,42).id.tolist()==['a']
    assert select(frame,'bottom_absolute',7,42).id.tolist()==['c']


def test_token_batches_visit_every_record_once_in_exact_steps():
    rows=[{'input_tokens':int(n)} for n in np.random.default_rng(7).integers(200,8192,size=207)]
    schedule=batches(rows,32,42)
    assert len(schedule)==32 and all(schedule)
    assert sorted(i for batch in schedule for i in batch)==list(range(len(rows)))
    assert schedule==batches(rows,32,42)
    totals=[sum(rows[i]['input_tokens'] for i in batch) for batch in schedule]
    assert max(totals)-min(totals)<=2*8192


class TinyText(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embed_tokens=torch.nn.Embedding(13,5);self.linear=torch.nn.Linear(5,5);self.dropout=torch.nn.Dropout(.1)
    def forward(self,inputs_embeds,use_cache=False):
        return SimpleNamespace(last_hidden_state=self.dropout(torch.tanh(self.linear(inputs_embeds.cumsum(1)))))


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=TinyText();self.lm_head=torch.nn.Linear(5,13,bias=False)


def test_continuation_optimizer_resume_matches_uninterrupted(tmp_path,monkeypatch):
    monkeypatch.setenv('DATA_PICKING_CHECKPOINT_IO','filesystem')
    cfg=load_config('configs/pr_selection.yaml')
    cfg.update(artifact_root=str(tmp_path/'artifacts'),runtime_root=str(tmp_path/'scratch'))
    cfg['training']={'steps':4,'seed':42,'learning_rate':.002,'betas':[.9,.95],'adam_epsilon':1e-8,
                     'weight_decay':0.,'warmup_steps':2,'max_grad_norm':1.,'optimizer_cpu_threads':2,
                     'save_steps':2,'save_optimizer':True,'auto_resume':True}
    records=[{'id':str(i),'input_ids':[i+1,2,3,4,5], 'input_tokens':5,
              'answer_only_runs':[[3,5]],'answer_only_tokens':2} for i in range(8)]
    schedule=batches(records,4,42)
    torch.manual_seed(17);model=TinyLM()
    cfg['run_id']='uninterrupted-continuation';full=Engine(copy.deepcopy(model),chunk_size=2)
    full_store=CheckpointStore(cfg,'arm','fixed');train(full,records,schedule,full_store,cfg,'answer_only')
    cfg['run_id']='resumed-continuation';partial=Engine(copy.deepcopy(model),chunk_size=2)
    store=CheckpointStore(cfg,'arm','fixed')
    train(partial,records,schedule,store,cfg,'answer_only',stop_after_steps=2)
    with (store.folder/'train.jsonl').open('ab') as stream:stream.write(b'{"step":3')
    resumed=Engine(copy.deepcopy(model),chunk_size=2)
    train(resumed,records,schedule,store,cfg,'answer_only')
    for p,q in zip(full.model.parameters(),resumed.model.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    state=store.load();expected=full_store.load()
    assert state['cursor']==expected['cursor']=={'step':4,'records':8,'input_tokens':40,'target_tokens':16}
    for key,values in expected['optimizer']['state'].items():
        for name,value in values.items():torch.testing.assert_close(value,state['optimizer']['state'][key][name],rtol=0,atol=0)


def test_evaluation_uses_target_token_weights_and_resumes(tmp_path):
    class Model:
        calls=0
        def losses(self,row,objectives):
            assert objectives==['guidance'];self.calls+=1;return {'guidance':row['expected']}
    records=[{'id':'a1','task_id':'a','prediction_tokens':1,'input_tokens':10,'expected':2.},
             {'id':'a2','task_id':'a','prediction_tokens':3,'input_tokens':20,'expected':4.},
             {'id':'b','task_id':'b','prediction_tokens':2,'input_tokens':30,'expected':1.}]
    engine=Model();result=evaluate(engine,records,tmp_path/'out',{'model':'M0'},tmp_path/'local',2)
    assert result['token_mean_nll']==pytest.approx(16/6)
    assert result['task_mean_nll']==pytest.approx((3.5+1)/2)
    evaluate(engine,records,tmp_path/'out',{'model':'M0'},tmp_path/'local',2)
    assert engine.calls==3


def test_bootstrap_clusters_tasks_and_preserves_constant_improvement():
    baseline=pd.DataFrame({'prediction_tokens':[1,100,7],'nll_sum':[2.,300.,14.]},index=['a','b','c'])
    after=baseline.copy();after['nll_sum']-=.1*after.prediction_tokens
    assert task_bootstrap(baseline,after,1000,42)==pytest.approx([.1,.1])
