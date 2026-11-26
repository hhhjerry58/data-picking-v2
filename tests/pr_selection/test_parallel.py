"""针对分布式评分的数据覆盖、身份隔离和合并做 CPU 检查。"""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from picking.common import REPO, digest, file_hash
from picking.pr_parallel.backend import gradient_statistics
from picking.pr_parallel.cluster import load_flat, merge, partition_indices, validate_numbers
from picking.pr_parallel.deployment import read_deployment, verify_execution
from picking.pr_selection.config import atomic_json


def test_full_model_fp64_dot_with_chunking():
    torch.manual_seed(42)
    parameters={name:torch.nn.Parameter(torch.randn(n)) for name,n in [('a',71),('b',25)]}
    reference={name:torch.randn_like(p) for name,p in parameters.items()}
    for p in parameters.values():p.grad=torch.randn_like(p)
    actual=gradient_statistics(SimpleNamespace(parameters=parameters),reference,chunk_elements=7)
    g=torch.cat([p.grad.double() for p in parameters.values()])
    q=torch.cat([reference[n].double() for n in parameters])
    assert actual==pytest.approx((float(g@q),float(g.norm())),rel=1e-12,abs=1e-12)


def test_forty_shards_cover_twenty_thousand_once():
    partitions=[list(partition_indices(20000,i,40)) for i in range(40)]
    assert all(len(p)==500 for p in partitions)
    assert sorted(i for part in partitions for i in part)==list(range(20000))
    with pytest.raises(ValueError):partition_indices(20000,40,40)


def test_flat_tensor_names_shape_dtype_and_completeness(tmp_path):
    path=tmp_path/'bundle/gq';path.mkdir(parents=True)
    save_file({'a':torch.ones(3),'b':torch.zeros(2)},path/'0.safetensors')
    descriptions=[{'file':'gq/0.safetensors','names':['a','b']}]
    parameters={'a':torch.nn.Parameter(torch.zeros(3)),'b':torch.nn.Parameter(torch.ones(2))}
    reference=load_flat(tmp_path,descriptions,parameters,reference=True)
    assert reference['a'].equal(torch.ones(3))
    assert parameters['a'].equal(torch.zeros(3))
    load_flat(tmp_path,descriptions,parameters,reference=False)
    assert parameters['a'].equal(reference['a'])
    for tensors,names in [({'a':torch.ones(4),'b':torch.zeros(2)},['a','b']),
                          ({'a':torch.ones(3,dtype=torch.bfloat16),'b':torch.zeros(2)},['a','b']),
                          ({'a':torch.ones(3)},['a'])]:
        save_file(tensors,path/'0.safetensors')
        with pytest.raises(ValueError):load_flat(tmp_path,[{'file':'gq/0.safetensors','names':names}],parameters,reference=True)


def make_bundle(tmp_path):
    bundle=tmp_path/'bundle';bundle.mkdir()
    pd.DataFrame({'id':[f'pr-{i}' for i in range(8)]}).to_parquet(bundle/'candidates.parquet')
    manifest={'records':8,'candidate_file':{'sha256':file_hash(bundle/'candidates.parquet')},
              'parent_config':{'objectives':['full_text','answer_only'],
                               'methods':['absolute_loss_drop','relative_loss_drop','grad_dot']}}
    deployment={'nodes':1,'workers_per_node':4,'bundle_identity':digest(manifest)}
    atomic_json(bundle/'manifest.json',manifest);atomic_json(bundle/'deployment.json',deployment)
    atomic_json(bundle/'READY.json',{'bundle_identity':digest(manifest),'deployment_identity':digest(deployment)})
    outputs=tmp_path/'outputs'
    for worker in range(4):
        folder=outputs/'node-00'/f'worker-{worker:03d}';folder.mkdir(parents=True)
        frame=pd.DataFrame([{'id':f'pr-{i}','objective':obj,'loss_before':2.,'loss_after':1.,'gradient_norm':3.,
                             'absolute_loss_drop':1.,'relative_loss_drop':.5,'grad_dot':2.}
                            for i in partition_indices(8,worker,4) for obj in ['full_text','answer_only']])
        frame.to_parquet(folder/'scores.parquet')
        atomic_json(folder/'COMPLETED.json',{'worker':worker,'workers':4,'records':2,'bundle':digest(manifest),
                    'deployment':digest(deployment),'scores_sha256':file_hash(folder/'scores.parquet')})
    return bundle,outputs


def test_merge_complete_then_reject_wrong_partition_and_identity(tmp_path):
    bundle,outputs=make_bundle(tmp_path)
    merge(outputs,bundle,tmp_path/'merged')
    assert len(pd.read_parquet(tmp_path/'merged/scores.parquet'))==16
    folder=outputs/'node-00/worker-000'
    frame=pd.read_parquet(folder/'scores.parquet');frame.loc[0,'id']='pr-1'
    frame.to_parquet(folder/'scores.parquet')
    meta=json.loads((folder/'COMPLETED.json').read_text());meta['scores_sha256']=file_hash(folder/'scores.parquet')
    atomic_json(folder/'COMPLETED.json',meta)
    with pytest.raises(ValueError,match='wrong partition'):merge(outputs,bundle,tmp_path/'bad')
    ready=json.loads((bundle/'READY.json').read_text());ready['deployment_identity']='changed'
    atomic_json(bundle/'READY.json',ready)
    with pytest.raises(ValueError,match='identity differs'):read_deployment(bundle)


def test_execution_identity_rejects_source_change(tmp_path):
    source=tmp_path/'scoring.py';source.write_text('version_one')
    deployment={'source_files':{'scoring.py':file_hash(source)},'packages':{}}
    verify_execution(deployment,tmp_path)
    source.write_text('version_two')
    with pytest.raises(ValueError,match='source differs'):verify_execution(deployment,tmp_path)


def test_predeclared_cross_device_bounds_do_not_relax():
    expected={'loss_before':1.,'grad_dot':2.,'gradient_norm':3.}
    actual={'loss_before':1.,'gradient_dot':2.,'gradient_norm':3.}
    assert validate_numbers(actual,expected,4.)['passed']
    for key,value in [('loss_before',1.001),('gradient_dot',2.1),('gradient_norm',3.1)]:
        changed={**actual,key:value}
        assert not validate_numbers(changed,expected,4.)['passed']


def test_worker_reuses_fixed_guidance_and_switches_to_m1_then_resumes(tmp_path,monkeypatch):
    import picking.pr_parallel.cluster as cluster
    local=tmp_path/'local';bundle=local/'bundle';bundle.mkdir(parents=True)
    output=tmp_path/'output';output.mkdir()
    cfg={'objectives':['full_text','answer_only'],
         'methods':['absolute_loss_drop','relative_loss_drop','grad_dot'],'data':{'sample_seed':42}}
    atomic_json(local/'engine_config.json',cfg)
    candidates=[{'id':f'pr-{i}','input_tokens':7,'full_text_tokens':6,'answer_only_tokens':2} for i in range(8)]
    pd.DataFrame(candidates).to_parquet(bundle/'candidates.parquet')
    save_file({'weight':torch.tensor([2.])},bundle/'gq.safetensors')
    save_file({'weight':torch.tensor([.5])},bundle/'m1.safetensors')
    manifest={'records':8,'gq_norm':2.,'gq_files':[{'file':'gq.safetensors','names':['weight']}],
              'm1_files':[{'file':'m1.safetensors','names':['weight']}]}
    atomic_json(local/'manifest.json',manifest)
    atomic_json(local/'deployment.json',{'nodes':1,'workers_per_node':4,'expected_gpu':'fake'})
    atomic_json(bundle/'validation.json',[{'id':'pr-0','objective':obj,'loss_before':scale,
                'loss_after':scale/2,'grad_dot':2*scale,'gradient_norm':scale}
                for obj,scale in [('full_text',1.),('answer_only',2.)]])

    calls=[]
    class Engine:
        def __init__(self):self.parameters={'weight':torch.nn.Parameter(torch.ones(1))}
        def zero_grad(self):self.parameters['weight'].grad=None
        def backward(self,row,objective):
            scale=1. if objective=='full_text' else 2.
            self.parameters['weight'].grad=torch.tensor([scale])
            calls.append(('backward',row['id'],objective))
            return float(self.parameters['weight'].detach())*scale
        def losses(self,row,objectives):
            calls.append(('losses',row['id']))
            return {obj:float(self.parameters['weight'].detach())*(1. if obj=='full_text' else 2.) for obj in objectives}
    monkeypatch.setattr(cluster,'verify_execution',lambda _:None)
    monkeypatch.setattr(cluster,'load_engine',lambda *_:Engine())
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda _:'fake gpu')
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda _:(8,0))
    cluster.worker(local,output,0,4)
    frame=pd.read_parquet(output/'scores.parquet')
    assert set(frame.id)=={'pr-0','pr-4'} and len(frame)==4
    assert frame.relative_loss_drop.tolist()==[.5]*4
    assert frame.grad_dot.tolist()==[2.,4.,2.,4.]
    calls.clear();cluster.worker(local,output,0,4)
    # Only the common validation example is recomputed; completed candidates resume from journals.
    assert all(call[1]=='pr-0' for call in calls)


def test_submit_configuration_and_embedded_powershell(tmp_path):
    spec=importlib.util.spec_from_file_location('submit_pr_a100',REPO/'scripts/submit_pr_a100.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    full=module.configuration('campaign','/mnt/tianzhu/bundle')
    smoke=module.configuration('campaign','/mnt/tianzhu/bundle',validate_only=True)
    assert len(full['jobs'])==10 and all(job['sku']=='1xG8-A100' for job in full['jobs'])
    assert len(smoke['jobs'])==1 and 'campaign-smoke' in smoke['jobs'][0]['command'][0]
    module.windows_wrapper(tmp_path/'submit.ps1')
    ps=(tmp_path/'submit.ps1').read_text()
    import base64,gzip,re
    encoded=re.search(r"Replace\('__PAYLOAD__', '([A-Za-z0-9+/=]+)'\)",ps).group(1)
    assert gzip.decompress(base64.b64decode(encoded))==(REPO/'scripts/submit_pr_a100.py').read_bytes()
    embedded=ps.split("$python = @'\n",1)[1].split("\n'@",1)[0]
    compile(embedded,'remote_submit','exec')
