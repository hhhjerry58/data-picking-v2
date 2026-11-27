import errno
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from picking.pr_cluster.sampling import experiments,subset_indices,step_plan,rank_batches
from picking.pr_cluster.storage import checked_mount_copy,sha256


def test_32_jobs_have_shared_random_seeds_and_nested_percentiles():
    jobs=experiments()
    assert len(jobs)==32 and len({job['name'] for job in jobs})==32
    for objective in ['full_text','answer_only']:
        random=[job for job in jobs if job['objective']==objective and job['score'] is None]
        assert [job['seed'] for job in random]==list(range(42,50))
    candidates=pd.DataFrame({'id':[str(i) for i in range(100)],'input_tokens':[1000]*100})
    scores=pd.DataFrame([{'id':str(i),'objective':obj,'absolute_loss_drop':i,'relative_loss_drop':i/100}
                         for obj in ['full_text','answer_only'] for i in range(100)])
    chosen=[subset_indices(candidates,scores,job) for job in jobs[:4]]
    assert set(chosen[0])<set(chosen[1])<set(chosen[2])
    assert not set(chosen[0])&set(chosen[3])


def test_step_plan_cycles_without_truncation_and_pads_collectives():
    candidates=pd.DataFrame({'input_tokens':[141,2048,8192,1000,750]})
    plan=step_plan(candidates,list(range(5)),42,steps=7,input_tokens_per_step=20000)
    assert len(plan)==7
    assert any(sample['epoch']>0 for step in plan for sample in step['samples'])
    for step in plan:
        assert abs(step['input_tokens']-20000)<=8192/2
        assert step['input_tokens']==sum(candidates.iloc[sample['index']].input_tokens for sample in step['samples'])
        rank_lists=[rank_batches(step,candidates,rank,8) for rank in range(8)]
        assert len({len(rows) for rows in rank_lists})==1
        flat=[i for rows in rank_lists for i in rows if i is not None]
        assert sorted(flat)==sorted(sample['index'] for sample in step['samples'])
    assert plan==step_plan(candidates,list(range(5)),42,steps=7,input_tokens_per_step=20000)


def test_mounted_eio_retries_and_never_publishes_partial_file(tmp_path,monkeypatch):
    source=tmp_path/'source';source.write_bytes(b'complete input')
    destination=tmp_path/'destination';original=Path.open;calls=0
    def flaky(self,*args,**kwargs):
        nonlocal calls
        if self==source:
            calls+=1
            if calls==1:raise OSError(errno.EIO,'simulated mount failure')
        return original(self,*args,**kwargs)
    expected=sha256(source)
    monkeypatch.setattr(Path,'open',flaky)
    monkeypatch.setattr('picking.pr_cluster.storage.time.sleep',lambda _:None)
    checked_mount_copy(source,destination,expected,source.stat().st_size)
    assert calls==2 and destination.read_bytes()==source.read_bytes()
    with pytest.raises(OSError):checked_mount_copy(source,tmp_path/'bad','0'*64,source.stat().st_size,attempts=2)
    assert not (tmp_path/'bad').exists()
