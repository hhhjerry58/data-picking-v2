import copy
from pathlib import Path
import pytest

from picking.common import config,paths,write_json,file_hash,digest,REPO
from picking.workflow import completed_job,training_complete


def test_finished_training_is_skipped_and_partial_training_is_queued(tmp_path):
    cfg=copy.deepcopy(config())
    cfg.update(artifact_root=str(tmp_path/'artifacts'),runtime_root=str(tmp_path/'runtime'),run_id='test-resume')
    folder=paths(cfg)['run']/'pretraining/random-seed-42'
    checkpoint=folder/'checkpoints/step-012213'
    checkpoint.mkdir(parents=True)
    state=checkpoint/'state.pt';state.write_bytes(b'unit-test-state')
    manifest={'fingerprint':'test-fingerprint','seen_tokens':3200000000,'save_optimizer':True,'auto_resume':True,
              'files':{'state.pt':{'size':state.stat().st_size,'sha256':file_hash(state)}}}
    write_json(checkpoint/'manifest.json',manifest)
    write_json(checkpoint/'COMPLETED.json',{'manifest_hash':digest(manifest)})
    write_json(folder/'TRAINING_COMPLETED.json',{'fingerprint':'test-fingerprint','seen_tokens':3200000000,'checkpoint':str(checkpoint)})
    write_json(folder/'effective_config.json',{'fingerprint':'test-fingerprint','identity':{
        'training':cfg['pretraining'],'implementation_sha256':file_hash(REPO/'src/picking/pretraining.py')}})
    assert completed_job(cfg,['pretrain','--arm','random','--seed','42'])
    assert not completed_job(cfg,['pretrain','--arm','relative_gain','--seed','43'])
    manifest['seen_tokens']=3013184000
    write_json(checkpoint/'manifest.json',manifest)
    write_json(checkpoint/'COMPLETED.json',{'manifest_hash':digest(manifest)})
    with pytest.raises(RuntimeError,match='token budget'):
        training_complete(cfg,'random',42)
