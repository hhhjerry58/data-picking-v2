import json
from pathlib import Path
import pytest
from picking.common import config,paths,write_json,file_hash
from picking.runtime import allowed_gpu_ids
from picking.storage import checkpoint_transport,verified_checkpoint_mirror
from picking.control_state import process_alive


def test_gpu_selection_uses_node_configuration(monkeypatch):
    monkeypatch.delenv('DATA_PICKING_GPU_IDS',raising=False)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    assert allowed_gpu_ids([0,1,2,3])=={0,1,2,3}
    monkeypatch.setenv('DATA_PICKING_GPU_IDS','0,2')
    assert allowed_gpu_ids(iter([0,1,2,3]))=={0,2}
    monkeypatch.setenv('DATA_PICKING_GPU_IDS','9')
    with pytest.raises(ValueError,match='not present'):allowed_gpu_ids([0,1])


def test_runtime_override_preserves_numerical_training_spec(tmp_path,monkeypatch):
    monkeypatch.delenv('DATA_PICKING_RUNTIME_ROOT',raising=False)
    original=config()
    monkeypatch.setenv('DATA_PICKING_RUNTIME_ROOT',str(tmp_path/'other-node'))
    adjusted=config()
    assert adjusted['pretraining']==original['pretraining']
    assert adjusted['artifact_root']==original['artifact_root']
    assert paths(adjusted)['scratch']==tmp_path/'other-node'


def test_filesystem_transport_and_old_node_mirror(tmp_path,monkeypatch):
    cfg={'artifact_root':str(tmp_path/'persistent'),'runtime_root':str(tmp_path/'scratch'),'run_id':'test'}
    p=paths(cfg);folder=p['run']/'pretraining'/'random-seed-44'
    monkeypatch.setenv('DATA_PICKING_CHECKPOINT_IO','filesystem')
    transport=checkpoint_transport(cfg,folder)
    source=p['scratch']/'state.pt';source.parent.mkdir(parents=True);source.write_bytes(b'checkpoint-transport-test'*1000)
    destination=folder/'checkpoints/step-000001/state.pt'
    record=transport.persist(source,destination,full_readback=True)
    assert destination.read_bytes()==source.read_bytes()
    assert record['sha256']==file_hash(source)
    write_json(folder/'checkpoint_read_records/step-000001.json',{'runtime_copy':'/a/different/node/state.pt','sha256':record['sha256']})
    assert verified_checkpoint_mirror(folder,destination.parent,record['sha256'],cfg) is None
    with pytest.raises(ValueError,match='scope'):transport.persist(source,tmp_path/'outside/state.pt')
    assert not process_alive(1,'run_pipeline.py',hostname='another-node.invalid')
