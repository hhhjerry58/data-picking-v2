import errno
from pathlib import Path

from picking import control_state


def settings(tmp_path, monkeypatch):
    monkeypatch.setattr(control_state, 'REPO', tmp_path/'repo')
    return {'artifact_root':str(tmp_path/'mount'), 'runtime_root':str(tmp_path/'ram'), 'run_id':'io-test'}


def test_mount_failure_does_not_prevent_control_updates_or_operator_stop(tmp_path, monkeypatch):
    cfg=settings(tmp_path,monkeypatch)
    original=Path.open
    def unreliable(path,*args,**kwargs):
        if path.is_relative_to(Path(cfg['artifact_root'])):
            raise OSError(errno.EIO,'injected artifact-mount failure')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',unreliable)
    control_state.write_control(cfg,'pipeline_progress.json',{'stage':'pretraining_seed_43','pid':123})
    control_state.write_control(cfg,'pipeline_progress.json',{'stage':'pretraining_seed_43','state':'waiting_for_gpu','pid':123})
    assert control_state.read_control(cfg,'pipeline_progress.json')['state']=='waiting_for_gpu'
    assert control_state.mirror_control(cfg)['failed']==['pipeline_progress.json']
    control_state.write_control(cfg,'STOP_REQUESTED.json',{'source':'explicit operator stop'})
    assert control_state.stop_requested(cfg)
    assert not control_state.process_alive(999999999,'run_pipeline.py')


def test_local_disk_backup_survives_loss_of_ram_spool(tmp_path, monkeypatch):
    cfg=settings(tmp_path,monkeypatch)
    original=Path.open
    def unavailable_ram(path,*args,**kwargs):
        if path.is_relative_to(Path(cfg['runtime_root'])):
            raise OSError(errno.ENOSPC,'injected RAM spool failure')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',unavailable_ram)
    control_state.write_control(cfg,'watchdog.json',{'pid':42,'controller_pid':43})
    assert control_state.read_control(cfg,'watchdog.json')['controller_pid']==43
