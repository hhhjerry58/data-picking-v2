"""启动层回归：noexec 挂载不能承载 Python，数据暂存仍可使用。"""
import errno
import importlib.util
import json
import os
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest


SCRIPT=Path(__file__).resolve().parents[1]/'scripts/pr_cluster_bootstrap.py'
spec=importlib.util.spec_from_file_location('pr_cluster_bootstrap',SCRIPT)
bootstrap=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def test_noexec_runtime_is_skipped_but_data_can_use_it(tmp_path,monkeypatch,capsys):
    noexec=tmp_path/'shm';noexec.mkdir()
    executable=tmp_path/'scratch';executable.mkdir()
    monkeypatch.setattr(bootstrap.os,'statvfs',lambda path:SimpleNamespace(f_flag=os.ST_NOEXEC if path==noexec else 0))
    monkeypatch.setattr(bootstrap.shutil,'disk_usage',lambda path:SimpleNamespace(free=200*1024**3))

    runtime=bootstrap.local_directory([noexec,executable],prefix='runtime-',min_free_gib=32,executable=True)
    staging=bootstrap.local_directory([noexec],prefix='data-',min_free_gib=160)

    assert runtime.parent==executable and staging.parent==noexec
    assert list(runtime.iterdir())==[]
    assert 'noexec' in capsys.readouterr().out


def test_actual_exec_denial_falls_back_and_removes_probe(tmp_path,monkeypatch):
    denied=tmp_path/'denied';denied.mkdir()
    allowed=tmp_path/'allowed';allowed.mkdir()
    original_run=bootstrap.subprocess.run
    monkeypatch.setattr(bootstrap.os,'statvfs',lambda path:SimpleNamespace(f_flag=0))
    monkeypatch.setattr(bootstrap.shutil,'disk_usage',lambda path:SimpleNamespace(free=200*1024**3))

    def run(command,**kwargs):
        if Path(command[0]).is_relative_to(denied):
            raise PermissionError(errno.EACCES,'simulated execute denial',command[0])
        return original_run(command,**kwargs)

    monkeypatch.setattr(bootstrap.subprocess,'run',run)
    selected=bootstrap.local_directory([denied,allowed],prefix='runtime-',min_free_gib=32,executable=True)
    assert selected.parent==allowed
    assert list(denied.iterdir())==[] and list(selected.iterdir())==[]


def test_missing_small_and_unwritable_roots_are_reported(tmp_path,monkeypatch):
    missing=tmp_path/'missing'
    small=tmp_path/'small';small.mkdir()
    denied=tmp_path/'denied';denied.mkdir()
    original_mkdtemp=bootstrap.tempfile.mkdtemp

    def disk_usage(path):
        if path==missing:raise FileNotFoundError(errno.ENOENT,'missing root',str(path))
        return SimpleNamespace(free=(10 if path==small else 200)*1024**3)

    def mkdtemp(*args,**kwargs):
        if kwargs['dir']==denied:raise PermissionError(errno.EACCES,'not writable',str(denied))
        return original_mkdtemp(*args,**kwargs)

    monkeypatch.setattr(bootstrap.shutil,'disk_usage',disk_usage)
    monkeypatch.setattr(bootstrap.tempfile,'mkdtemp',mkdtemp)
    with pytest.raises(RuntimeError) as failure:
        bootstrap.local_directory([missing,small,denied],prefix='data-',min_free_gib=160)
    message=str(failure.value)
    assert all(str(root) in message for root in [missing,small,denied])
    assert 'need 160 GiB free' in message and 'not writable' in message


def test_explicit_noexec_runtime_fails_before_copying_archives(tmp_path,monkeypatch):
    monkeypatch.setattr(bootstrap.os,'statvfs',lambda path:SimpleNamespace(f_flag=os.ST_NOEXEC))
    with pytest.raises(RuntimeError,match='No usable executable runtime directory.*noexec'):
        bootstrap.local_directory([tmp_path],prefix='runtime-',min_free_gib=32,executable=True)
    assert list(tmp_path.iterdir())==[]


def test_bootstrap_launches_archived_runtime_with_separate_staging(tmp_path,monkeypatch):
    bundle=tmp_path/'bundle';bundle.mkdir()
    source=tmp_path/'source';source.mkdir()
    # 轻量可执行替身记录真实子进程的参数与环境，不下载模型或启动 GPU。
    python=source/'python';python.write_text(
        f'#!{sys.executable}\n'
        'import json, os, sys\n'
        'if sys.argv[1] == "-c":\n'
        '    exec(sys.argv[2])\n'
        'else:\n'
        '    with open(os.environ["BOOTSTRAP_TEST_LOG"], "w") as out:\n'
        '        json.dump({"argv": sys.argv, "env": dict(os.environ)}, out)\n'
        '    sys.exit(23)\n'
    );python.chmod(0o755)
    code=source/'code';code.write_text('# frozen source\n')
    archives={}
    for name,path,arcname in [('runtime.tar.gz',python,'python/bin/python3.12'),
                              ('code.tar.gz',code,'src/picking/__init__.py')]:
        destination=bundle/name
        with tarfile.open(destination,'w:gz') as archive:archive.add(path,arcname=arcname)
        archives[name]={'bytes':destination.stat().st_size,'sha256':bootstrap.checksum(destination)}
    manifest={'smoke_only':True,'archives':archives,'protocol':{'bundle_prefix':'frozen-smoke'}}
    (bundle/'manifest.json').write_text(json.dumps(manifest))
    identity=bootstrap.hashlib.sha256(json.dumps(manifest,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    (bundle/'READY.json').write_text(json.dumps({'manifest_identity':identity}))
    runtime_root=tmp_path/'runtime-root';runtime_root.mkdir()
    staging_root=tmp_path/'staging-root';staging_root.mkdir()
    log=tmp_path/'launch.json'
    monkeypatch.setenv('BOOTSTRAP_TEST_LOG',str(log))
    monkeypatch.setenv('TMPDIR','/dev/shm/noexec-temp')
    monkeypatch.setattr(sys,'argv',[str(SCRIPT),'--bundle',str(bundle),'--job','example','--mode','smoke',
                                  '--runtime-root',str(runtime_root),'--staging-root',str(staging_root)])
    # 小测试文件不实际占用 160 GiB；容量检查必须发生在 runtime 已解压后。
    def disk_usage(path):
        if path==staging_root:
            assert list(runtime_root.glob('*/runtime/python/bin/python3.12'))
        return SimpleNamespace(free=200*1024**3)
    monkeypatch.setattr(bootstrap.shutil,'disk_usage',disk_usage)
    with pytest.raises(SystemExit) as result:bootstrap.main()
    assert result.value.code==23
    launched=json.loads(log.read_text());argv=launched['argv'];env=launched['env']
    assert Path(argv[0]).is_relative_to(runtime_root)
    assert argv[1:6]==['-u','-m','picking.pr_cluster.node','--bundle-prefix','frozen-smoke']
    assert argv[-4:]==['--job','example','--mode','smoke']
    assert Path(argv[argv.index('--local')+1]).is_relative_to(staging_root)
    for key in ['TMPDIR','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR']:
        assert Path(env[key]).is_relative_to(runtime_root)
    assert all(Path(path).is_relative_to(runtime_root) for path in env['PYTHONPATH'].split(':'))
    assert env['DATA_PICKING_GPU_IDS']=='0,1,2,3,4,5,6,7'

