#!/usr/bin/env python
"""Explicitly continue the approved run after moving to a compatible devnode."""
from pathlib import Path
import sys,os,argparse,subprocess,time,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from picking.common import config,paths,REPO,utc_now
from picking.control_state import read_control,write_control,process_alive,control_directories,mirror_control
from picking.migration import check_resume


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-for-gpu',action='store_true',help='Queue on this node when all selected GPUs are busy')
    parser.add_argument('--full-checkpoint',action='store_true',help='Hash/load the resumable checkpoint before starting')
    args=parser.parse_args()
    cfg=config();p=paths(cfg)
    status=read_control(cfg,'pipeline_progress.json') or {}
    if process_alive(status.get('pid'),'run_pipeline.py',status.get('hostname')):
        raise RuntimeError('A controller is already active on this node')
    checked=check_resume(cfg,require_gpu=not args.wait_for_gpu,full_checkpoint=args.full_checkpoint)
    # Running this command is the explicit operator resume after the old-node handoff.
    for directory in [*control_directories(cfg),p['run']]:
        for name in ['STOP_REQUESTED.json','WATCHDOG_NEEDS_ATTENTION.json']:
            (directory/name).unlink(missing_ok=True)
    logs=p['scratch']/'logs';logs.mkdir(parents=True,exist_ok=True)
    processes={}
    for script in ['run_pipeline.py','watch_pipeline.py','sync_logs.py']:
        with (logs/(script.removesuffix('.py')+'_node.log')).open('ab',buffering=0) as output:
            child=subprocess.Popen([str(REPO/'.venv/bin/python'),str(REPO/'scripts'/script)],cwd=REPO,
                                   env=dict(os.environ),stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        processes[script]=child.pid
        if script=='run_pipeline.py':
            for _ in range(40):
                state=read_control(cfg,'pipeline_progress.json') or {}
                if state.get('pid')==child.pid:break
                if child.poll() is not None:raise RuntimeError('Controller exited during startup; inspect its node log')
                time.sleep(.25)
    write_control(cfg,'devnode_resume.json',{'time':utc_now(),'processes':processes,'compatibility_check':checked})
    mirror_control(cfg)
    print(json.dumps({'started':processes,'run_directory':str(p['run']),'next':'python scripts/status.py'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
