#!/usr/bin/env python
"""Keep the authorized resumable pipeline alive across console disconnects."""
from pathlib import Path
import sys,os,time,subprocess,json,fcntl
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from picking.common import config,paths,read_json,write_json,utc_now,REPO
from picking.control_state import read_control,write_control,process_alive,stop_requested

def alive(pid):
    return process_alive(pid,'run_pipeline.py')

def main():
    cfg=config();p=paths(cfg)
    lock=(p['scratch']/'watchdog.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    attempts=0
    controller_pid=None
    write_control(cfg,'watchdog.json',{'pid':os.getpid(),'started_at_utc':utc_now(),'max_automatic_restarts':3,'restarts':attempts})
    while not stop_requested(cfg):
        try:
            if (p['run']/'EXPERIMENT_COMPLETED.json').exists():return
        except OSError:pass
        state=read_control(cfg,'pipeline_progress.json') or {}
        observed_pid=controller_pid if controller_pid is not None and alive(controller_pid) else (
            state.get('pid',0) if process_alive(state.get('pid'), 'run_pipeline.py', state.get('hostname')) else 0)
        if alive(observed_pid):
            controller_pid=observed_pid
            write_control(cfg,'watchdog.json',{'pid':os.getpid(),'controller_pid':controller_pid,'restarts':attempts,'state':'monitoring','time':utc_now()})
            time.sleep(30);continue
        if attempts>=3:
            write_control(cfg,'WATCHDOG_NEEDS_ATTENTION.json',{'reason':'Three automatic controller restarts exhausted','last_state':state,'time':utc_now()})
            return
        attempts+=1
        log=p['scratch']/'logs'/f'pipeline_restart_{attempts}.log'
        with log.open('ab',buffering=0) as output:
            process=subprocess.Popen([str(REPO/'.venv/bin/python'),str(REPO/'scripts/run_pipeline.py')],cwd=REPO,
                                     env=dict(os.environ),stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        controller_pid=process.pid
        write_control(cfg,'watchdog.json',{'pid':os.getpid(),'controller_pid':controller_pid,'restarts':attempts,'time':utc_now()})
        time.sleep(60)

if __name__=='__main__':main()
