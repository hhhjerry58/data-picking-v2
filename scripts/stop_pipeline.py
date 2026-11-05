#!/usr/bin/env python
"""Explicit operator stop; completed data and checkpoints are preserved."""
from pathlib import Path
import sys,os,signal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from picking.common import config,paths,read_json,write_json,utc_now,REPO
from picking.control_state import write_control,read_control,mirror_control,process_alive

cfg=config();p=paths(cfg)
write_control(cfg,'STOP_REQUESTED.json',{'requested_at_utc':utc_now(),'source':'explicit stop_pipeline command'})
for name in ['watchdog.json','pipeline_progress.json']:
    try:
        record=read_control(cfg,name) or {}
        pid=record['pid']
        script='watch_pipeline.py' if name=='watchdog.json' else 'run_pipeline.py'
        if not process_alive(pid,script,record.get('hostname')):continue
        proc=Path('/proc')/str(pid)
        if (proc/'cwd').resolve()!=REPO:continue
        os.kill(pid,signal.SIGTERM)
    except (OSError,KeyError):pass
mirror_control(cfg)
print('Stopped controller/watchdog. Completed checkpoints remain available.')
