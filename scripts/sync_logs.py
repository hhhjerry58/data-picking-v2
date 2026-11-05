#!/usr/bin/env python
"""Persist closed snapshots of POSIX live logs without blocking compute jobs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import time,shutil
from picking.common import config,paths
from picking.control_state import mirror_control

def main():
    cfg=config();p=paths(cfg);last={}
    while True:
        mirror_control(cfg)
        items=[(path,p['run']/path.name) for path in (p['scratch']/'logs').glob('*.log')]
        progress=p['scratch']/'corpus_state/data_progress.json'
        if progress.exists():items.append((progress,p['run']/'data_progress.json'))
        for source,destination in items:
            try:
                modified=source.stat().st_mtime_ns
                if last.get(str(source))==modified:continue
                shutil.copyfile(source,destination)
                last[str(source)]=modified
            except (OSError,FileNotFoundError):
                continue
        try:
            if (p['run']/'EXPERIMENT_COMPLETED.json').exists():return
        except OSError:pass
        time.sleep(30)

if __name__=='__main__':main()
