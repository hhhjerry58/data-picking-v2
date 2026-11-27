#!/usr/bin/env python3
"""只读检查 smoke、100k 输入包和 32 个集群任务的状态。"""
import argparse
from pathlib import Path
import sys
import json
import yaml

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from picking.pr_cluster.storage import BlobStore
from picking.pr_cluster.sampling import experiments


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/pr_cluster.yaml'))
    args=parser.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())
    store=BlobStore(cfg['storage_account'],cfg['storage_container'])
    for key,label in [(cfg['bundle_prefix']+'-smoke/READY.json','smoke_bundle'),(cfg['bundle_prefix']+'/READY.json','training_bundle')]:
        response=store.request('GET',key,accepted=(200,404))
        print(label,response.json() if response.status_code==200 else 'not_published')
    for job in [{'name':experiments()[0]['name']+'-smoke'},*experiments()]:
        response=store.request('GET',cfg['output_prefix']+'/'+job['name']+'/STATE.json',accepted=(200,404))
        if response.status_code==404:print(job['name'],'not_started');continue
        state=response.json();latest=state.get('latest') or {}
        print(job['name'],json.dumps({'completed':state.get('completed',False),'committed_step':latest.get('step',0),
              'progress_step':state.get('progress',{}).get('step',0),'final_nll':state.get('final_nll')}))


if __name__=='__main__':main()
