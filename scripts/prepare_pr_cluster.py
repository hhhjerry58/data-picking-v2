#!/usr/bin/env python3
"""等待 100k 评分完成后生成并发布不可变训练输入包，不提交集群任务。"""
import argparse
from pathlib import Path
import sys
import time

import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from picking.pr_cluster.bundle import build


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/pr_cluster.yaml'))
    parser.add_argument('--staging',default='/tmp/pr-cluster-bundle-v1')
    parser.add_argument('--wait-for-pool',action='store_true')
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--smoke-only',action='store_true',help='Publish runtime/storage smoke using three already-available PRs')
    args=parser.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())
    marker=Path(cfg['artifact_root'])/'runs'/cfg['pool_run']/'SCORING_COMPLETED.json'
    if args.wait_for_pool:
        while not marker.exists():
            status=marker.parent/'status.json'
            if status.exists():
                import json
                value=json.loads(status.read_text())
                if value.get('status')=='failed':raise RuntimeError('Local 100k scoring failed; no cluster input was published')
            print('WAITING_FOR_POOL',cfg['pool_run'],flush=True);time.sleep(45)
    build(args.config,args.staging,args.publish,args.smoke_only)


if __name__=='__main__':main()
