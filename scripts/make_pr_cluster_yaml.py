#!/usr/bin/env python3
"""生成 32 个单节点任务及 smoke；本脚本不执行 amlt 提交。"""
import argparse
import os
from pathlib import Path
import sys

import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from picking.pr_cluster.sampling import experiments


def configuration(jobs,bundle,mode,code_local_dir='$CONFIG_DIR/../../..'):
    description=('Qwen3.5-9B A100 smoke: storage, FSDP and checkpoint recovery' if mode=='smoke'
                 else f'Qwen3.5-9B PR selection: {len(jobs)} independent 8xA100 jobs, 100 updates, checkpoint every 5')
    return {'description':description,
            'target':{'service':'sing','name':'turing-codex-westus3','workspace_name':'Workspace_NLC'},
            'environment':{'image':'6eoye/verl2:v0.3.0-vllm0112-cu128'},
            'code':{'local_dir':code_local_dir,
                    'ignore':['.git/','.venv/','.venv-pr/','.cache/','.runtime/','.pytest_cache/',
                              '**/__pycache__/','**/*.pyc','.env','.env.*','blob/']},
            'storage':{'tianzhu':{'storage_account_name':'msranlpinternhot','container_name':'tianzhu'}},
            'jobs':[{'name':job['name']+('-smoke' if mode=='smoke' else ''),'identity':'managed','priority':'high',
                     'sku':'1xG8-A100','process_count_per_node':1,
                     'command':[f'python3 scripts/pr_cluster_bootstrap.py --bundle {bundle} --job {job["name"]} --mode {mode}']}
                    for job in jobs]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=str(ROOT/'yamls/guanheng/pr_selection_100k'))
    parser.add_argument('--bundle',default='/mnt/tianzhu/data-picking-v2/cluster_bundles/pr-100k-a100-100step-v1-20261126')
    args=parser.parse_args();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    import re
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+',args.bundle):parser.error('Invalid bundle path')
    code_local_dir='$CONFIG_DIR/'+os.path.relpath(ROOT,output)
    jobs=experiments()
    for name,selected,mode in [('smoke_a100',jobs[:1],'smoke'),('train_a100',jobs,'train'),
                               *[(f'train_batch_{i+1}_a100',jobs[i*8:(i+1)*8],'train') for i in range(4)]]:
        bundle=args.bundle+'-smoke' if mode=='smoke' else args.bundle
        relative=os.path.relpath(output/(name+'.yaml'),ROOT)
        header=f'# 在仓库根目录、已登录的 AMLT 环境中提交：\n# amlt run {relative} pr100k-{name}-v1 -y --sla Premium\n\n'
        content=yaml.safe_dump(configuration(selected,bundle,mode,code_local_dir),allow_unicode=True,sort_keys=False,width=110)
        (output/(name+'.yaml')).write_text(header+content)
    print('GENERATED',len(jobs),'jobs in',output)


if __name__=='__main__':main()
