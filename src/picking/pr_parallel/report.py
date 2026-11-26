from __future__ import annotations

import argparse
import fcntl
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import nbformat
import pandas as pd
from jupyter_client import KernelManager
from nbclient import NotebookClient

from picking.common import REPO
from picking.pr_selection.config import atomic_json
from picking.pr_selection.diagnostics import statistics


def report(run):
    run=Path(run)
    if not (run/'COMPLETED.json').exists():raise ValueError('Preview is not complete')
    manifest=json.loads((run/'manifest.json').read_text())
    pointer=run/'latest_report.json'
    root=Path('/home/v-guanhechen/experiments')
    if pointer.exists():folder=Path(json.loads(pointer.read_text())['path'])
    else:
        stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H-%M')
        folder=root/(stamp+'-pr-score-preview')
    folder.mkdir(parents=True,exist_ok=True)
    frame=pd.read_parquet(run/'scores.parquet')
    frame.to_parquet(folder/'scores.parquet',index=False)
    summary=statistics(frame)
    summary.to_csv(folder/'distribution_statistics.csv',index=False)
    for objective in manifest['parent_config']['objectives']:
        group=frame[frame.objective==objective]
        group[[*manifest['parent_config']['methods'],'loss_before','input_tokens','prediction_tokens','gradient_norm']].corr(method='spearman').to_csv(folder/f'correlations_{objective}.csv')
    atomic_json(folder/'effective_config.json',{**manifest['parent_config'],'preview':{
        'records':manifest['records'],'full_pool_records':manifest['parent_records'],'scope':manifest['scope']}})
    atomic_json(folder/'provenance.json',manifest)
    notebook=nbformat.read(REPO/'notebooks/score_distributions.ipynb',as_version=4)
    notebook.cells[0].source=(f'# PR 六个评分分布：{manifest["records"]} 条先导预览\n\n'
        f'来自原 {manifest["parent_records"]} 条池中已经完成的 ID 排序前缀，**不是全池的随机样本**。'
        '完整保留原 M0 loss/GradDot；M1 使用相同权重、原始输入和两个 mask 补算。'
        '只用于快速检查评分信号，不能据此宣称全池分布或目标测试集收益。')
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    NotebookClient(notebook,km=manager,timeout=600,resources={'metadata':{'path':str(folder)}}).execute(cleanup_kc=True)
    nbformat.write(notebook,folder/'score_distributions.ipynb')
    lines=['# PR 六个分布的先导预览','',
           f'已完成 **{manifest["records"]} / {manifest["parent_records"]}** 条 PR 的双目标、三方法配对分数。',
           '', '**范围限制：这是原任务按 ID 排序先算完的前缀，不是随机抽样，不能代表整个候选池。**',
           '该结果用来尽快检查信号与量级；尚未选数、继续训练 PR 或计算 T loss。','',
           f'原运行：`{manifest["parent_run"]}`。预览运行：`{run}`。',
           'M1 与 Q 梯度均复用原训练结果；单卡 M0 已对照原四卡缓存，各 M1 worker 也做了共同样例一致性检查。','',
           '| 监督目标 | 方法 | 正分比例 | 中位数 | 1% 分位 | 99% 分位 |','|---|---|---:|---:|---:|---:|']
    for row in summary.to_dict('records'):
        lines.append(f'| {row["objective"]} | {row["method"]} | {row["positive_fraction"]:.1%} | {row["q50"]:.6g} | {row["q1"]:.6g} | {row["q99"]:.6g} |')
    lines += ['', '![六个分布](figures/six_distributions.png)','',
              '[执行过的 Notebook](score_distributions.ipynb) · [统计](distribution_statistics.csv) · [来源](provenance.json)','',
              '复现：在代码仓库执行：','',f'`PYTHONPATH=src .venv-pr/bin/python -m picking.pr_parallel.report --run {run}`','']
    (folder/'REPORT.md').write_text('\n'.join(lines))
    with (root/'.index.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        index=root/'README.md';content=index.read_text().rstrip().splitlines()
        if not any(f']({folder.name}/)' in line for line in content):
            stamp=folder.name[:16].replace('_',' ').rsplit('-',1);timestamp=stamp[0]+':'+stamp[1]
            entry=f'| {timestamp} | [PR 2,141 条先导评分预览]({folder.name}/) | [报告]({folder.name}/REPORT.md) · [Notebook]({folder.name}/score_distributions.ipynb) |'
            existing=[i for i,line in enumerate(content) if re.match(r'^\| \d{4}-\d{2}-\d{2} \d{2}:\d{2} \|',line)]
            pos=next((i for i in existing if content[i].split('|')[1].strip()>timestamp),existing[-1]+1 if existing else len(content))
            content.insert(pos,entry);index.write_text('\n'.join(content)+'\n')
    atomic_json(pointer,{'path':str(folder)})
    print(folder,flush=True)
    return folder


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    report(parser.parse_args().run)
