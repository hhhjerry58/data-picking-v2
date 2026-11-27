from __future__ import annotations

import json
import fcntl
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import nbformat
import numpy as np
import pandas as pd
from jupyter_client import KernelManager
from nbclient import NotebookClient

from picking.common import REPO
from .config import METHODS, OBJECTIVES, atomic_json, run_dir


def statistics(frame, methods=METHODS):
    rows = []
    for objective in OBJECTIVES:
        group = frame[frame.objective == objective]
        for name in methods:
            values = group[name].to_numpy(float)
            if not len(values) or not np.isfinite(values).all():
                raise ValueError('Distributions require complete finite scores')
            rows.append({'objective': objective, 'method': name, 'count': len(values),
                         'mean': float(values.mean()), 'std': float(values.std()),
                         'min': float(values.min()), 'max': float(values.max()),
                         'positive_fraction': float((values > 0).mean()),
                         'negative_fraction': float((values < 0).mean()),
                         'zero_fraction': float((values == 0).mean()),
                         **{f'q{p:g}': float(np.percentile(values, p)) for p in (1, 5, 25, 50, 75, 95, 99)}})
    return pd.DataFrame(rows)


def analyze(cfg):
    run = run_dir(cfg)
    if not (run / 'SCORING_COMPLETED.json').is_file():
        raise ValueError('All four distributions require completed paired scoring')
    frame = pd.read_parquet(run / 'scores/paired_scores.parquet')
    pointer = run / 'latest_report.json'
    if pointer.exists():
        folder = Path(json.loads(pointer.read_text())['path'])
    else:
        stamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H-%M')
        folder = Path(cfg['analysis_root']) / f'{stamp}-pr-score-distributions-{cfg["run_id"]}'
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / 'scores.parquet', index=False)
    summary = statistics(frame, cfg['methods'])
    summary.to_csv(folder / 'distribution_statistics.csv', index=False)
    shutil.copyfile(run / 'effective_config.json', folder / 'effective_config.json')
    shutil.copyfile(run / 'data/COMPLETED.json', folder / 'data_summary.json')
    correlations = []
    for objective in OBJECTIVES:
        group = frame[frame.objective == objective]
        corr = group[[*cfg['methods'], 'loss_before', 'input_tokens', 'prediction_tokens']].corr(method='spearman')
        corr.to_csv(folder / f'correlations_{objective}.csv')
    notebook = nbformat.read(REPO / 'notebooks/score_distributions.ipynb', as_version=4)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    NotebookClient(notebook, km=manager, timeout=600,
                   resources={'metadata': {'path': str(folder)}}).execute(cleanup_kc=True)
    nbformat.write(notebook, folder / 'score_distributions.ipynb')
    (folder / 'REPORT.md').write_text(
        '# PR 四个评分分布\n\n'
        f'评分运行：`{run}`。同一候选池，两种监督目标，两种方法。\n\n'
        '已完成指导集 probe 和候选评分；没有选数、PR 继续训练或测试集 loss。\n'
        '分布只能诊断候选信号，不能证明对独立目标测试集的训练效用。\n\n'
        '- [分布统计](distribution_statistics.csv)\n'
        '- [执行过的 notebook](score_distributions.ipynb)\n'
        '- [逐条分数](scores.parquet)\n'
        '- [有效配置](effective_config.json)\n\n'
        f'复现：在代码仓库运行 `.venv-pr/bin/python scripts/pr_selection.py analyze --config {run}/effective_config.json`。\n')
    archive = Path(cfg['analysis_root'])
    timestamp = folder.name[:16].replace('_', ' ')
    timestamp = timestamp[:13] + ':' + timestamp[14:]
    with (archive.parent/'.index.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for index in (archive/'README.md', archive.parent/'README.md'):
            relative = folder.relative_to(index.parent)
            lines = index.read_text().rstrip().splitlines() if index.exists() else ['# 训练实验分析', '', '| 分析时间 | 实验 | 报告 |', '|---|---|---|']
            if not any(f']({relative}/)' in line for line in lines):
                entry = f'| {timestamp} | [PR 四个评分分布]({relative}/) | [报告]({relative}/REPORT.md) · [Notebook]({relative}/score_distributions.ipynb) |'
                existing = [i for i, line in enumerate(lines) if re.match(r'^\| \d{4}-\d{2}-\d{2} \d{2}:\d{2} \|', line)]
                insert = next((i for i in existing if lines[i].split('|')[1].strip() > timestamp), existing[-1]+1 if existing else len(lines))
                lines.insert(insert, entry)
                index.write_text('\n'.join(lines)+'\n')
    atomic_json(pointer, {'path': str(folder)})
    return folder
