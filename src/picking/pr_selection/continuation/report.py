from __future__ import annotations

import fcntl
import json
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

from ..config import atomic_json, identity, run_dir


def task_bootstrap(reference, treatment, samples=10000, seed=20261126):
    if set(reference.index) != set(treatment.index): raise ValueError('Task coverage differs')
    treatment = treatment.loc[reference.index]
    if not np.array_equal(reference.prediction_tokens, treatment.prediction_tokens):
        raise ValueError('Target token counts changed across evaluations')
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(reference), size=(samples, len(reference)))
    differences = reference.nll_sum.to_numpy() - treatment.nll_sum.to_numpy()
    values = differences[indexes].sum(axis=1) / reference.prediction_tokens.to_numpy()[indexes].sum(axis=1)
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def report(cfg):
    root = run_dir(cfg); current = identity(cfg)
    arms = json.loads((root/'data/arms.json').read_text())
    lanes = json.loads((root/'parallel.json').read_text())['lanes']
    baselines = [pd.read_json(root/'evaluation'/f'M0-lane-{lane}'/'losses.jsonl', lines=True).set_index('id') for lane in range(lanes)]
    for baseline in baselines[1:]:
        if set(baseline.index) != set(baselines[0].index): raise ValueError('M0 baseline coverage differs')
        baseline = baseline.loc[baselines[0].index]
        if (abs(baseline.loss-baselines[0].loss) > 2e-5*np.maximum(1., abs(baselines[0].loss))).any():
            raise ValueError('M0 loss differs across GPU pairs')
    before = json.loads((root/'evaluation/M0-lane-0/COMPLETED.json').read_text())
    baseline_tasks = pd.read_csv(root/'evaluation/M0-lane-0/tasks.csv', index_col='task_id')
    rows = []
    for arm in arms:
        completed = json.loads((root/'arms'/arm['name']/'COMPLETED.json').read_text())
        if completed['identity'] != current: raise ValueError('Arm identity differs')
        actual = completed['evaluation']
        tasks = pd.read_csv(root/'evaluation'/arm['name']/'tasks.csv', index_col='task_id')
        random = pd.read_csv(root/'evaluation'/(arm['objective']+'-random')/'tasks.csv', index_col='task_id')
        interval = task_bootstrap(baseline_tasks, tasks, cfg['evaluation']['bootstrap_samples'], cfg['evaluation']['bootstrap_seed'])
        random_interval = task_bootstrap(random, tasks, cfg['evaluation']['bootstrap_samples'], cfg['evaluation']['bootstrap_seed'])
        random_nll = float(random.nll_sum.sum()/random.prediction_tokens.sum())
        drop = before['token_mean_nll'] - actual['token_mean_nll']
        rows.append({**arm, 'test_nll': actual['token_mean_nll'], 'task_mean_nll': actual['task_mean_nll'],
                     'drop_vs_m0': drop, 'relative_drop_vs_m0': drop/before['token_mean_nll'],
                     'drop_ci_low': interval[0], 'drop_ci_high': interval[1],
                     'drop_vs_random': random_nll-actual['token_mean_nll'],
                     'random_ci_low': random_interval[0], 'random_ci_high': random_interval[1]})
    table = pd.DataFrame(rows)
    pointer = root/'latest_report.json'
    if pointer.exists(): folder = Path(json.loads(pointer.read_text())['path'])
    else: folder = Path(cfg['analysis_root'])/(datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H-%M')+'-pr-usefulness-1m')
    folder.mkdir(parents=True, exist_ok=True)
    table.to_csv(folder/'results.csv', index=False)
    atomic_json(folder/'baseline.json', before)
    shutil.copyfile(root/'effective_config.json', folder/'effective_config.json')
    shutil.copyfile(root/'data/COMPLETED.json', folder/'data_provenance.json')
    for name in ['M0-lane-0', *[arm['name'] for arm in arms]]:
        dest = folder/'evaluation'/name; dest.mkdir(parents=True, exist_ok=True)
        for file in ['losses.jsonl', 'tasks.csv', 'COMPLETED.json']:
            shutil.copyfile(root/'evaluation'/name/file, dest/file)
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell('# PR 小预算八组：独立 T 评测\n\n每组从 M0 出发、约 100 万输入 tokens、32 次更新、seed 42。T 共 53 个任务；主要指标为目标 token 平均 NLL。区间为按任务配对 bootstrap，不能反映训练 seed 的变异。'),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nroot=Path.cwd()\nresults=pd.read_csv(root/'results.csv')\nbaseline=json.loads((root/'baseline.json').read_text())\n(root/'figures').mkdir(exist_ok=True)\ndisplay(results)\nprint('M0 T NLL:',baseline['token_mean_nll'])"),
        nbformat.v4.new_code_cell("fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')\nfor ax,objective in zip(axes,['answer_only','full_text']):\n    group=results[results.objective==objective]\n    x=np.arange(len(group))\n    y=group.drop_vs_m0.to_numpy()\n    errors=np.vstack([np.maximum(0,y-group.drop_ci_low),np.maximum(0,group.drop_ci_high-y)])\n    ax.bar(x,y,color=['#4c87b0' if v>0 else '#c57658' for v in y])\n    ax.errorbar(x,y,yerr=errors,fmt='none',color='black',capsize=3)\n    ax.axhline(0,color='gray',lw=1)\n    ax.set(xticks=x,xticklabels=group.selection,ylabel='T NLL reduction vs M0 (nats/token)',title=objective)\n    ax.tick_params(axis='x',rotation=25)\nfig.suptitle('Qwen3.5-9B | 1M input tokens | 32 updates | seed 42')\nfig.savefig(root/'figures/test_loss_reduction.png',dpi=180,bbox_inches='tight')\nplt.show()"),
        nbformat.v4.new_code_cell("fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')\nfor ax,objective in zip(axes,['answer_only','full_text']):\n    group=results[results.objective==objective]\n    x=np.arange(len(group));y=group.drop_vs_random.to_numpy()\n    errors=np.vstack([np.maximum(0,y-group.random_ci_low),np.maximum(0,group.random_ci_high-y)])\n    ax.errorbar(x,y,yerr=errors,fmt='o',capsize=4)\n    ax.axhline(0,color='gray',lw=1)\n    ax.set(xticks=x,xticklabels=group.selection,ylabel='T NLL reduction vs Random',title=objective)\n    ax.tick_params(axis='x',rotation=25)\nfig.savefig(root/'figures/versus_random.png',dpi=180,bbox_inches='tight')\nplt.show()")])
    notebook.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    km = KernelManager(kernel_name='python3'); km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    NotebookClient(notebook, km=km, timeout=180, resources={'metadata': {'path': str(folder)}}).execute(cleanup_kc=True)
    nbformat.write(notebook, folder/'results.ipynb')
    lines = ['# PR 小预算八组 T 评测', '', f'M0 的 T token 平均 NLL：**{before["token_mean_nll"]:.6f}**。', '',
             '每组从原始 M0 出发，完整 PR 一遍，约 100 万输入 tokens，32 次 optimizer 更新；seed 42。',
             '训练配置和选数预算在计算 T loss 前锁定。主要指标为参考动作语义区域的 token 平均 NLL；正下降量表示改善。', '',
             '| 监督目标 | 选数 | PR 数 | 输入 tokens | 监督 tokens | T NLL | 较 M0 下降 | 较 Random 下降 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["objective"]} | {row["selection"]} | {row["records"]} | {row["input_tokens"]} | {row["target_tokens"]} | {row["test_nll"]:.6f} | {row["drop_vs_m0"]:.6f} | {row["drop_vs_random"]:.6f} |')
    lines += ['', '![T loss 下降](figures/test_loss_reduction.png)', '',
              '区间按 53 个任务做配对 bootstrap，保留每个任务全部轨迹；这是单训练 seed 的 pilot，区间不包含训练随机性。',
              '各组输入预算近似相同，监督 token 数不同；不能把差异全部归因于排序语义。没有按 T 挑选 checkpoint 或调整超参。', '',
              '[原始结果](results.csv) · [执行过的 notebook](results.ipynb) · [有效配置](effective_config.json) · [数据来源](data_provenance.json)', '',
              f'运行目录：`{root}`。复现绘图可直接重新执行本目录 notebook；训练恢复入口为 `scripts/pr_usefulness.py run --config {root}/effective_config.json`。']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n')
    archive = Path(cfg['analysis_root'])
    stamp = folder.name[:16].replace('_', ' '); stamp = stamp[:13]+':'+stamp[14:]
    with (archive.parent/'.index.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for index in [archive/'README.md', archive.parent/'README.md']:
            relative = folder.relative_to(index.parent)
            lines = index.read_text().rstrip().splitlines()
            if not any(f']({relative}/)' in line for line in lines):
                entry = f'| {stamp} | [PR 小预算八组 T 评测]({relative}/) | [报告]({relative}/REPORT.md) · [Notebook]({relative}/results.ipynb) |'
                positions = [i for i, line in enumerate(lines) if re.match(r'^\| \d{4}-', line)]
                pos = next((i for i in positions if lines[i].split('|')[1].strip()>stamp), positions[-1]+1 if positions else len(lines))
                lines.insert(pos, entry); index.write_text('\n'.join(lines)+'\n')
    atomic_json(pointer, {'path': str(folder)})
    atomic_json(root/'EXPERIMENT_COMPLETED.json', {'identity': current, 'arms': len(arms), 'report': str(folder)})
    print('EXPERIMENT_COMPLETED', folder, flush=True)
    return folder
