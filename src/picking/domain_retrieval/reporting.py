from __future__ import annotations

import fcntl
from importlib.metadata import version
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from zoneinfo import ZoneInfo

import nbformat
import pandas as pd
import yaml
from jupyter_client import KernelManager
from nbclient import NotebookClient

from picking.common import REPO, digest, read_json
from .config import run_dir, write_json


def figure_notebook(folder):
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(
            '# 领域检索与继续训练\n\n'
            '数据来自同目录冻结统计；每张图的绘图代码和执行输出都保存在本 notebook。\n\n'
            '检索正例是来源类别标签。继续训练各组从同一个预训练 M0 出发，'
            '在从未用于本次训练或排序的独立测试文档上比较 token 加权 NLL。'),
        nbformat.v4.new_code_cell(dedent('''\
            %matplotlib inline
            import json
            from pathlib import Path
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt

            folder = Path.cwd()
            summary = json.loads((folder/'metrics.json').read_text())
            table = pd.read_csv(folder/'metrics.csv')
            figures = folder/'figures'
            figures.mkdir(exist_ok=True)
            names = {'absolute_loss_drop': 'Absolute loss drop', 'relative_loss_drop': 'Relative loss drop',
                     'random': 'Random', 'oracle_target': 'Oracle target', 'other': 'Other sources',
                     'M0': 'Pretrained M0', 'M1_probe': 'Probe M1'}
            display(table)
        ''')),
        nbformat.v4.new_code_cell(dedent('''\
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            for name, result in summary['retrieval'].items():
                curve = pd.read_csv(folder/'data'/name/'pr_curve.csv')
                ax.step(curve.recall, curve.precision, where='post',
                        label=f"{names.get(name, name)} (AP={result['average_precision']:.4f})")
            prevalence = next(iter(summary['retrieval'].values()))['prevalence']
            ax.axhline(prevalence, linestyle=':', color='gray', label=f'Random expectation ({prevalence:.0%})')
            ax.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02),
                   title='Recover held-out target-source documents')
            ax.legend(fontsize=9)
            ax.grid(alpha=.2)
            fig.tight_layout()
            fig.savefig(figures/'precision_recall.png', dpi=180)
            fig.savefig(figures/'precision_recall.pdf')
            plt.show()
            plt.close(fig)
        ''')),
        nbformat.v4.new_code_cell(dedent('''\
            paired = pd.read_csv(folder/'data'/'paired_losses.csv').merge(
                pd.read_csv(folder/'data'/'candidate_ground_truth.csv')[['id', 'is_target']],
                on='id', validate='one_to_one')
            paired['absolute_drop'] = paired.loss_before - paired.loss_after
            paired['relative_drop'] = paired.absolute_drop / paired.loss_before.clip(lower=1e-8)
            display(pd.DataFrame(summary['paired_loss_changes']))
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for ax, metric, label in zip(axes, ['absolute_drop', 'relative_drop'],
                                         ['Absolute NLL drop (nats/token)', 'Relative NLL drop']):
                for is_target, group in paired.groupby('is_target'):
                    values = np.sort(group[metric].to_numpy())
                    ax.step(values, np.arange(1, len(values)+1)/len(values), where='post',
                            label='Target source' if is_target else 'Other sources')
                ax.axvline(0, color='gray', linestyle=':')
                ax.set(xlabel=label, ylabel='Empirical CDF')
                ax.legend()
                ax.grid(alpha=.2)
            fig.suptitle('Probe M1 loss changes; positive drop means the document loss decreased')
            fig.tight_layout()
            fig.savefig(figures/'candidate_loss_changes.png', dpi=180)
            plt.show()
            plt.close(fig)
        ''')),
        nbformat.v4.new_code_cell(dedent('''\
            arms = summary['config']['continuation']['arms']
            base = table[table.arm == 'M0'].iloc[0]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            for ax, metric, title in zip(axes, ['target_nll', 'other_nll'],
                                        ['Independent target test', 'Independent other-source test']):
                for position, arm in enumerate(arms):
                    values = table[table.arm == arm][metric].to_numpy()
                    ax.scatter(np.repeat(position, len(values)), values, s=35, zorder=3)
                    ax.errorbar(position, values.mean(),
                                yerr=values.std(ddof=1) if len(values) > 1 else 0,
                                color='black', fmt='_', markersize=18, capsize=5)
                ax.axhline(base[metric], color='gray', linestyle='--', label='Pretrained M0')
                ax.set_xticks(range(len(arms)), [names.get(a, a) for a in arms], rotation=25, ha='right')
                ax.set(title=title, ylabel='Token-weighted NLL (nats/token; lower is better)')
                ax.legend()
                ax.grid(axis='y', alpha=.2)
            fig.suptitle('Same M0 and prediction-token budget; points are paired seeds, whiskers are seed SD')
            fig.tight_layout()
            fig.savefig(figures/'continuation_nll.png', dpi=180)
            plt.show()
            plt.close(fig)
        ''')),
        nbformat.v4.new_code_cell(dedent('''\
            comparisons = summary['oracle_vs_other']
            values = np.array([r['other_minus_oracle_target']['difference'] for r in comparisons])
            bounds = np.array([r['other_minus_oracle_target']['ci95'] for r in comparisons])
            fig, ax = plt.subplots(figsize=(6.5, 4))
            positions = np.arange(len(values))
            ax.vlines(positions, bounds[:, 0], bounds[:, 1], color='tab:blue')
            ax.scatter(positions, values, color='tab:blue')
            ax.axhline(0, color='gray', linestyle='--')
            ax.set_xticks(positions, [str(r['seed']) for r in comparisons])
            ax.set(xlabel='Paired continuation seed', ylabel='Other NLL - oracle target NLL',
                   title='Target-test benefit of domain data (positive is better)')
            ax.text(.02, .02, '95% paired document bootstrap; conditional on each trained model pair',
                    transform=ax.transAxes, fontsize=8)
            fig.tight_layout()
            fig.savefig(figures/'oracle_vs_other.png', dpi=180)
            plt.show()
            plt.close(fig)
        ''')),
        nbformat.v4.new_code_cell(dedent('''\
            first_seed = summary['config']['continuation']['seeds'][0]
            fig, ax = plt.subplots(figsize=(7, 4))
            for arm in summary['config']['continuation']['arms']:
                history = pd.read_json(folder/'data'/'training'/f'{arm}-seed-{first_seed}.jsonl', lines=True)
                assert int(history.iloc[-1].seen_tokens) == summary['config']['continuation']['prediction_tokens']
                ax.plot(history.seen_tokens, history.loss.rolling(5, min_periods=1).mean(), label=names.get(arm, arm))
            ax.set(xlabel='Effective prediction tokens', ylabel='Training NLL (5-step moving mean)',
                   title=f'Training diagnostics, seed {first_seed}; training distributions differ')
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(figures/'training_curves.png', dpi=180)
            plt.show()
            plt.close(fig)
        ''')),
    ], metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}})
    target = folder / 'analysis.ipynb'
    nbformat.write(notebook, target)
    manager = KernelManager(kernel_name='python3')
    manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    NotebookClient(notebook, km=manager, timeout=300, resources={'metadata': {'path': str(folder)}}).execute(cleanup_kc=True)
    nbformat.write(notebook, target)


def register_report(folder):
    root = folder.parent
    index = root / 'README.md'
    with (root / '.index.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        lines = index.read_text().splitlines() if index.exists() else [
            '实验分析归档，时间使用 America/Los_Angeles。', '', '| 分析时间 | 实验 | 报告 |', '|---|---|---|']
        if any(f']({folder.name}/)' in line for line in lines):
            return
        stamp = folder.name[:10] + ' ' + folder.name[11:16].replace('-', ':')
        entry = (f'| {stamp} | [{folder.name[17:]}]({folder.name}/) | '
                 f'[报告]({folder.name}/REPORT.md) · [Notebook]({folder.name}/analysis.ipynb) |')
        position = next((i for i, line in enumerate(lines) if line.startswith('| 20') and line.split('|')[1].strip() > stamp), len(lines))
        lines.insert(position, entry)
        index.write_text('\n'.join(lines) + '\n')


def report(cfg):
    run = run_dir(cfg)
    summary = read_json(run / 'summary.json')
    if summary['config'] != cfg:
        raise RuntimeError('Report config differs from the completed experiment')
    table = pd.read_csv(run / 'metrics.csv')
    required = {(arm, seed) for arm in cfg['continuation']['arms'] for seed in cfg['continuation']['seeds']}
    actual = {(row.arm, int(row.seed)) for row in table.itertuples() if pd.notna(row.seed) and row.arm != 'M1_probe'}
    if actual != required:
        raise RuntimeError('The report needs all configured paired runs')
    paired = pd.read_csv(run / 'scores/paired_losses.csv').merge(
        pd.read_csv(run / 'data/candidate_ground_truth.csv')[['id', 'is_target']], on='id', validate='one_to_one')
    paired['drop'] = paired.loss_before - paired.loss_after
    summary['paired_loss_changes'] = [
        {'is_target': bool(label), 'documents': len(group), 'mean_before': float(group.loss_before.mean()),
         'mean_after': float(group.loss_after.mean()), 'mean_drop': float(group['drop'].mean()),
         'median_drop': float(group['drop'].median()), 'fraction_improved': float((group['drop'] > 0).mean())}
        for label, group in paired.groupby('is_target')]
    m0 = table[table.arm == 'M0'].iloc[0]
    m1 = table[table.arm == 'M1_probe'].iloc[0]
    summary['probe_evaluation'] = {'M0_target_nll': float(m0.target_nll), 'M1_target_nll': float(m1.target_nll),
                                   'M0_other_nll': float(m0.other_nll), 'M1_other_nll': float(m1.other_nll)}
    fingerprint = digest(summary)
    latest = run / 'latest_report.json'
    if latest.exists():
        previous = read_json(latest)
        if previous['fingerprint'] == fingerprint and (Path(previous['directory']) / 'REPORT.md').exists():
            return Path(previous['directory'])
    root = Path(cfg['analysis_root'])
    root.mkdir(parents=True, exist_ok=True)
    pending = run / 'pending_report.json'
    if pending.exists() and (read_json(pending).get('fingerprint') == fingerprint
                             or read_json(pending).get('config_fingerprint') == digest(cfg)):
        folder = Path(read_json(pending)['directory'])
    else:
        stamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H-%M')
        name = f'{stamp}-domain-retrieval-{cfg["data"]["target_category"].lower()}-pilot'
        folder = root / name
        suffix = 2
        while folder.exists():
            folder = root / f'{name}-{suffix}'
            suffix += 1
        folder.mkdir()
        write_json(pending, {'directory': str(folder), 'fingerprint': fingerprint, 'config_fingerprint': digest(cfg)})
    data = folder / 'data'
    data.mkdir(exist_ok=True)
    for source, destination in [(run / 'metrics.csv', folder / 'metrics.csv'),
                                (run / 'scores/paired_losses.csv', data / 'paired_losses.csv'),
                                (run / 'data/candidate_ground_truth.csv', data / 'candidate_ground_truth.csv'),
                                (run / 'data/split_manifest.csv', data / 'split_manifest.csv'),
                                (run / 'data/COMPLETED.json', data / 'data_summary.json')]:
        shutil.copyfile(source, destination)
    write_json(folder / 'metrics.json', summary)
    pd.DataFrame(summary['paired_loss_changes']).to_csv(data / 'paired_loss_summary.csv', index=False)
    (folder / 'experiment.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    for method in cfg['methods']:
        shutil.copytree(run / 'methods' / method, data / method, dirs_exist_ok=True)
    evaluation = []
    for path in sorted((run / 'evaluation').glob('*.parquet')):
        frame = pd.read_parquet(path)
        frame.insert(0, 'model', path.stem)
        evaluation.append(frame)
    pd.concat(evaluation, ignore_index=True).to_csv(data / 'per_document_nll.csv', index=False)
    (data / 'training').mkdir(exist_ok=True)
    for path in sorted((run / 'pretraining').glob('*/train.jsonl')):
        shutil.copyfile(path, data / 'training' / f'{path.parent.name}.jsonl')
        shutil.copyfile(path.parent / 'effective_config.json', data / 'training' / f'{path.parent.name}.effective_config.json')
    shutil.copytree(run / 'selection', data / 'selection', dirs_exist_ok=True)
    write_json(folder / 'runtime_versions.json', {'python': sys.version, **{
        name: version(name) for name in ['torch', 'transformers', 'tokenizers', 'numpy', 'pandas',
                                        'pyarrow', 'scikit-learn', 'matplotlib', 'nbformat', 'nbclient']}})
    for relative in ['src/picking/domain_retrieval', 'methods']:
        shutil.copytree(REPO / relative, folder / 'implementation' / relative, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (folder / 'implementation/scripts').mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPO / 'scripts/domain_retrieval.py', folder / 'implementation/scripts/domain_retrieval.py')
    preflight = cfg.get('numerical_preflight_run')
    if preflight:
        shutil.copytree(Path(preflight) / 'diagnostics', folder / 'diagnostics', dirs_exist_ok=True)
        shutil.copyfile(Path(preflight) / 'effective_config.json', folder / 'diagnostics' / 'rejected_run_config.json')
    figure_notebook(folder)
    baseline = float(table[table.arm == 'M0'].target_nll.iloc[0])
    mean_gain = summary['mean_other_minus_oracle']
    outcome = '本次所有 seed 中，领域正例组的独立目标领域 loss 都低于其他来源组。' if summary['all_seeds_oracle_better'] else '本次结果没有在所有 seed 上一致支持领域正例组优于其他来源组。'
    lines = ['领域检索与继续训练：首轮实验', '============================', '',
             f'{outcome} `other NLL − oracle target NLL` 的 seed 均值为 **{mean_gain:.6f} nats/token**。', '',
             f'起点是 `{cfg["model"].get("name", cfg["model"].get("local_path"))}` 的已有权重，'
             f'目标类别 `{cfg["data"]["target_category"]}`；预训练 M0 的目标测试 NLL 为 **{baseline:.6f}**。', '',
             '实验协议', '--------', '',
             f'- 来源：`{cfg["data"].get("dataset", "local_files")}`，固定 revision 与 shard 列表见 [experiment.yaml](experiment.yaml)。',
             f'- 目标类别拆成 {cfg["data"]["probe_documents"]} 篇 probe、{cfg["data"]["positive_documents"]} 篇检索正例、{cfg["data"]["target_test_documents"]} 篇最终独立测试文档。',
             f'- 候选池加入 {cfg["data"]["negative_documents"]} 篇其他来源负例；另留 {cfg["data"]["other_test_documents"]} 篇其他来源测试文档。所有划分先做文档与评分前缀去重。',
             f'- 每篇只用前 {cfg["data"]["max_length"]} tokens，不添加 chat template 或 EOS。排序 loss 是逐篇平均 token NLL；最终测试是 token 加权 NLL。',
             '- probe 从 M0 微调得到 M1。绝对分数是 `L0−L1`，相对分数是 `(L0−L1)/max(L0,1e-8)`，越大越靠前。',
             '- 类别标签不进入排序接口。PR 使用完整候选池，并列分数按共同阈值处理；AP 使用阶梯积分。',
             f'- 继续训练各组均重新加载 M0，各选 {cfg["continuation"]["selected_documents"]} 篇候选文档，使用 {cfg["continuation"]["prediction_tokens"]:,} 个有效预测 tokens；配对 seeds 为 {cfg["continuation"]["seeds"]}。',
             '- 每次更新也使用相同 token 预算，尾步精确截断。允许重访所选文档，保存 optimizer、scheduler、RNG 和 token 游标，默认自动恢复。', '',
             '检索效果', '--------', '', '| 方法 | AP | P@K | R@K |', '|---|---:|---:|---:|']
    k = cfg['continuation']['selected_documents']
    for method, result in summary['retrieval'].items():
        lines.append(f'| {method} | {result["average_precision"]:.6f} | {result[f"precision_at_{k}"]:.6f} | {result[f"recall_at_{k}"]:.6f} |')
    lines += ['', f'K={k}；随机期望 precision 等于候选池正例占比。', '',
              '![PR 曲线](figures/precision_recall.png)', '',
              f'probe 的独立目标测试 NLL：M0 **{m0.target_nll:.6f}** → M1 **{m1.target_nll:.6f}**；'
              f'其他来源测试 NLL：**{m0.other_nll:.6f}** → **{m1.other_nll:.6f}**。'
              '高 AP 说明两类数据的 loss 变化可以分离，不能单独证明 probe 改善了目标领域泛化。', '',
              '| 候选类别 | 平均 L0−L1 | 中位 L0−L1 | loss 实际下降的篇数占比 |', '|---|---:|---:|---:|']
    for row in summary['paired_loss_changes']:
        lines.append(f'| {"目标类别" if row["is_target"] else "其他来源"} | {row["mean_drop"]:.6f} | {row["median_drop"]:.6f} | {row["fraction_improved"]:.2%} |')
    lines += ['', '正分表示 loss 下降；负分仍参与检索排序。这里的高 AP 需要结合负例 loss 的变化一起解释。', '',
              '![候选 loss 变化](figures/candidate_loss_changes.png)', '', '继续训练效果', '------------', '',
              '| 训练数据 | 目标测试 NLL 均值 | seed 标准差 | 较 M0 改善 | 其他来源测试 NLL |', '|---|---:|---:|---:|---:|']
    for arm in cfg['continuation']['arms']:
        values = table[table.arm == arm]
        spread = values.target_nll.std(ddof=1) if len(values) > 1 else 0.
        lines.append(f'| {arm} | {values.target_nll.mean():.6f} | {spread:.6f} | {baseline-values.target_nll.mean():.6f} | {values.other_nll.mean():.6f} |')
    lines += ['', '`oracle_target` 是候选池内已知目标类别正例，`other` 是候选池内其他来源；所有组都在第三份独立测试集评估。', '',
              '![继续训练 NLL](figures/continuation_nll.png)', '', '| seed | other − oracle target | 95% 文档 bootstrap 区间 |', '|---|---:|---|']
    for item in summary['oracle_vs_other']:
        value = item['other_minus_oracle_target']
        lines.append(f'| {item["seed"]} | {value["difference"]:.6f} | [{value["ci95"][0]:.6f}, {value["ci95"][1]:.6f}] |')
    lines += ['', '正差值表示领域正例组更好。区间按测试文档配对重采样，条件于给定的一对训练模型；不是跨领域或跨 probe 的置信区间。', '',
              '![配对差值](figures/oracle_vs_other.png)', '', '![训练诊断](figures/training_curves.png)', '',
              '结论边界', '--------', '',
              '- 来源类别是检索 ground truth，不等于每篇文档对目标 loss 的真实因果贡献；其他来源也可能含目标主题。',
              '- 独立测试集仅保证未参与本次 probe、排序或继续训练；不能保证原始预训练模型从未见过相关语料。',
              '- 首轮使用一个类别、一个 probe 模型和固定数据划分。三个继续训练 seed 不能代替多领域、多划分验证。',
              '- 只评估文档前缀；上游去重之外，本次近重复检查是 MinHash 候选检索加 Jaccard 核对，可能漏掉部分近重复。',
              '- 训练 loss 来自不同分布，仅作诊断；是否有效以独立测试 NLL 为准。', '',
              '复现与产物', '----------', '',
              '[Notebook](analysis.ipynb) 包含绘图代码及内嵌执行结果；[metrics.csv](metrics.csv)、[metrics.json](metrics.json) 和 [data/](data/) 保存原始统计、逐篇 loss、PR 阈值、选数清单与训练日志。', '',
              '[implementation/](implementation/) 冻结本实验的代码、方法目录和入口（依赖仓库已有的公共存储模块）；'
              '[runtime_versions.json](runtime_versions.json) 记录依赖版本，`data/training/` 同时保存各组的有效配置和训练日志。', '',
              '复现图表：', '', '```bash', f'cd {folder}', f"{REPO}/.venv/bin/python - <<'PY'", 'import nbformat',
              'from nbclient import NotebookClient', "nb = nbformat.read('analysis.ipynb', as_version=4)",
              "NotebookClient(nb, timeout=300, kernel_name='python3').execute()", "nbformat.write(nb, 'analysis.ipynb')", 'PY', '```', '',
              '重跑或自动续跑实验（选择一张当前空闲的允许 GPU）：', '', '```bash', f'cd {REPO}',
              f'DATA_PICKING_GPU_IDS=0 .venv/bin/python scripts/domain_retrieval.py run --config {folder}/experiment.yaml', '```', '',
              f'持久化运行目录：`{run}`。']
    if preflight:
        lines += ['', '数值预检', '--------', '',
                  '在未更新的 M0、同一批 probe 文本上，FP32 NLL 为 2.956960，BF16 autocast 为 3.451031；'
                  '对应梯度范数约 20.84 与 216.49。基于这个 probe 数值检查，正式实验统一采用 FP32，学习率和 token 预算未改变。',
                  '此前 BF16 运行已中止并保留，不计入上面的统计。原始检查数值、复现 helper 及弃用配置见 [diagnostics/](diagnostics/)。']
    (folder / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    write_json(folder / 'manifest.json', {'source_run': str(run), 'fingerprint': fingerprint,
                                        'model_identity': read_json(run / 'M0_identity.json')})
    register_report(folder)
    write_json(latest, {'directory': str(folder), 'fingerprint': fingerprint})
    return folder
