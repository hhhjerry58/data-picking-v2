"""Write paired mathematical utility results with immutable review snapshots."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .common import config, paths, read_json, write_json, file_hash, digest, utc_now, REPO, emit


def paired_interval(values, group_ids, seed, samples=2000):
    values = np.asarray(values, dtype=float)
    groups, inverse = np.unique(np.asarray(group_ids), return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.stack([np.bincount(inverse, weights=row) for row in values])
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        seed_sample = rng.integers(0, len(values), size=len(values))
        group_sample = rng.integers(0, len(groups), size=len(groups))
        estimates.append(float(sums[seed_sample][:, group_sample].sum() / (len(seed_sample)*counts[group_sample].sum())))
    return {'mean': float(values.mean()), 'ci95': np.quantile(estimates, [.025,.975]).tolist(),
            'bootstrap': 'paired seed and problem-group resampling' if len(values)>1 else 'paired problem-group bootstrap conditional on one seed'}


def report(cfg=None):
    cfg = cfg or config()
    p = paths(cfg)
    arms = cfg['pretraining']['arms']
    seeds = [seed for seed in cfg['pretraining']['seeds'] if all((p['run']/'evaluation'/f'{arm}-seed-{seed}'/'COMPLETED.json').exists() for arm in arms)]
    if not seeds:
        raise RuntimeError('No fully evaluated paired seed is available')
    source_files = {}
    results, math_frames, training_curves = [], {}, {}
    reference_training = None
    for seed in seeds:
        initial_hashes = set()
        for arm in arms:
            folder = p['run']/'evaluation'/f'{arm}-seed-{seed}'
            metrics_path = folder/'metrics.json'
            if file_hash(metrics_path)!=read_json(folder/'COMPLETED.json')['metrics_sha256']:
                raise RuntimeError('Evaluation metrics changed after completion')
            metrics = read_json(metrics_path)
            trained = p['run']/'pretraining'/f'{arm}-seed-{seed}'
            effective = read_json(trained/'effective_config.json')
            initial_hashes.add(effective['initial_parameters_sha256'])
            comparison = {key: effective['identity'][key] for key in ['training','architecture_sha256','implementation_sha256','attention_backend','compile_model','torch_version']}
            if reference_training is None:
                reference_training = comparison
            elif comparison != reference_training:
                raise RuntimeError('The experiment arms use different training recipes/implementations')
            if metrics['training_tokens'] != 3200000000:
                raise RuntimeError('Unmatched supervised token budget')
            frame = pd.read_parquet(folder/'math_test_nll.parquet').sort_values('id').reset_index(drop=True)
            math_frames[(arm,seed)] = frame
            final_checkpoint = Path(read_json(trained/'TRAINING_COMPLETED.json')['checkpoint'])
            checkpoint_manifest = read_json(final_checkpoint/'manifest.json')
            history_path = final_checkpoint/'train.jsonl'
            if file_hash(history_path) != checkpoint_manifest['files']['train.jsonl']['sha256']:
                raise RuntimeError('Final checkpoint training history changed')
            history = pd.read_json(history_path,lines=True).drop_duplicates('step',keep='last').sort_values('step')
            if int(history.iloc[-1]['seen_tokens']) != 3200000000:
                raise RuntimeError('Training history does not end at the fixed budget')
            training_curves[(arm,seed)] = history
            results.append({'arm':arm,'seed':seed,'prediction_tokens':metrics['training_tokens'],
                'math_nll':metrics['math_test']['mean_document_nll'],
                'general_nll':metrics['general']['mean_document_nll'],
                'normalized_answer_accuracy':metrics['generation']['accuracy'],
                'generation_questions':metrics['generation']['questions'],
                'answer_extraction_rate':metrics['generation']['answer_extraction_rate'],
                'measured_update_hours':float(history['seconds'].sum()/3600)})
            for path in [metrics_path,folder/'math_test_nll.parquet',trained/'effective_config.json',trained/'TRAINING_COMPLETED.json']:
                source_files[str(path)] = file_hash(path)
        if len(initial_hashes)!=1:
            raise RuntimeError('Paired initializations differ between arms')
    contrasts = {}
    reference_ids = math_frames[('random',seeds[0])]['id']
    reference_groups = math_frames[('random',seeds[0])]['group_id'].tolist()
    for arm in ['delta_loss','relative_gain']:
        absolute,relative = [],[]
        for seed in seeds:
            base,method = math_frames[('random',seed)],math_frames[(arm,seed)]
            if not base['id'].equals(reference_ids) or not method['id'].equals(reference_ids):
                raise RuntimeError('Different mathematical questions were evaluated')
            if not base['scored_tokens'].equals(method['scored_tokens']):
                raise RuntimeError('Different answer tokens were evaluated')
            delta = base['nll'].to_numpy()-method['nll'].to_numpy()
            ratio = delta/np.maximum(base['nll'].to_numpy(),1e-6)
            absolute.append(delta);relative.append(ratio)
            for row in results:
                if row['arm']==arm and row['seed']==seed:
                    row['absolute_math_improvement']=float(delta.mean())
                    row['relative_math_improvement']=float(ratio.mean())
        contrasts[arm]={'absolute':paired_interval(absolute,reference_groups,cfg['evaluation']['bootstrap_seed'],cfg['evaluation']['bootstrap_samples']),
                        'relative':paired_interval(relative,reference_groups,cfg['evaluation']['bootstrap_seed'],cfg['evaluation']['bootstrap_samples'])}
    for row in results:
        if row['arm']=='random':
            row['absolute_math_improvement']=0.0;row['relative_math_improvement']=0.0
    fingerprint = digest({'files':source_files,'implementation':file_hash(__file__)})
    latest = p['run']/'latest_report.json'
    if latest.exists() and read_json(latest)['fingerprint']==fingerprint:
        return
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%SZ')
    folder = REPO/'experiments-analysis'/f'{stamp}_{cfg["run_id"]}_seeds-{"-".join(map(str,seeds))}'
    folder.mkdir(parents=True,exist_ok=True)
    table = pd.DataFrame(results)
    table.to_csv(folder/'metrics.csv',index=False)
    corpus = read_json(p['data']/'corpus_summary.json')
    ft = {metric:read_json(p['run']/'fasttext'/metric/'heldout_metrics.json') for metric in cfg['fasttext']['metrics']}
    summary = {'run_id':cfg['run_id'],'seeds':seeds,'results':results,'paired_contrasts':contrasts,
               'formal_label_validity':read_json(p['data']/'label_validity.json'),
               'candidate_documents':corpus['document_counts']['candidate'],
               'candidate_pythia_tokens':corpus['pythia_tokens_including_eos']['candidate'],
               'math_test_preparation':read_json(p['data']/'targets/summary.json'),
               'fasttext_heldout_metrics':ft,'training_recipe':reference_training,'completed_at_utc':utc_now()}
    write_json(folder/'metrics.json',summary)
    write_json(folder/'manifest.json',{'fingerprint':fingerprint,'source_run':str(p['run']),'source_files':source_files})
    shutil.copyfile(REPO/'configs/experiment.yaml',folder/'experiment.yaml')
    names={'random':'Random','delta_loss':'FT-delta','relative_gain':'FT-relative'}
    colors={'random':'#555555','delta_loss':'#2b6cb0','relative_gain':'#c05621'}
    fig,ax=plt.subplots(figsize=(6,4))
    for i,arm in enumerate(arms):
        values=table[table.arm==arm].math_nll.to_numpy()
        ax.scatter(np.repeat(i,len(values)),values,color=colors[arm],s=45)
        ax.errorbar(i,values.mean(),yerr=values.std(ddof=1) if len(values)>1 else 0,color=colors[arm],capsize=6,fmt='_',markersize=22)
    ax.set_xticks(range(len(arms)),[names[a] for a in arms]);ax.set_ylabel('Held-out MATH answer NLL')
    ax.set_title('160M / 3.2B prediction tokens (lower is better)');fig.tight_layout();fig.savefig(folder/'math_nll.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for (arm,seed),history in training_curves.items():
        ax.plot(history.seen_tokens/1e9,history.loss.rolling(100,min_periods=1).mean(),color=colors[arm],alpha=.85 if seed==seeds[0] else .4,label=f'{names[arm]} / {seed}')
    ax.set_xlabel('Prediction tokens (B)');ax.set_ylabel('Training NLL (100-step moving mean)');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(folder/'training_curves.png',dpi=180);plt.close(fig)
    lines=[f'数学 loss-drop 选数实验：已完成 seeds {seeds}', '='*44, '',
           f'每组从零预训练 160M，实际消耗 3,200,000,000 个有效预测 tokens。最终数学评估 {len(reference_ids):,} 道题。', '',
           '| 实验组 | seed | 数学 NLL | 相对 Random 绝对改善 | 相对改善 | 规范化答案准确率 | 通用 NLL |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in results:
        lines.append(f"| {names[row['arm']]} | {row['seed']} | {row['math_nll']:.6f} | {row['absolute_math_improvement']:+.6f} | {100*row['relative_math_improvement']:+.3f}% | {100*row['normalized_answer_accuracy']:.3f}% | {row['general_nll']:.6f} |")
    lines += ['', '正的改善值表示选数方法优于同 seed 的 Random。相对改善先逐题计算，再对题目和 seeds 平均。', '',
              '| 方法 | 绝对改善均值及 95% CI | 相对改善均值及 95% CI |', '| --- | --- | --- |']
    for arm,contrast in contrasts.items():
        a,r=contrast['absolute'],contrast['relative']
        lines.append(f"| {names[arm]} | {a['mean']:+.6f} [{a['ci95'][0]:+.6f}, {a['ci95'][1]:+.6f}] | {100*r['mean']:+.3f}% [{100*r['ci95'][0]:+.3f}%, {100*r['ci95'][1]:+.3f}%] |")
    lines += ['', '只有一个 seed 时，区间仅描述给定模型下的题目不确定性；多个 seeds 时同时重采样 seed 和题组。',
              '生成评估使用固定的答案提取、LaTeX 字符串规范化与精确有理数等价；不包含通用符号推理判题，不能直接与其他 MATH leaderboard 准确率混比。',
              '三个训练集的分布不同，训练 loss 曲线用于诊断，主要效果比较使用独立留出数学 NLL。', '',
              '![数学 NLL](math_nll.png)', '', '![训练曲线](training_curves.png)', '',
              '逐运行结果见 [metrics.csv](metrics.csv)，完整数值、分类器留出指标及数据统计见 [metrics.json](metrics.json)。',
              f'持久化运行产物：`{p["run"]}`。']
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(latest,{'fingerprint':fingerprint,'directory':str(folder),'seeds':seeds,'time':utc_now()})
    index=REPO/'experiments-analysis/README.md'
    if not index.exists():index.write_text('实验结果索引\n============\n\n')
    with index.open('a') as stream:stream.write(f'- [{stamp} · seeds {seeds}]({folder.name}/REPORT.md)\n')
    emit('review_report_written',directory=str(folder),seeds=seeds)
