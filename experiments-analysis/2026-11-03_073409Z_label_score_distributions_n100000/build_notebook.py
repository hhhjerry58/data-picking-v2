"""Build and execute a self-contained, CPU-only label-distribution notebook."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
import importlib.metadata as metadata

HERE = Path(__file__).resolve().parent
for key, relative in {
    'MPLCONFIGDIR': '.cache/matplotlib',
    'JUPYTER_RUNTIME_DIR': '.cache/jupyter',
    'IPYTHONDIR': '.cache/ipython',
    'XDG_CACHE_HOME': '.cache',
    'TMPDIR': '.cache/tmp',
}.items():
    target = HERE / relative
    target.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(target)
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[key] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec
pa.set_cpu_count(2)
pa.set_io_thread_count(2)

SOURCE_RUN = Path('/mnt/tianzhu_hot/data-picking-v2/runs/math-lossdrop-fasttext-qwen15b-pythia160m-n100000-seed20261102')
SOURCE = SOURCE_RUN / 'label_scores.parquet'
def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            hasher.update(block)
    return hasher.hexdigest()

source_record = json.loads(Path(str(SOURCE) + '.complete.json').read_text())
source_sha = sha256(SOURCE)
assert source_sha == source_record['sha256']
assert SOURCE.stat().st_size == source_record['size']
columns = ['id', 'partition', 'ref_loss', 'sft_loss', 'delta_loss', 'relative_gain', 'pythia_tokens']
frame = pq.read_table(SOURCE, columns=columns).to_pandas()
assert len(frame) == source_record['documents'] == 100000
assert frame['id'].is_unique
assert np.isfinite(frame[['ref_loss', 'sft_loss', 'delta_loss', 'relative_gain']].to_numpy()).all()
np.testing.assert_allclose(frame.delta_loss, frame.ref_loss-frame.sft_loss, rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(frame.relative_gain, frame.delta_loss/np.maximum(frame.ref_loss, 1e-6), rtol=1e-12, atol=1e-12)
snapshot = HERE / 'snapshot/label_scores.parquet'
frame.to_parquet(snapshot, index=False, compression='zstd')
thresholds = {}
threshold_sources = {}
for metric in ['delta_loss', 'relative_gain']:
    path = SOURCE_RUN / 'fasttext' / metric / 'threshold.json'
    thresholds[metric] = json.loads(path.read_text())
    threshold_sources[metric] = {'path': str(path), 'sha256': sha256(path)}
(HERE/'snapshot/thresholds.json').write_text(json.dumps(thresholds, ensure_ascii=False, indent=2)+'\n')
manifest = {
    'analysis_kind': 'label_score_distributions',
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'source_run': str(SOURCE_RUN),
    'source_scores': str(SOURCE),
    'source_scores_sha256': source_sha,
    'source_completion': source_record,
    'snapshot': str(snapshot.relative_to(HERE)),
    'snapshot_sha256': sha256(snapshot),
    'snapshot_columns': columns,
    'documents': len(frame),
    'partition_counts': {k: int(v) for k,v in frame.partition.value_counts().items()},
    'threshold_sources': threshold_sources,
    'relative_display_units': 'percent = 100 * relative_gain; snapshot retains raw ratio',
    'central_interval': [0.005, 0.995],
    'figure_weighting': 'document weighting unless explicitly labelled Pythia-token weighting',
    'versions': {name: metadata.version(name) for name in ['numpy','pandas','pyarrow','matplotlib','nbformat','nbclient','ipykernel']},
}
(HERE/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')

nb = nbformat.v4.new_notebook()
nb.metadata.update({'kernelspec': {'name':'python3','display_name':'Python 3','language':'python'},
                    'language_info': {'name':'python','version':sys.version.split()[0]}})
cells = []
def md(text): cells.append(nbformat.v4.new_markdown_cell(text))
def code(text): cells.append(nbformat.v4.new_code_cell(text))

md('''# 标注集的 delta 与 relative delta 分布

数据为已完成评分的 **100,000 篇标注文档**，包含 label_train / label_dev / label_test。

对每篇文档，先按实际预测 token 数聚合平均 NLL，再计算：

$$\\Delta L=L_{\\mathrm{ref}}-L_{\\mathrm{sft}},\\qquad r=\\frac{\\Delta L}{\\max(L_{\\mathrm{ref}},10^{-6})}.$$

**正值表示数学 SFT 后该文档的 loss 降低；负值表示升高。** `relative delta` 对应项目里的 `relative_gain`；图表用 $100r$（百分比）显示，快照保留原始比例。

本 notebook 使用独立的本地快照，全部计算在 CPU 上进行。数据来源与 SHA-256 见 `manifest.json`。图像输出到 `figures/`。''')

code('''from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from IPython.display import display, Markdown

get_ipython().run_line_magic('matplotlib', 'inline')
BASE = Path.cwd()
manifest = json.loads((BASE/'manifest.json').read_text())
snapshot = BASE/manifest['snapshot']
assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == manifest['snapshot_sha256']
df = pd.read_parquet(snapshot)
thresholds = json.loads((BASE/'snapshot/thresholds.json').read_text())
assert len(df) == 100000 and df.id.is_unique
assert np.isfinite(df[['delta_loss', 'relative_gain']].to_numpy()).all()
np.testing.assert_allclose(df.delta_loss, df.ref_loss-df.sft_loss, rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(df.relative_gain, df.delta_loss/np.maximum(df.ref_loss,1e-6), rtol=1e-12, atol=1e-12)
df['relative_pct'] = 100*df.relative_gain

font_file = Path('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
if font_file.exists():
    font_manager.fontManager.addfont(str(font_file))
    cjk_name = font_manager.FontProperties(fname=str(font_file)).get_name()
    plt.rcParams['font.family'] = ['DejaVu Sans', cjk_name]
plt.rcParams.update({'font.size':11, 'axes.titlesize':13, 'axes.labelsize':11,
                     'axes.spines.top':False, 'axes.spines.right':False,
                     'axes.unicode_minus':True, 'figure.dpi':110, 'savefig.dpi':180})
FIGURES = BASE/'figures'
FIGURES.mkdir(exist_ok=True)
specs = [
    {'key':'delta_loss', 'column':'delta_loss', 'title':'绝对下降量 delta',
     'xlabel':'Δ loss (nats / token)', 'color':'#3574BA', 'scale':1.0},
    {'key':'relative_gain', 'column':'relative_pct', 'title':'相对下降量 relative delta',
     'xlabel':'Relative delta (%)', 'color':'#BD7332', 'scale':100.0},
]
limits = {s['key']: np.quantile(df[s['column']].to_numpy(), [0.005,0.995]) for s in specs}

def reference_lines(ax, spec, median=False):
    ax.axvline(0, color='#237651', lw=1.4, label='0：loss 不变')
    cutoff = thresholds[spec['key']]['threshold']*spec['scale']
    ax.axvline(cutoff, color='#B74759', ls='--', lw=1.5, label='fastText 标签阈值（训练部分）')
    if median:
        ax.axvline(df[spec['column']].median(), color='#545E68', ls=':', lw=1.3, label='全量中位数')
    ax.set_xlabel(spec['xlabel'])
    if spec['scale'] == 100:
        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=1))

def finish(fig, name):
    fig.savefig(FIGURES/(name+'.png'), bbox_inches='tight')
    plt.show()
    plt.close(fig)

display(pd.Series(df.partition.value_counts(), name='文档数').to_frame())
print('快照 SHA-256:', manifest['snapshot_sha256'])
print('数据源完成时间:', manifest['source_completion']['completed_at_utc'])''')

md('''**1. 总体统计与分位数**

均值、标准差、分位数均按文档等权计算。相对下降量先逐篇计算后再取均值。下方额外列出按 Pythia token 数加权的正值占比，与分类器标注时的 token 预算口径对应。''')

code('''summaries = []
for spec in specs:
    x = df[spec['column']].to_numpy()
    weight = df.pythia_tokens.to_numpy(dtype=np.float64)
    summaries.append({
        '指标': spec['title'],
        '单位': 'nats/token' if spec['scale']==1 else '%',
        '文档数':len(x), '均值':x.mean(), '中位数':np.median(x), '标准差':x.std(ddof=1),
        '最小值':x.min(), '最大值':x.max(),
        '正值篇数':int((x>0).sum()), '零值篇数':int((x==0).sum()), '负值篇数':int((x<0).sum()),
        '正值文档占比(%)':100*np.mean(x>0),
        '正值token占比(%)':100*weight[x>0].sum()/weight.sum(),
        'fastText标签阈值':thresholds[spec['key']]['threshold']*spec['scale'],
    })
summary = pd.DataFrame(summaries).set_index('指标')
summary.to_csv(BASE/'summary.csv')
display(summary.T)

quantiles = [0,0.001,0.005,0.01,0.05,0.25,0.5,0.75,0.95,0.99,0.995,0.999,1]
qtable = pd.DataFrame({
    '分位点':[f'{100*q:g}%' for q in quantiles],
    'delta (nats/token)':df.delta_loss.quantile(quantiles).to_numpy(),
    'relative delta (%)':df.relative_pct.quantile(quantiles).to_numpy(),
}).set_index('分位点')
qtable.to_csv(BASE/'quantiles.csv')
display(qtable.style.format('{:.6f}'))
print('分母保护触发数:', int((df.ref_loss<1e-6).sum()))
assert np.array_equal(np.sign(df.delta_loss), np.sign(df.relative_gain))''')

md('''**2. 主体分布：中央 99% 区间**

横轴为各指标的 0.5%–99.5% 分位区间。纵轴是每个 bin 占**全量 100,000 篇**的百分比，未将可见部分重新归一化；图内注明窗口外的文档数。完整尾部见下一组图。

红虚线来自已经冻结的 fastText 训练标签阈值。阈值按 `label_train` 的 token 预算确定；这里展示的是三个划分的全量分布。''')

code('''fig, axes = plt.subplots(1,2, figsize=(13,4.7), layout='constrained')
for ax, spec in zip(axes, specs):
    x = df[spec['column']].to_numpy()
    low, high = limits[spec['key']]
    ax.hist(x, bins=np.linspace(low,high,121), weights=np.full(len(x),100/len(x)),
            color=spec['color'], alpha=0.85, edgecolor='white', linewidth=0.25)
    ax.set_xlim(low,high)
    reference_lines(ax,spec,median=True)
    ax.set_title(spec['title'])
    ax.set_ylabel('文档占比（每个 bin）')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax.grid(axis='y',alpha=.18)
    ax.text(.02,.97,
            f'均值：{x.mean():.6f}'+ ('%' if spec['scale']==100 else '')+
            f'\\n中位数：{np.median(x):.6f}'+ ('%' if spec['scale']==100 else '')+
            f'\\n正值文档：{100*np.mean(x>0):.3f}%'+
            f'\\n窗口外：{int((x<low).sum()):,} 左 / {int((x>high).sum()):,} 右',
            transform=ax.transAxes, va='top', fontsize=9,
            bbox={'facecolor':'white','edgecolor':'#E2E6EB','alpha':.9,'boxstyle':'round,pad=.5'})
    ax.legend(loc='upper right',fontsize=8)
fig.suptitle('10 万篇标注数据：主体分布（中央 99%）',fontsize=15)
finish(fig,'01_central_distributions')''')

md('''**3. 全范围分布：包括所有极端值**

横轴覆盖最小值到最大值，纵轴使用对数文档数。这里不删除、不截断或 winsorize 分数；空 bin 不显示正柱高。''')

code('''fig, axes = plt.subplots(1,2, figsize=(13,4.5), layout='constrained')
for ax, spec in zip(axes,specs):
    x = df[spec['column']].to_numpy()
    ax.hist(x,bins=180,color=spec['color'],alpha=.85,log=True,edgecolor='white',linewidth=.2)
    reference_lines(ax,spec)
    ax.set_title(spec['title']+'：完整范围')
    ax.set_ylabel('文档数（对数刻度）')
    ax.set_ylim(bottom=.8)
    ax.grid(axis='y',alpha=.18)
    ax.text(.03,.95,f'min = {x.min():.6f}\\nmax = {x.max():.6f}',
            transform=ax.transAxes,va='top',fontsize=10,
            bbox={'facecolor':'white','edgecolor':'#E2E6EB','alpha':.9})
    ax.legend(loc='upper right',fontsize=8)
finish(fig,'02_full_range_log_histograms')''')

md('''**4. 累积分布：文档权重与 token 权重**

实线表示低于或等于横轴分数的文档占比；虚线表示同一条件下的 Pythia token 占比。CDF 使用全量数据计算，仅将横轴放大到中央 99% 区间。''')

code('''fig, axes = plt.subplots(1,2, figsize=(13,4.5), layout='constrained')
for ax,spec in zip(axes,specs):
    x = df[spec['column']].to_numpy()
    order = np.argsort(x,kind='stable')
    weight = df.pythia_tokens.to_numpy(dtype=np.float64)[order]
    ax.plot(x[order],100*np.arange(1,len(x)+1)/len(x),
            color=spec['color'],lw=2,label='等文档权重',drawstyle='steps-post')
    ax.plot(x[order],100*np.cumsum(weight)/weight.sum(),
            color='#556575',lw=1.6,ls='--',label='按 Pythia tokens 加权',drawstyle='steps-post')
    reference_lines(ax,spec)
    ax.set_xlim(*limits[spec['key']]); ax.set_ylim(0,100)
    ax.set_title(spec['title'])
    ax.set_ylabel('累计占比')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    ax.grid(alpha=.18)
    ax.legend(loc='lower right',fontsize=8)
finish(fig,'03_document_token_ecdf')''')

md('''**5. train / dev / test 的分布对照**

三个划分使用相同的横轴区间和 bin 边界，各自按该划分的文档数归一化；分母包含各自窗口外的尾部样本。此图用于查看按域名划分后，分布是否发生变化。''')

code('''split_colors={'label_train':'#3574BA','label_dev':'#2E8B77','label_test':'#CA7E38'}
split_rows=[]
fig, axes = plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
for ax,spec in zip(axes,specs):
    edges=np.linspace(*limits[spec['key']],101)
    for split in ['label_train','label_dev','label_test']:
        group=df[df.partition==split]
        x=group[spec['column']].to_numpy()
        ax.hist(x,bins=edges,weights=np.full(len(x),100/len(x)),histtype='step',lw=1.7,
                color=split_colors[split],label=f'{split} (n={len(x):,})')
        split_rows.append({'split':split,'metric':spec['key'],'display_unit':'%' if spec['scale']==100 else 'nats/token',
                           'documents':len(x),'mean':float(x.mean()),'median':float(np.median(x)),
                           'positive_document_percent':float(100*np.mean(x>0))})
    ax.axvline(0,color='#237651',lw=1.1)
    ax.set_xlim(*limits[spec['key']]);ax.set_title(spec['title']);ax.set_xlabel(spec['xlabel'])
    ax.set_ylabel('划分内文档占比（每个 bin）')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    if spec['scale']==100:ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=100,decimals=1))
    ax.grid(axis='y',alpha=.18);ax.legend(fontsize=9)
finish(fig,'04_split_distributions')
split_summary=pd.DataFrame(split_rows)
split_summary.to_csv(BASE/'split_summary.csv',index=False)
display(split_summary)''')

md('''**6. 两种分数的联合分布**

散点以 hexbin 聚合，颜色表示文档数（对数刻度）。显示窗口为两种指标各自中央 99% 区间的交集；图中明确列出可见样本数。Pearson 与 Spearman 相关系数均使用全部 100,000 篇计算。''')

code('''x=df.delta_loss.to_numpy();y=df.relative_pct.to_numpy()
lo_x,hi_x=limits['delta_loss'];lo_y,hi_y=limits['relative_gain']
inside=(x>=lo_x)&(x<=hi_x)&(y>=lo_y)&(y<=hi_y)
pearson=float(df.delta_loss.corr(df.relative_gain,method='pearson'))
spearman=float(df.delta_loss.corr(df.relative_gain,method='spearman'))
fig,ax=plt.subplots(figsize=(8.5,5.7),layout='constrained')
hexes=ax.hexbin(x[inside],y[inside],gridsize=75,bins='log',mincnt=1,cmap='viridis')
ax.axvline(0,color='#67747B',lw=.9);ax.axhline(0,color='#67747B',lw=.9)
ax.axvline(thresholds['delta_loss']['threshold'],color='#B74759',ls='--',lw=1.2)
ax.axhline(100*thresholds['relative_gain']['threshold'],color='#B74759',ls='--',lw=1.2)
ax.set_xlim(lo_x,hi_x);ax.set_ylim(lo_y,hi_y)
ax.set_xlabel('Δ loss (nats / token)');ax.set_ylabel('Relative delta (%)')
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100,decimals=1))
ax.set_title(f'联合分布：展示 {int(inside.sum()):,} / {len(df):,} 篇')
fig.colorbar(hexes,ax=ax,label='每个 hexbin 的文档数（对数刻度）')
ax.text(.02,.98,f'全量 Pearson = {pearson:.4f}\\n全量 Spearman = {spearman:.4f}',
        transform=ax.transAxes,va='top',fontsize=10,
        bbox={'facecolor':'white','edgecolor':'#E2E6EB','alpha':.94})
finish(fig,'05_joint_distribution')

diagnostics={
    'documents':len(df),
    'positive_documents':int((df.delta_loss>0).sum()),
    'negative_documents':int((df.delta_loss<0).sum()),
    'zero_documents':int((df.delta_loss==0).sum()),
    'positive_document_fraction':float((df.delta_loss>0).mean()),
    'mean_delta_loss':float(df.delta_loss.mean()),
    'median_delta_loss':float(df.delta_loss.median()),
    'mean_relative_gain':float(df.relative_gain.mean()),
    'median_relative_gain':float(df.relative_gain.median()),
    'pearson':pearson,'spearman':spearman,
    'central_joint_documents':int(inside.sum()),
    'relative_display_multiplier':100,
    'source_sha256':manifest['source_scores_sha256'],
}
(BASE/'distribution_summary.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2)+'\\n')
display(Markdown(
    f'全量数据中，**{100*diagnostics["positive_document_fraction"]:.3f}%** 的文档 loss 降低，'
    f'**{diagnostics["negative_documents"]:,} 篇**升高，**{diagnostics["zero_documents"]} 篇**恰为零。'
    f'绝对下降量均值为 **{diagnostics["mean_delta_loss"]:.6f} nats/token**；'
    f'相对下降量均值为 **{100*diagnostics["mean_relative_gain"]:.4f}%**。'))''')

md('''两项 fastText 标签阈值均小于零。因此 include 集合包含一部分 loss 上升较小的文档；标签表达的是固定预算下的高分排名。该 notebook 展示 teacher 的分数分布，后续预训练收益请结合主实验报告评估。

图像、统计表和绘图所需快照均保存在此目录，可以独立复查；原始评分、分类器和训练状态未被修改。''')

nb.cells=cells
notebook_path=HERE/'label_score_distributions.ipynb'
nbformat.write(nb,notebook_path)
km=KernelManager(kernel_name='python3',connection_file=str(HERE/'.cache/jupyter/analysis-kernel.json'))
km._kernel_spec=KernelSpec(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                           display_name='Python 3 (local analysis)',language='python')
client=NotebookClient(nb,km=km,timeout=180,allow_errors=False,resources={'metadata':{'path':str(HERE)}})
try:
    client.execute(cwd=str(HERE),env=dict(os.environ))
finally:
    if km.has_kernel:
        km.shutdown_kernel(now=True)
    km.cleanup_resources()
nbformat.write(nb,notebook_path)
assert sha256(SOURCE)==source_sha
manifest['notebook_executed_at_utc']=datetime.now(timezone.utc).isoformat()
manifest['code_cells_executed']=sum(cell.cell_type=='code' for cell in nb.cells)
manifest['artifacts']={str(path.relative_to(HERE)):{'bytes':path.stat().st_size,'sha256':sha256(path)}
                       for path in [notebook_path,*sorted((HERE/'figures').glob('*.png')),
                                    HERE/'summary.csv',HERE/'quantiles.csv',HERE/'split_summary.csv',HERE/'distribution_summary.json']}
(HERE/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
overview=json.loads((HERE/'distribution_summary.json').read_text())
readme='''标注集分数分布\n================\n\n打开 [label_score_distributions.ipynb](label_score_distributions.ipynb) 查看已执行的图表与统计。\n\n包含：中央 99% 直方图、全范围对数直方图、文档/token 加权 CDF、train/dev/test 对照和联合分布。\n\n- 数据：完整的 100,000 篇标注集，来源与校验和见 [manifest.json](manifest.json)。\n- relative delta 在图表中显示为百分比，快照保留原始比例。\n- 主体视图明确标注显示区间，全范围图保留所有尾部数据。\n- 主要图：[主体分布](figures/01_central_distributions.png) · [全范围](figures/02_full_range_log_histograms.png)。\n- 表格：[总体统计](summary.csv) · [分位数](quantiles.csv) · [划分统计](split_summary.csv)。\n\n本目录为独立分析快照，未修改主实验的评分、配置、状态或其他分析目录。\n'''
(HERE/'README.md').write_text(readme)
print(json.dumps({'directory':str(HERE),'notebook':str(notebook_path),'executed_code_cells':manifest['code_cells_executed'],
                  'figures':len(list((HERE/'figures').glob('*.png'))),'summary':overview},ensure_ascii=False,indent=2))

