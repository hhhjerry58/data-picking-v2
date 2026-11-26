# 领域检索与继续训练实验

这条实验线检验两个不同的问题：前后模型的 loss 差能否找回同一来源类别的数据，
以及该类别数据是否比其他数据更能降低独立目标领域测试集上的 loss。
所有模型都加载已有预训练权重，使用原始文本的自回归 loss 训练。

## 数据与模型关系

将带有真实来源类别字段的预训练语料先去重，再划分：

| 部分 | 首轮篇数 | 用途 |
|---|---:|---|
| 目标类别 probe | 512 | 从 M0 得到 M1，仅用于建立 loss 差 |
| 目标类别 retrieval heldout | 1,024 | 候选池正例；也可作为 oracle 继续训练数据 |
| 其他类别候选数据 | 4,096 | 候选池负例；提供 other 对照 |
| 目标类别 final test | 512 | 所有模型共用的独立目标领域测试集 |
| 其他类别 final test | 512 | 检查对其他来源 loss 的影响 |

检索 heldout 是相对 probe 而言的留出集。它后续可以用于继续训练，因此不能同时承担最终测试。
最终测试数据不参与 probe、候选打分、选数或训练，也不用于调学习率和训练步数。
这里只能确认这些划分在本次实验内独立，不能确认原始预训练模型从未见过相关语料。

首轮使用公开 `EleutherAI/pythia-160m` 已训练 checkpoint，固定 revision
`50f5173d932e8e61f858120bcb800b97af589f46`。语料为 `DKYoon/SlimPajama-6B`，固定 revision
`b5f90f419b7489cdba26fdbc8c022fcb5562f968`，类别取 `meta.redpajama_set_name`，目标为 `RedPajamaArXiv`。
扫描配置中明确列出的八个训练 shard，按固定 ID hash 做有界抽样。
ground truth 表示来源类别；它不是逐篇“训练效用”的真实标签，其他来源也可能含科学或数学文本。

所有划分之前按 NFKC、大小写和空白归一化的整篇文本去重；额外检查实际评分 token 前缀的完全重复，
并以 MinHash128 检索候选、核对 token 5-gram Jaccard ≥ 0.8 的近重复。近重复检索是近似的，可能漏检。
默认至少 128 个有效预测 token，每篇最多使用前 512 tokens，不添加 EOS 或对话模板。

## 检索评价

在相同 tokenizer、截断、上下文和 mask 下，用 FP32 前向得到逐篇平均 token NLL。
训练也使用 FP32：未更新 M0 的同一 probe batch 在 BF16 autocast 下 NLL 偏差约 0.494，
因此数值预检后弃用 BF16；预检与弃用运行保留在独立目录，不用于正式结果。

- 绝对下降量：`L(M0, x) − L(M1, x)`。
- 相对下降量：`[L(M0, x) − L(M1, x)] / max(L(M0, x), 1e-8)`。
- 随机对照：seed 与文档 ID 的固定 hash，独立于 loss 和类别。

分数越大越优先；负分仍保留在完整候选池中。排序方法只能获得 ID、前后 loss 和 token 数。
类别标签在评分结束后才用于 PR、AP、precision@K、recall@K。
PR 按共同分数阈值合并 ties；AP 为阶梯积分，不与梯形面积混用。
固定 top-K 中的 ties 用独立 ID hash 打破。图中同时画真实随机排序和 20% 正例占比基线。

## 继续训练对照

用户已确认各组均从 **原始 M0** 出发，不从 probe 后的 M1 出发。
每组选择 1,024 篇候选文档，分别为：

- `oracle_target`：已知目标类别正例。
- `other`：其他来源负例的固定随机子集。
- `random`：整个混合候选池的随机 top-K。
- `absolute_loss_drop`、`relative_loss_drop`：两种排序的 top-K。

选择清单在所有配对 seeds 42、43、44 之间保持相同。各组采用相同学习率、optimizer、scheduler，
每次更新使用 8,192 个有效预测 tokens，总共 524,288 tokens（64 步）。probe 也训练 64 步。
训练按文档循环打乱采样；耗尽时可以重访。有效 token 预算不同于去重后的数据量，两者都记录。
跨更新边界的文档保留上下文，仅继续计算尚未用过的 targets；最后一批精确掩码到预算。

主要指标为独立目标测试集的 token 加权 NLL。`other NLL − oracle target NLL` 为正说明目标类别组更好。
同时报告逐 seed 指标、seed 均值/标准差，以及给定模型对下按测试文档配对 bootstrap 的区间。
三个训练 seeds 共用一个 probe 和一个划分；首轮结论不自动推广到其他领域或模型。

## 代码与扩展

```text
methods/
  absolute_loss_drop/{method.py,README.md}
  relative_loss_drop/{method.py,README.md}
  random/{method.py,README.md}
src/picking/domain_retrieval/
  config.py       配置检查与目录约束
  data.py         带类别语料读取、抽样、去重与划分
  training.py     已有权重加载、等 token 训练、完整恢复、逐篇 NLL
  metrics.py      方法加载、排名、PR/AP 与配对统计
  experiment.py   公共实验流程和训练对照
  reporting.py    统计归档和绘图 notebook 生成/执行
configs/domain_retrieval.yaml
scripts/domain_retrieval.py
tests/test_domain_retrieval.py
```

新增一个 `methods/<name>/method.py`，实现 `score(rows, *, seed)`，并补充该目录的 `README.md`。
然后把名字加入 YAML 的 `methods`；若要测它选出的数据的训练效果，同时加入 `continuation.arms`。
方法按目录自动加载，不需要修改公共注册表。当前接口面向基于前后 loss 的打分方法。
方法产物也各自保存在运行目录 `methods/<name>/`。

换领域修改 `data.target_category`；换已有模型可提供 `model.local_path`（HuggingFace 模型和 tokenizer 目录）。
本地数据可配置 `data.local_files`，支持 parquet/JSONL 和可配置的文本、类别字段（支持点号嵌套）。
修改数据、模型或数值配置时使用新的 `run_id`，不改写已完成实验或绕过恢复身份检查。

## 运行与恢复

```bash
.venv/bin/python scripts/domain_retrieval.py check
.venv/bin/python scripts/domain_retrieval.py prepare
DATA_PICKING_GPU_IDS=0 .venv/bin/python scripts/domain_retrieval.py run
```

`check` 输出将使用的有效配置要点，包括 M0 起点、optimizer 保存和自动恢复开关。
`run` 也会完成缺失的数据准备阶段。它只选择 `DATA_PICKING_GPU_IDS` 明确允许的空闲 GPU，
只传一张卡时顺序执行；传多张卡（如 `DATA_PICKING_GPU_IDS=0,1,2`）时，
probe/评分使用首卡，后续独立 arm/seed 在已允许且空闲的卡上用独立进程并行。
父进程持有该节点/run 的独占写锁，每个子进程只写自己的 arm/seed 目录。
没有空闲卡时退出，不干预其他任务。

每 32 步及最后一步将模型、optimizer、scheduler、RNG、epoch/文档/token 游标写入持久化 checkpoint。
同一命令会验证配置、模型和数据身份，并从最近完整 checkpoint 恢复。
传输复用已有的 blob 分块上传/读回检查；其他持久化文件系统可显式设置
`DATA_PICKING_CHECKPOINT_IO=filesystem`。训练日志与运行状态位于新运行自己的目录。

同节点使用独占写锁；持久化 `OWNER.json` 限制跨节点自动接管。
迁移时先确认旧节点写入者已经退出，再人工将该新运行的 owner 转交给新节点。
不要在两个节点同时写相同 run，也不要改写完成标记或绕过 checkpoint 身份检查。
本实验的 `data.py` 和 `training.py` 已记录实现指纹；修改其数值实现时使用新 `run_id`。

完成后自动生成 `/home/v-guanhechen/experiments/YYYY-MM-DD_HH-MM-<topic>/`，时间使用 America/Los_Angeles，
按从旧到新的顺序登记索引。报告、CSV、选数/划分清单、逐篇 loss、训练日志和 `analysis.ipynb` 同目录保存。
图表由 notebook 中的绘图代码执行生成并保留内嵌输出。
`latest_report.json` 指向归档；所有训练和报告完成后才写 `EXPERIMENT_COMPLETED.json`。

针对性检查：

```bash
.venv/bin/python -m pytest -q
```

测试覆盖分数方向、相对分母、负分、PR ties、标签隔离、去重、划分选择边界、
因果 mask、精确 token 游标、配对指标，以及带 dropout 的真实 optimizer/RNG 断点恢复等价性。
CPU 与显式允许的 GPU 上均已检查恢复前后的参数和 optimizer 状态逐项一致。

需要重新执行 GPU 恢复测试时，先选择一张空闲卡，例如：

```bash
DATA_PICKING_GPU_IDS=0 CUDA_VISIBLE_DEVICES=0 DOMAIN_RETRIEVAL_TEST_CUDA=1 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python -m pytest -q \
  'tests/test_domain_retrieval.py::test_full_state_resume_matches_uninterrupted_continuation[cuda]'
```
