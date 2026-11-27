# PR 数据评分与领域继续训练

20k 四个评分分布已完成。当前扩大并评分 100k 候选，准备 32 组单节点八卡 A100 实验：
两种监督 × 两种评分 × 四个百分比组，加上两种监督各 8 个 Random seed。
每组 100 步、每步约 256k 输入 tokens、每 5 步保存完整 checkpoint，集群由用户自行提交。
见 [集群协议及提交说明](docs/PR_CLUSTER.md)。原八组小预算实验已暂停，历史协议见 [这里](docs/PR_USEFULNESS.md)。

集群提交入口集中在 [`yamls/guanheng/pr_selection_100k/`](yamls/guanheng/pr_selection_100k/README.md)。
在已登录 AMLT 的机器 clone 本分支后，从仓库根目录直接提交：

```bash
amlt run yamls/guanheng/pr_selection_100k/smoke_a100.yaml pr100k-smoke-v1 -y --sla Premium
```

```bash
DATA_PICKING_GPU_IDS=0,1,2,3 bash scripts/run_pr_scores.sh configs/pr_selection_100k.yaml
```

当前实验使用 **Qwen3.5-9B**，对同一批 20,000 条 PR 计算全文／答案两个监督目标的
绝对、相对 loss 下降量，共四个分布。第一阶段只到分布分析，不选数、不继续训练 PR、不计算 T loss。

每张 GPU 运行一个独立的前向评分副本，M0/M1 各做一次主体 forward；每个 forward 共享两个监督区域。
默认复用已完成的 Q 微调 checkpoint M1，校验原训练指纹后导入新的评分 run。
当前默认保持原 SDPA/cuDNN 数值口径；Flash Attention 可在新 run 中单独配置和验证。

| Review 内容 | 入口 |
|---|---|
| 数据、目标和四种分布 | [实验协议](docs/PR_SELECTION.md) · [配置](configs/pr_selection.yaml) |
| 多 GPU 前向、状态、恢复和历史 M1 导入 | [本地并行评分](docs/PR_PARALLEL.md) |
| 方法公式 | [methods/](methods/README.md) |
| 逐模型 loss 与严格配对 | [scoring.py](src/picking/pr_selection/scoring.py) |
| 单卡 worker 和分片合并 | [parallel.py](src/picking/pr_selection/parallel.py) |
| 历史源码及 checkpoint 身份检查 | [imports.py](src/picking/pr_selection/imports.py) |

```bash
bash scripts/setup_pr_env.sh
.venv-pr/bin/python scripts/pr_selection.py check
DATA_PICKING_GPU_IDS=0,1,2,3 bash scripts/run_pr_scores.sh configs/pr_selection.yaml
```

`probe_from_run` 指定已有 M1 时，流水线校验并导入 checkpoint；否则先执行 Q 的预检与全参数微调。
新建训练始终保存 optimizer、scheduler、RNG 和游标，并启用自动恢复。评分阶段逐条缓存，重复执行可续接。
当前节点只使用明确指定且空闲的 GPU；同一 run 只允许一个节点写入。

```text
methods/<name>/                方法公式和 method.py
yamls/guanheng/<experiment>/   集群任务 YAML 与提交说明
configs/pr_selection.yaml     当前 PR 实验配置
scripts/pr_selection.py       唯一的 PR 阶段入口
scripts/run_pr_scores.sh      可恢复流水线
src/picking/pr_selection/     数据、训练、纯前向评分、并行、身份检查、报告
notebooks/                    报告绘图模板
src/picking/domain_retrieval/  已完成 arXiv pilot 的固定数值实现
```

模型、数据、checkpoint 和评分缓存位于 `/mnt/tianzhu_hot/data-picking-v2`。
实际分析集中在 `/home/v-guanhechen/experiments/data-picking/`，同时维护项目与 experiments 总索引。
历史配置、产物和源码快照保留原身份，清理当前仓库不会改写已有实验结果。

## 已完成的 arXiv pilot

旧 pilot 从 M0/M1 的 loss 下降找回目标类别，再做等 token 继续训练对照。
实现与旧 `.venv` 保留用于复现，详见 [协议](docs/DOMAIN_RETRIEVAL.md)、
[配置](configs/domain_retrieval.yaml) 和 [节点说明](docs/DEVNODE.md)。

```bash
.venv/bin/python -m pytest -q
```

默认测试使用 CPU；真实 GPU 测试须显式选择空闲卡。
