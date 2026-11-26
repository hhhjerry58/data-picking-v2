# PR 数据评分与领域继续训练

当前新增实验使用 **Qwen3.5-9B**，在同一 PR 候选池上计算全文／答案两个监督目标的
绝对下降量、相对下降量、GradDot，共六个分布。第一阶段到分布分析为止，选数和续训另行确定。
协议和运行顺序见 [PR 评分说明](docs/PR_SELECTION.md)。
已完成前缀的快速预览、十节点 A100 评分和 Windows 提交入口见
[并行评分说明](docs/PR_PARALLEL.md)；并行代码位于 `src/picking/pr_parallel/`。

新实验入口为 `scripts/pr_selection.py`，公共实现位于 `src/picking/pr_selection/`，
方法继续各自放在 `methods/<name>/`。Qwen3.5 使用独立 `.venv-pr`；下面的旧入口和
`.venv` 保留用于已完成的 arXiv pilot 复现，不改写其数值实现或 checkpoint 身份。

```bash
bash scripts/setup_pr_env.sh
.venv-pr/bin/python scripts/pr_selection.py check
```

## 已完成的 arXiv pilot

从已有预训练 checkpoint `M0` 出发，用目标类别的一小份 probe 数据得到 `M1`，
按微调前后的绝对或相对 loss 变化，从混合候选池中找回同类文档并绘制 PR 曲线。
再从同一个 `M0` 分别用已知同类数据、其他数据及排序选出的数据继续训练，在第三份独立测试集比较 loss。

## Review 入口

| 看什么 | 文件 |
|---|---|
| 当前 PR 六分布实验 | [PR 评分协议](docs/PR_SELECTION.md) · [配置](configs/pr_selection.yaml) |
| 快速预览、十节点 A100 评分与提交 | [并行评分](docs/PR_PARALLEL.md) |
| 实验假设、数据划分和对照 | [实验协议](docs/DOMAIN_RETRIEVAL.md) |
| 首轮模型、类别和预算 | [配置](configs/domain_retrieval.yaml) |
| 各选数方法的公式与实现 | [methods/](methods/README.md) |
| PR/AP 与配对统计 | [metrics.py](src/picking/domain_retrieval/metrics.py) |
| 数据独立性、去重与采样 | [data.py](src/picking/domain_retrieval/data.py) |
| 已有权重加载、等 token 训练与恢复 | [training.py](src/picking/domain_retrieval/training.py) |
| 完整实验流程 | [experiment.py](src/picking/domain_retrieval/experiment.py) |

## 目录

```text
methods/<name>/                  每种选数方法的 method.py 和 README.md
configs/domain_retrieval.yaml    唯一的实验配置
scripts/domain_retrieval.py      唯一的实验运行入口
scripts/{setup_env,env}.sh       环境安装与缓存设置
src/picking/domain_retrieval/    数据、训练、评分、统计和归档
src/picking/                    下载、GPU、存储、日志等公共模块
tests/                          当前实验和公共存储/GPU 测试
docs/                           实验协议及节点运行说明
```

新增方法时增加 `methods/<name>/`，实现统一评分接口并将名称加入配置；不需要修改注册表。
具体约定见 [方法说明](methods/README.md)。

## 运行

需要 Python 3.12；正式首轮使用 H100、PyTorch `2.11.0+cu128` 和 FP32。
依赖版本保存在 [runtime_versions.json](runtime_versions.json) 和 `requirements*.txt`。

```bash
bash scripts/setup_env.sh
source scripts/env.sh
.venv/bin/python scripts/domain_retrieval.py check
DATA_PICKING_GPU_IDS=0 .venv/bin/python scripts/domain_retrieval.py run
```

默认配置为 Pythia-160M 预训练权重、SlimPajama-6B / arXiv、三个配对 seed。
也可只执行 `prepare`，或在完整运行结束后执行 `report`。
允许多张卡时设置 `DATA_PICKING_GPU_IDS=0,1,2`，各 arm/seed 会在空闲卡上独立运行。
所有训练保存 optimizer、scheduler、RNG 和 token 游标；重复运行同一命令会校验身份并自动续接。

语料、权重和 checkpoint 位于 `/mnt/tianzhu_hot/data-picking-v2`。
报告、统计、代码快照和执行过的绘图 notebook 归档到 `/home/v-guanhechen/experiments/`；
运行目录内的 `latest_report.json` 指向报告，`EXPERIMENT_COMPLETED.json` 表示全部完成。
节点准备、状态位置和迁移步骤见 [DEVNODE.md](docs/DEVNODE.md)。

## 检查

```bash
.venv/bin/python -m pytest -q
```

默认测试使用 CPU；GPU 恢复测试须显式选择空闲卡并启用，命令见实验协议。
