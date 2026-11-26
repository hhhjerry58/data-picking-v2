# Qwen3.5-9B：PR 六个评分分布

第一阶段只生成实际评分及分布，随后再讨论选数策略。没有开发集，没有 T loss，
没有最终 PR 续训。原有 arXiv pilot 的实现及数值身份保持不变。

## 固定协议

- M0：`Qwen/Qwen3.5-9B`，revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。
- 只加载完整语言模型参数，包括 embedding 和输出 head；视觉参数不加载、不更新。
- FP32 参数、梯度与 AdamW 状态，BF16 矩阵计算，FP32 token cross entropy，FP64 梯度内积累加。
- 保留轨迹全部历史，原生 262,144 窗口，不做位置扩展。使用 Qwen3.5 官方模板。
- Q/T 按 DeepSWE 任务固定 hash 划分，各 53 个任务；同任务所有 trial 同侧。
- Q loss 沿用 Wenhui 的语义字段：参考动作正文、函数名、参数键和值；历史及工具结果只作条件。
  不评分格式外壳，不含独立 reasoning，使用目标 token 平均 NLL，不使用 BPB。
- P：同一批 20,000 条唯一 PR，来源 `base/8k/git_diff`。按 Qwen tokenizer 检查完整输入 ≤8192，
  不截断；两种 mask 共有输入，全文从第 2 个 token 预测，答案 mask 从唯一 `## Answer` 后开始。
- `score` 质量字段不进入任何方法。按 PR ID、query ID、完整文本身份去重；
  排除轨迹中显式 GitHub PR/issue 引用及可配置额外排除清单。语义补丁重合尚不能由这些检查完全排除。

顶层 PR JSON 是分片路径清单。当前先固定随机选择最多 64 个来源分片，
在其全部可用去重记录上用 ID-hash reservoir 选 20,000 条；这是固定子池实验，
不声称对整个 PR 仓库做了均匀采样。数量不足时报错，调整分片预算须使用新 run_id。

## 方法与固定 probe

每条 PR 的当前目标平均 NLL 为 Lx。绝对分数是 Lx(M0)−Lx(M1)，
相对分数再除以 max(Lx(M0),1e−8)。两种监督目标共用同一个 M1。

GradDot 在 M0 对全部语言模型参数计算 gQ·gx，不投影、不选层、不做 cosine 归一化。
gQ 按 Q 的所有有效目标 token 统一归一化；gx 按该 PR 当前监督目标的 token 数归一化。
候选梯度逐条计算后丢弃，只保留内积和范数。Q 梯度保存完整恢复状态。

probe 默认仅遍历 Q 一轮，每步累积 4 条记录。梯度分母为
`Q 总有效 tokens × 当步记录数 / Q 记录数`，使均匀记录 minibatch 对全局 token NLL
给出无偏估计，避免与 GradDot 的 Q 权重不同；日志另报告当步实际 token 平均 NLL。
AdamW：LR 2e−6，betas=(0.9,0.95)，eps=1e−8，weight decay=0，warmup=2 步，clip=1。
这是预先固定的保守配方，不承诺最佳；不能用 T 调参或选择 M1。
保存 optimizer、scheduler、RNG 和记录游标，自动恢复，保留最近两个完整 checkpoint。
FP32 master 参数和 Adam 状态放在 CPU，所有语言模型参数均更新并同步回 GPU。
checkpoint 的模型值与 CPU master 完全相同，恢复时先重建 master，再恢复 optimizer。

## 代码职责

| 路径 | 职责 |
|---|---|
| `data/` | PR 解析、任务划分、共同候选池；T 仅保留清单 |
| `modeling/masks.py` | 原生模板语义标记、精确往返核对、字节到 token 区间 |
| `modeling/loss.py` | 分块词表 loss、完整上下文反向的链式法则 |
| `modeling/qwen35.py` | 固定权重、文本参数、多 GPU 模型并行与 FLA |
| `modeling/gradients.py` | Q 梯度累积及恢复 |
| `methods/grad_dot/` | GradDot 准备、逐条方法评分 |
| `training.py` | 固定 probe 与 optimizer/RNG 恢复 |
| `scoring.py` | 配对缓存、三种方法输出；无选数策略 |
| `artifacts.py` | 独占写入、身份检查、checkpoint 和逐行评分恢复 |
| `diagnostics.py`、`notebooks/score_distributions.ipynb` | 统计与实际 notebook 图表 |

普通训练后端会保存完整序列×词表 logits，长历史下开销过大。这里逐块计算输出 head 的
loss 和 hidden adjoint，再一次性反传完整上下文图。不是 detach 历史的近似梯度。
GPU 按层划分，FP32 模型参数分布在显式允许的 GPU 上，optimizer 使用 CPU offload。尚需真实最长样例通过预检，
不能把 CPU 检查或 tokenizer 长度检查当作显存验证。
超过 32K 的反向使用 CPU activation offload，只搬运保存的激活，不截断历史或梯度。
预检在 optimizer 对应设备保留两份 FP32 Adam moment，并预先分配完整累计梯度，
防止只测第一个 microbatch 而低估峰值。

## 环境和阶段入口

`scripts/setup_pr_env.sh` 创建 `.venv-pr`，通过只读 Python 路径复用旧 `.venv` 的 Torch
和分析依赖，独立安装固定版本的 Transformers/FLA。不会卸载或升级旧环境。
H100 上使用 FLA 支持的 TileLang 后端，避免 Triton 3.6 的已知 gated backward 错误；
同时固定 causal-conv1d。模型大权重从持久目录校验复制到 RAM mirror 后加载。

```bash
bash scripts/setup_pr_env.sh
.venv-pr/bin/python scripts/pr_selection.py check
.venv-pr/bin/python scripts/pr_selection.py prepare
DATA_PICKING_GPU_IDS=0,1,2,3 .venv-pr/bin/python scripts/pr_selection.py preflight
DATA_PICKING_GPU_IDS=0,1,2,3 .venv-pr/bin/python scripts/pr_selection.py probe
DATA_PICKING_GPU_IDS=0,1,2,3 .venv-pr/bin/python scripts/pr_selection.py score
.venv-pr/bin/python scripts/pr_selection.py analyze
```

完成有效配置检查后，也可使用可恢复的第一阶段串行入口：

```bash
DATA_PICKING_GPU_IDS=0,1,2,3 bash scripts/run_pr_scores.sh configs/pr_selection.yaml
```

该入口使用已冻结的 JSON 配置继续各阶段，状态在 `pipeline_status.json`，
子阶段状态在 `status.json`，日志在 `logs/`。它不会进入 PR 选数或续训。

`preflight` 必须验证最长 Q 的完整反向和两种 PR mask；`probe`、`score` 检查同一配置及代码身份的
通过标记。配置变化使用 `--run-id` 指定新运行，不能复用不同身份的 checkpoint 或评分缓存。
只使用明确允许且空闲的卡；同一 run 由单节点独占写入，不自动接管其他节点。

产物在 `/mnt/tianzhu_hot/data-picking-v2/runs/<run_id>/`：
`data/`、`pretraining/probe/`、`pretraining/target-gradient/`、`scores/<objective>/`、
`methods/<method>/<objective>/scores.parquet`。每条候选分数可独立恢复。
全部六个评分完成后，真实实验分析才归档到 `/home/v-guanhechen/experiments/`；
notebook 直接包含绘图代码，并执行、保存内嵌输出。
`prepared_from_run` 可以复用另一运行已经完成的数据集，但必须匹配任务划分、采样配置、
tokenizer、模板及数据代码身份，并检查复制后的文件完整性。新数值后端使用新的 run_id；
不会修改原数据运行或绕过 checkpoint 身份检查。运行状态见 `status.json`，
每个数值运行在 `source/` 保存实际代码快照。

分布保留负值和极端值，报告分位数、正负比例、ECDF、长度关系、排名相关性和累计 token 曲线。
分布形态不是训练效用 ground truth。没有选数规则与续训预算时不自动进入下一阶段。

## 针对性验证

```bash
.venv/bin/python -m pytest -q tests/pr_selection
```

CPU 检查覆盖完整上下文的梯度、归一化、GradDot 方向、同输入双 mask、任务隔离、
不完整评分行恢复、optimizer 和数据游标续接。真实 GPU 预检使用上面的显式分卡命令。
