# Qwen3.5-9B：PR 四个 loss-drop 分布

第一阶段仅计算分数并分析分布，再讨论选数。没有开发集、T loss 或最终 PR 续训。

## 固定数据与目标

- M0：`Qwen/Qwen3.5-9B`，revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。
- 使用全部语言模型参数，视觉模块不加载。FP32 参数、BF16 矩阵计算、FP32 token cross entropy。
- Q/T 按 DeepSWE 任务固定 hash 划分，各 53 个任务；同任务所有 trial 同侧。
- Q 的条件保留完整历史，监督沿用 Wenhui 的动作正文、函数名、参数键和值；不计独立 reasoning 或格式外壳。
- 使用目标 token 平均 NLL，不使用 BPB。
- P 固定为同一批 20,000 条 `base/8k/git_diff` PR，完整输入不超过 8,192 tokens，不截断。
- 全文和答案目标共用原始 `text` 的输入 token；全文从第 2 个 token 预测，答案从唯一 `## Answer` 后开始。
- PR 的质量字段不进入评分，也不充当训练效用 ground truth。

候选池先固定选择最多 64 个来源分片，再按 ID-hash reservoir 取 20k。该池不是整个 PR 仓库的均匀样本。
按 PR ID、query ID、完整文本去重，排除轨迹显式引用的 GitHub PR/issue；语义补丁重合仍未完全检查。
两种目标始终使用相同候选记录。

## 两种方法

每种监督目标分别计算 `Lx(M0)` 和 `Lx(M1)`：

- 绝对下降量：`Lx(M0) - Lx(M1)`。
- 相对下降量：`[Lx(M0) - Lx(M1)] / max(Lx(M0), 1e-8)`。

两种方法乘两个监督目标，共四个分布。保留负值，不裁剪或据此自动选数。
候选阶段仅做前向，每个模型的主体 forward 共享全文与答案两个 loss mask。

## 固定 M1 与历史导入

M1 由 Q 微调得到，两种监督目标共用同一个 M1。当前配置用 `probe_from_run` 引用已经完成的 v6 probe。
导入器重新计算原源码快照、原配置和固定依赖的身份，检查 Q 数据与训练参数，核对 checkpoint
指纹、manifest、完整文件校验和及完成状态。导入信息记录到新 run 的 `imported_probe.json`。
旧 run 的配置、checkpoint、结果和指纹保持不变。

如果不设置 `probe_from_run`，则执行新 probe：Q 一轮，每步累积 4 条，梯度分母采用
`Q 总有效 tokens × 当步记录数 / Q 记录数`，使均匀记录采样对应全局 token 目标。
AdamW LR 2e-6，betas=(0.9,0.95)，eps=1e-8，weight decay=0，warmup=2，clip=1。
FP32 master 和 Adam 状态在 CPU，每次保存完整 optimizer、scheduler、RNG、游标，自动恢复。
全参数训练用完整历史，超过 32K 的反向启用 activation offload，不截断历史或梯度。

## 数值后端

`scoring.attention_backend` 支持 `default` 与 `flash`，只影响候选 forward；新 probe 保留原训练后端。
默认值沿用原 SDPA/cuDNN。Flash 的首次长度开销更小，但与 cuDNN 的 loss-drop 存在可测差异，
因此更换后端必须使用新 run，并重新计算 M0/M1，不能混用旧缓存。
线性注意力使用固定 FLA，训练保留 TileLang 和 causal-conv1d 依赖；旧 `.venv` 不升级。

## 运行

```bash
.venv-pr/bin/python scripts/pr_selection.py check
DATA_PICKING_GPU_IDS=0,1,2,3 bash scripts/run_pr_scores.sh configs/pr_selection.yaml
```

流水线顺序为 prepare → 导入 M1（或 preflight → probe）→ 多卡纯前向 score → analyze。
原数据可通过 `prepared_from_run` 复用，但数据实现签名、tokenizer、采样配置必须一致。
新数值实现使用新 run_id；不得改旧配置来绕过身份检查。

代码职责：`data/` 处理记录与 mask，`modeling/` 提供模型和 loss，`training.py` 执行新 Q probe，
`imports.py` 导入历史 M1，`scoring.py` 计算及配对 loss，`parallel.py` 管理单卡副本，
`diagnostics.py` 和 `notebooks/score_distributions.ipynb` 生成实际分析。

评分文件位于 `runs/<run_id>/workers/<worker>/scores/<objective>/m0.jsonl` 与 `m1.jsonl`。
合并后为 `scores/paired_scores.parquet`，完成标志为 `SCORING_COMPLETED.json`。
详细运行与恢复见 [本地并行说明](PR_PARALLEL.md)。

分析保存到 `/home/v-guanhechen/experiments/data-picking/`，图由 notebook 单元生成并保存内嵌输出。
检查分布、长度关系和方法排序相关性；分布本身不能证明在独立 T 上的训练收益。
