# PR 小预算八组与独立 T

从已完成的 20k 分数生成八组：全文／答案各有 Top-Absolute、Top-Relative、Random、Bottom-Absolute。
每组按固定排序取不超过 100 万输入 tokens 的完整 PR 前缀，不截断、不重复训练。
Random 固定 seed 42，两种监督区域使用同一随机 PR 集合。

| 组 | 答案目标 PR 数 | 全文目标 PR 数 |
|---|---:|---:|
| Top-Absolute | 207 | 213 |
| Top-Relative | 283 | 194 |
| Random | 302 | 302 |
| Bottom-Absolute | 442 | 527 |

全部从原 M0 加载独立模型；M1 只参与了上游评分，不作为这轮训练的起点。
选中记录用 seed 42 打乱一次，按累计输入 token 的近似等分位置形成恰好 32 个非空批次。
全部记录每条只用一次，不为对齐步数复制短组。

所有语言模型参数更新，视觉模块不加载。参数、梯度和 Adam 状态 FP32，矩阵计算 BF16。
AdamW LR 2e-6，betas=(0.9,0.95)，eps=1e-8，weight decay=0，warmup=2，clip=1。
各步骤的梯度分母固定为该组总监督 tokens / 32；日志同时记录当步实际 token 平均 NLL。
输入 token 预算相近并不意味着监督 token 数相同，报告须列出两者。

默认 SDPA/cuDNN 和 FLA/TileLang 沿用原数值后端；不在看过 T loss 后切换后端或调整超参。
四张 H100 分成两个双卡 worker，同一节点内每组有独立 checkpoint 和日志。
每 16 步及最终一步保存 optimizer、scheduler、RNG、批次清单、游标和全部参数；自动恢复。
保留两份持久化 checkpoint，每个活跃组只保留最新本地 mirror；大文件串行保存，控制 RAM 占用。

T 严格按第一阶段冻结的 53 个任务、135 条记录读取；没有开发集，不重新划分 Q/T。
历史完整保留，使用相同原生模板和参考动作语义 mask，不截断长轨迹。
M0 和八组最终模型都计算同一 T 的 token 平均 NLL，另报告任务平均 NLL。
每个任务的全部轨迹成组做配对 bootstrap；同一监督目标下比较各组与 Random 的差值。
这是单训练 seed 的 pilot，任务 bootstrap 不代表跨训练 seed 的不确定性。

配置在计算 T loss 前固定。T 只用于最终评测，不用于挑 checkpoint、修改预算或决定各组超参。
GPU 预检只检查完整前向／反向能否执行与显存，预检不修改参数。

```bash
.venv-pr/bin/python scripts/pr_usefulness.py prepare
DATA_PICKING_GPU_IDS=0,1,2,3 DATA_PICKING_BLOB_CONFIG=/path/to/checkpoint-storage.yaml \
  .venv-pr/bin/python scripts/pr_usefulness.py run
```

`src/picking/pr_selection/continuation/` 分别负责数据与批次、训练、T 评测、调度和报告。
产物位于 `runs/<run_id>/data/arms/`、`pretraining/<arm>/`、`evaluation/<arm>/`、`logs/lane-*.log`。
`EXPERIMENT_COMPLETED.json` 仅在八组及报告全部完成后生成；`latest_report.json` 指向 experiments/data-picking 下的实际报告。
