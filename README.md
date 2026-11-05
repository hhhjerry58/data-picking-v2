数学 loss-drop 数据选择实验
==========================

已批准的计划见 [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)，正式标注集为 **100,000 篇**。
配置：[configs/experiment.yaml](configs/experiment.yaml)。三组为 Random、FT-delta、FT-relative；
现有 1.5B 数学 SFT 模型提供标签，每组从零预训练 Pythia 160M、3.2B 有效预测 tokens。

源码位于 `src/picking/`，脚本位于 `scripts/`，必要的完整恢复/预算/污染检查位于 `tests/`。
在新节点运行 `bash scripts/setup_env.sh` 建立独立环境，再 `source scripts/env.sh`，使用 `.venv/bin/python`。
通过 `DATA_PICKING_GPU_IDS` 指定允许使用的物理 GPU 编号，流水线只选择其中空闲的卡。

**迁移与续跑步骤见 [docs/DEVNODE.md](docs/DEVNODE.md)。** 当前节点的等待队列已停止，可以在挂载了同一存储的新节点继续。
数值训练文件保持原来的字节哈希，可恢复既有 optimizer / scheduler / RNG / 数据游标。

```bash
gh repo clone hhhjerry58/data-picking-v2
cd data-picking-v2
# 先把原存储挂载到 /mnt/tianzhu_hot
export DATA_PICKING_RUNTIME_ROOT=/dev/shm/data-picking-v2-runtime
bash scripts/setup_env.sh
source scripts/env.sh
export DATA_PICKING_GPU_IDS=0,1,2
.venv/bin/python scripts/check_resume.py --full-checkpoint
.venv/bin/python scripts/start_run.py
```

本次交接：seed 42 的三组已训练并评估；seed 43 的 Random / FT-delta 已训练，FT-relative 可从 step 11,500 恢复；seed 44 尚待运行。
查看 [seed 42 报告](experiments-analysis/2026-11-03_050912Z_math-lossdrop-fasttext-qwen15b-pythia160m-n100000-seed20261102_seeds-42/REPORT.md)
和 [10 万篇标签分布分析](experiments-analysis/2026-11-03_073409Z_label_score_distributions_n100000/README.md)。

持久化产物根目录：`/mnt/tianzhu_hot/data-picking-v2`。
本轮运行目录：
`/mnt/tianzhu_hot/data-picking-v2/runs/math-lossdrop-fasttext-qwen15b-pythia160m-n100000-seed20261102`。

数据准备过程记录于运行目录的 `prepare_data.log` 和 `data_progress.json`。
最终实验结果以完成标记、真实训练 token 计数和评估产物为准；实现检查不作为选数效果结果。

查看状态：

```bash
source scripts/env.sh
.venv/bin/python scripts/status.py
```

完整流程由 `scripts/run_pipeline.py` 串联，已有输出会进行指纹/校验和检查后复用。
各任务带独占写锁；训练保存完整状态并自动续接。执行中使用 POSIX live logs，
`scripts/sync_logs.py` 定期把日志快照保存到持久化运行目录。
每个配对 seed 的三组评估完成后，自动生成 `experiments-analysis/` 下的报告、CSV 和图表。
`latest_report.json` 指向最近一份完整报告；九次训练及评估结束时写入 `EXPERIMENT_COMPLETED.json`。
后台 watchdog 在控制进程意外退出时最多重启三次；各阶段复用已完成产物和完整训练状态。
需要主动停止时运行 `.venv/bin/python scripts/stop_pipeline.py`；显式停止标记会阻止自动重启。

2026-11-05 UTC 修复了控制进程和 watchdog 因挂载 I/O 错误而同时退出的问题。
心跳、控制状态和停止指令先写 POSIX RAM 与本地磁盘，日志同步进程再尽力镜像到持久化挂载；
挂载状态文件写入失败不会终止控制进程。数据与 checkpoint 的完成标记仍要求持久化成功。
状态查询会核对 PID 是否仍存活，并区分正在运行、等待 GPU 和已中断；恢复时跳过已完成阶段。

fastText 固定源码版本与本地修复说明见 [VENDOR_PATCHES.md](vendor/fasttext/VENDOR_PATCHES.md)。
修复完整初始化与状态保存，保留上游分类特征、loss 和 SGD 更新。

代码校验：`python -m pytest -q tests`。启动迁移前可用 `scripts/check_resume.py --no-gpu` 在 CPU 上检查配置与存储。
大数据和 checkpoint 不在 Git 中；仓库保留代码、配置、依赖、fastText 源码以及小型报告/图表/结果快照。
