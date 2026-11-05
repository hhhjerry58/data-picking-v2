在另一台 devnode 继续本轮实验
===========================

本轮还需要完成 FT-relative / seed 43 的末段训练、seed 43 的三组评估，以及 seed 44 的三组训练与评估。
已验证的恢复点是 **step 11,500 / 3,013,184,000 有效预测 tokens**，包含 optimizer、scheduler、RNG 和数据游标。
旧节点的等待队列已为本次交接停止。一个 run 同时只在一台节点执行。

1. **克隆仓库，挂载相同的持久化数据。**

   ```bash
   gh repo clone hhhjerry58/data-picking-v2
   cd data-picking-v2
   ```

   将原存储挂载到 `/mnt/tianzhu_hot`，确认下面的目录可读写：

   ```text
   /mnt/tianzhu_hot/data-picking-v2/
     cache/assets/pythia160m/
     data/processed/
     runs/math-lossdrop-fasttext-qwen15b-pythia160m-n100000-seed20261102/
   ```

   Git 中包含代码、固定配置、fastText 源码和小型结果快照。语料、token 文件、完整模型和 checkpoint 留在上述存储。
   已有 manifest 记录了绝对路径，因此续跑时保持这个挂载路径。不要重新生成或改写已有数据清单。

2. **建立独立的 Python 环境。**

   需要 Python **3.12**、Python 开发头文件、C++17 编译器、`make` 和兼容 CUDA 12.8 的 NVIDIA 驱动。
   本轮使用 H100 80GB；冻结的 micro batch 32 需要同级显存。建议使用相同型号以减少跨硬件数值差异。

   ```bash
   # 运行目录需要较大 POSIX 空间：单个训练任务至少 32 GiB，剩余整轮建议预留 100–160 GiB。
   # Docker 默认的 64 MiB /dev/shm 不够；可以改到节点本地的快速磁盘。
   export DATA_PICKING_RUNTIME_ROOT=/dev/shm/data-picking-v2-runtime
   bash scripts/setup_env.sh
   source scripts/env.sh
   ```

   安装脚本固定 PyTorch `2.11.0+cu128`、Transformers `4.57.1` 等版本，并编译本仓库的 fastText 扩展。
   `requirements.txt`、`requirements-dev.txt` 和 `runtime_versions.json` 记录了依赖。
   环境独立安装，checkpoint 上传代码也已包含在仓库内。

3. **设置节点 GPU 和 checkpoint 存储方式。**

   ```bash
   export DATA_PICKING_GPU_IDS=0,1,2
   # 支持只选一张卡，例如 DATA_PICKING_GPU_IDS=0；各任务仍使用相同的有效全局 batch。
   ```

   默认从当前 blobfuse 进程读取挂载配置，使用已验证的分块上传方式。如果无法自动定位配置：

   ```bash
   export DATA_PICKING_BLOB_CONFIG=/path/outside/repo/blobfuse2.yaml
   ```

   该方式需要外部挂载配置中具有 `azstorage.endpoint`、`container` 和 `sas`。凭据文件不应放进仓库。
   对于使用其他认证方式或其他持久化文件系统的挂载，可以显式选择文件系统方式：

   ```bash
   export DATA_PICKING_CHECKPOINT_IO=filesystem
   ```

   文件系统方式仍保存完整状态，执行 fsync、关闭文件和 SHA-256 读回校验；上传性能可能不同。
   这些是节点运行/传输设置，不修改学习率、batch、token 预算或已有训练指纹。

4. **核对恢复条件，再启动。**

   ```bash
   .venv/bin/python scripts/check_resume.py --full-checkpoint
   .venv/bin/python scripts/start_run.py
   ```

   检查程序会确认框架版本、训练文件哈希、持久化数据、完整 optimizer 状态和节点资源。
   如暂时没有空闲 GPU，可以分别用 `check_resume.py --no-gpu` 检查存储，并用
   `start_run.py --wait-for-gpu` 排队。`start_run.py` 表示明确恢复，会清除交接时留下的停止标记。

   新节点会先续接 seed 43 的 FT-relative，再完成后续评估和 seed 44。已经完成的评分、分类器、
   选数以及模型会根据完成标记跳过。跨节点发现旧的 RAM mirror 不存在时，会从持久化 checkpoint 重新加载。

   **不要修改 `src/picking/pretraining.py` 或 `configs/experiment.yaml` 的 `pretraining` 部分来绕过恢复检查。**
   现有 checkpoint 会验证它们；`save_optimizer: true` 和 `auto_resume: true` 保持启用。

5. **查看状态与结果。**

   ```bash
   .venv/bin/python scripts/status.py
   # 显式停止，保存过的 checkpoint 可继续使用：
   .venv/bin/python scripts/stop_pipeline.py
   ```

   每个 seed 的三组评估结束后，报告自动写到本地 checkout 的 `experiments-analysis/`。
   持久化运行目录中的 `latest_report.json` 指向最新报告，`EXPERIMENT_COMPLETED.json` 表示全部完成。
   控制进程、watchdog、日志同步进程由启动入口放到后台；状态带节点名，避免把另一台节点的 PID 当成本机进程。

当前首轮结果见 [seed 42 报告](../experiments-analysis/2026-11-03_050912Z_math-lossdrop-fasttext-qwen15b-pythia160m-n100000-seed20261102_seeds-42/REPORT.md)。
历史报告中 `/home/v-guanhechen/...` 路径是生成时的 provenance。续跑剩余任务所需的运行数据均在持久化根目录中。
