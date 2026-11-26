# 快速预览与多节点 PR 评分

原始 `pr_selection` 数值代码和 v6 结果保持不变。加速实现位于
`src/picking/pr_parallel/`，每个新运行明确引用原 M1 和 Q 梯度的 checkpoint 指纹。

## 已完成的快速预览

原任务已完成的 2,141 条 PR 的 M0 loss 和 GradDot 已全部保留。
用四个单卡 H100 前向副本补算 M1，得到六个先导分布。
这是按原任务 ID 排序先完成的前缀，不是全池随机抽样。
报告位于 `/home/v-guanhechen/experiments/2026-11-25_20-34-pr-score-preview/`。

## A100 运行方式

用户已授权十个节点、每节点八张 A100 80GB。
每两张卡运行一个完整语言模型评分副本，每节点四个 worker，总共四十个。
每个 worker 接收候选行号 `worker_id, worker_id+40, ...`，完整覆盖 20,000 条。
不重跑 probe 或 Q 梯度，不做跨节点 DDP/all-reduce，不计算 T loss。

- `parent.py`：在原配置下校验原 checkpoint，再只读复用。
- `preview.py` / `report.py`：部分样本 M1 补算、校验和 notebook。
- `backend.py`：FP64 内积在 GPU 累加，每 GPU 最后读回一次；不投影或归一化梯度。
- `bundle.py`：输出 FP32 M1 参数与 Q 梯度分片，保留原 checkpoint 来源。
- `cluster.py`：节点准备、独立 worker、校验与严格合并。
- `deployment.py`：冻结代码、依赖与输入身份；拒绝不匹配的部署。
- `scripts/pr_cluster_node.sh`：使用已打包的 Python/依赖，避免十个节点重复安装。
- `scripts/submit_pr_a100.py`：只依赖标准库，在已有 amlt 项目目录生成或提交 YAML。

双卡与原四卡 BF16 反向不是逐位等价：本地检查见
`/mnt/tianzhu_hot/data-picking-v2/runs/pr-acceleration-benchmark-v2-20261125/profile.json`。
同一梯度上的新旧 FP64 内积累加通过严格一致性检查；不同设备划分有轻微梯度差异。
因此集群使用独立运行，重新计算所有 PR，不把旧 H100 结果伪装成同一批 A100 结果。

节点先检查环境和数值：NLL 差 ≤ `5e-4 * max(1, |reference|)`，
GradDot 差除以 `||gQ|| ||gx||` ≤ `1e-3`，梯度范数相对差 ≤ 2%。
这是预先规定的跨设备 BF16 检查，不会自动放宽。失败时停止该 worker，保留诊断。

每个节点有独立的 run_id/OWNER；worker 分别写入自己的目录。
记录逐条在本地保存，每 32 条持久化；恢复时校验 bundle、代码/环境部署及分片身份。
合并要求四十个 worker 完整，PR ID/监督目标无重复、无缺失，并核对结果文件校验值。
完整结果仍需单独生成最终分析；先导预览不能当作全量结果。

## 提交机命令

在已登录 Azure/amlt 的项目目录运行（例如用户给的 `~/rl`）：

```bash
python3 /path/to/submit_pr_a100.py --smoke --submit
python3 /path/to/submit_pr_a100.py --submit
```

第一条只启动一个八卡节点做环境/数值检查。通过后第二条启动十个独立节点。
去掉 `--submit` 只生成配置。计算代码从挂载的 bundle 读取，不依赖提交机 checkout 的代码。
Windows 的私钥路径不能直接在当前 Linux 节点使用；提交需可用的 SSH agent/Linux 密钥，
或由用户在已登录的提交机执行上面命令。

## 从 Windows 提交

已生成自包含脚本：[`submit.ps1`](../configs/pr_a100_generated/submit.ps1)。
把它保存到 Windows 后，在该目录的 PowerShell 中执行：

```powershell
.\submit.ps1 -Phase Smoke
# 一个节点的环境与 M0/M1 数值检查通过后：
.\submit.ps1 -Phase Full
```

默认使用 `C:\Users\v-guanhechen\.ssh\lidong\priv.openk` 连接
`lidong1@microsoft.com@10.209.224.207`（脚本用 SSH 的 `-l` 单独传用户名）。
脚本只让 Windows OpenSSH 读取密钥，不会复制密钥。提交程序已内嵌，Windows 不需要 Python，
提交机也不需要本仓库；远端需要 Python 3 和已经能用的 AMLT 登录环境。

脚本会在远端寻找含 `.amltconfig` 的 `~/rl`、`~/ytz/rl` 或 `/home/v-guanhechen/rl`。
实际目录不在这些位置时，显式指定：

```powershell
.\submit.ps1 -Phase Smoke -RemoteProject /actual/path/to/rl
```

`-Phase Generate` 只生成 YAML。提交命令使用用户现有示例中的 `amlt run ... -y`。
Smoke 和 Full 使用不同输出目录；正式候选评分不会覆盖先导预览。

共享包位于 `/mnt/tianzhu/data-picking-v2/runs/pr-a100-bundle-v1-20261125/`，
在当前机器对应 `/mnt/tianzhu_hot/data-picking-v2/runs/pr-a100-bundle-v1-20261125/`。
其中 `READY.json` 只表示输入和部署准备完成，**不表示 A100 检查已通过或作业已提交**。

Smoke 成功的依据是以下文件及四个 worker 的 `VALIDATED.json`，不能只看提交命令退出成功：

```text
runs/pr-qwen35-a100-10n-v1-20261125-smoke/node-00/VALIDATED.json
```

Full 的四十个 `worker-*/COMPLETED.json` 全部齐全后，在当前机器合并：

```bash
PYTHONPATH=src .venv-pr/bin/python -m picking.pr_parallel.cluster merge \
  --outputs /mnt/tianzhu_hot/data-picking-v2/runs/pr-qwen35-a100-10n-v1-20261125 \
  --bundle /mnt/tianzhu_hot/data-picking-v2/runs/pr-a100-bundle-v1-20261125 \
  --output /mnt/tianzhu_hot/data-picking-v2/runs/pr-qwen35-a100-10n-v1-20261125/merged
```

每个 worker 的 `worker.log`、`m0_validation.json`、`m1_validation.json` 和逐条 journal
都位于对应节点目录。OWNER 阻止在新主机上自动接管旧输出；更换节点恢复之前必须确认旧任务退出，
再明确交接 OWNER。同主机重复启动也受进程锁保护。

## 本地验证与重新打包

```bash
.venv/bin/python -m pytest -q tests/pr_selection/test_parallel.py
python3 scripts/submit_pr_a100.py --output configs/pr_a100_generated \
  --windows-wrapper configs/pr_a100_generated/submit.ps1
python3 scripts/submit_pr_a100.py --smoke --output configs/pr_a100_generated
```

代码与运行环境只允许发布到尚无 `READY.json` 的导出包。改变数值实现、依赖或分片时，
使用新 bundle 和 campaign，不改写已发布部署：

```bash
PYTHONPATH=src .venv-pr/bin/python -m picking.pr_parallel.deployment \
  --bundle /path/to/new/exported/bundle --runtime /path/to/runtime.tar.gz
```

运行环境和代码归档校验和用于部署身份校验；每个节点还会检查实际复制的模型、Q 梯度和候选数据。
CPU 检查不能替代实际 A100 环境的首节点验证。
