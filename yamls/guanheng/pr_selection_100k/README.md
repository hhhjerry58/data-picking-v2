# 100k PR 的 A100 提交入口

在已登录 AMLT 的机器 clone `experiment/pr-selection-qwen35` 分支，并从仓库根目录提交。
`.amltconfig` 复用现有 `llm_new` 项目元数据；无需另建提交目录或下载 ZIP。

| 文件 | Job 数 | 用途 |
|---|---:|---|
| [smoke_a100.yaml](smoke_a100.yaml) | 1 | 八卡环境、存储、FSDP 与恢复检查 |
| [train_a100.yaml](train_a100.yaml) | 32 | 全部训练组 |
| [train_batch_1_a100.yaml](train_batch_1_a100.yaml) | 8 | 全文监督的绝对／相对排序组 |
| [train_batch_2_a100.yaml](train_batch_2_a100.yaml) | 8 | 全文监督 Random，seed 42–49 |
| [train_batch_3_a100.yaml](train_batch_3_a100.yaml) | 8 | 答案监督的绝对／相对排序组 |
| [train_batch_4_a100.yaml](train_batch_4_a100.yaml) | 8 | 答案监督 Random，seed 42–49 |

```bash
amlt run yamls/guanheng/pr_selection_100k/smoke_a100.yaml pr100k-smoke-v1 -y --sla Premium
```

匹配的 smoke 通过后，可选择按批提交：

```bash
amlt run yamls/guanheng/pr_selection_100k/train_batch_1_a100.yaml pr100k-batch1-v1 -y --sla Premium
```

或者提交全部任务：

```bash
amlt run yamls/guanheng/pr_selection_100k/train_a100.yaml pr100k-all-v1 -y --sla Premium
```

每个 job 占一个 8×A100 80GB 节点。100 次更新，每步约 256k 输入 tokens，每 5 步保存完整 checkpoint。
YAML 从仓库根目录调用 `scripts/pr_cluster_bootstrap.py`；节点自动校验和载入固定环境及数据。
数值设置、存储权限、smoke 判据和恢复说明见 [PR_CLUSTER.md](../../../docs/PR_CLUSTER.md)。

修改生成器后，在仓库根目录重新生成 YAML：

```bash
.venv-pr/bin/python scripts/make_pr_cluster_yaml.py
```
