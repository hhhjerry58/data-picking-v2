# 本地并行 PR 前向评分

每张允许使用的 GPU 启动一个模型副本。四张 H100 对应四个独立 worker，
每个负责候选行号 `worker_id, worker_id+4, ...`，各 5,000 条。
每个 worker 先计算 M0，再加载已完成的 M1；每次 forward 同时得到两个监督区域的 loss。

当前采用本地评分，不保留已经过时的多节点启动器。历史部署包和实验产物仍在共享存储。

## 启动与观察

```bash
DATA_PICKING_GPU_IDS=0,1,2,3 bash scripts/run_pr_scores.sh configs/pr_selection.yaml
```

`scoring.attention_backend: default` 保留原数值口径；`flash` 必须使用新 run_id。
配置改变后不可复用旧评分缓存。模型权重及候选数据从共享存储读取，本地 RAM mirror 由一个协调进程准备。

在 `/mnt/tianzhu_hot/data-picking-v2/runs/<run_id>/` 查看：

- `pipeline_status.json` / `status.json`：整体与当前阶段。
- `workers/00/progress.json`：当前模型、完成记录数及最近一条耗时。
- `workers/00/worker.log`：逐条计时及异常，其余 worker 类似。
- `workers/*/parity.json`：各 worker 对共同长短样例的 M0/M1 loss。
- `SCORING_COMPLETED.json`：全量严格配对完成。
- `latest_report.json`：实际四分布分析位置。

## 恢复和身份

相同配置、相同 worker 数重复启动会跳过已保存记录。每条在本地 journal 保存，每 32 条持久化；
正常异常退出也会尽力上传已经完整写入的记录。节点丢失最多重算最近未上传的小段。
分片计划、源码/依赖、checkpoint、候选文件均参与身份校验。合并检查每个分片、每个目标的 ID 覆盖，
拒绝重复、缺失、目标 token 数不一致或非有限 loss。

CLI 对 GPU 和 run 加锁；只使用显式指定且空闲的 GPU，不停止别人的任务。
同一 run 的 OWNER 阻止其他节点自动接管，迁移前必须先确认旧进程结束。

历史 M1 的导入验证使用原 run 的 `source/` 与 `source_identity.json`，不会让新实现冒充旧实现。
完整文件检查只在首次导入或 mirror 变化时执行，worker 复用已经验证的只读 mmap，并核对 payload 指纹。
已有 checkpoint 包含 optimizer 和自动恢复状态；候选评分无需重新执行 Q 微调。

当前项目的实际报告集中在 `/home/v-guanhechen/experiments/data-picking/`。
完整 20,000 条候选的四分布统计见 [全量报告](/home/v-guanhechen/experiments/data-picking/2026-11-26_02-02-pr-score-distributions-20k/REPORT.md)；旧前缀预览归档已替换。
