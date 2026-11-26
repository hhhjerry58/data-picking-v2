# 在 devnode 运行与恢复实验

当前主实验为 [PR 六分布评分](PR_SELECTION.md)，使用 `.venv-pr`；
跨节点 A100 评分及 Windows 提交见 [PR_PARALLEL.md](PR_PARALLEL.md)。
以下内容保留用于 `scripts/domain_retrieval.py` 对应的已完成 pilot 恢复，
协议见 [DOMAIN_RETRIEVAL.md](DOMAIN_RETRIEVAL.md)，参数见 [domain_retrieval.yaml](../configs/domain_retrieval.yaml)。

## 节点准备

使用 Python **3.12**、兼容 CUDA **12.8** 的驱动，以及挂载在 `/mnt/tianzhu_hot` 的持久化存储。
首轮使用 H100 80GB；迁移已有 checkpoint 时使用相同框架版本和数值配置。

```bash
bash scripts/setup_env.sh
source scripts/env.sh
.venv/bin/python scripts/domain_retrieval.py check
```

安装脚本创建独立 `.venv`、安装固定版本的 PyTorch 和项目依赖，并检查训练框架版本。
`scripts/env.sh` 设置缓存、临时目录、线程数和 `PYTHONPATH`；运行目录由 YAML 的
`artifact_root`、`runtime_root` 和 `run_id` 决定。

默认持久化布局：

```text
/mnt/tianzhu_hot/data-picking-v2/
  cache/domain-retrieval/       固定版本的模型和带类别语料
  runs/<run_id>/
    effective_config.json      已冻结的配置
    OWNER.json                 运行所属节点
    data/                      数据划分与候选池
    scores/                    M0/M1 配对 loss
    methods/<name>/            各方法排名、PR 和指标
    selection/<arm>/           继续训练文档清单
    pretraining/<arm-seed>/     完整训练 checkpoint 和日志
    evaluation/                独立测试集逐篇 NLL
    status.json                当前阶段
    latest_report.json         分析归档位置
```

`pretraining/` 是首轮已经使用的产物目录名，里面保存的是**从 M0 继续训练**的状态。
保持这个路径可以直接恢复已保存的 checkpoint。
节点本地 scratch 使用配置的 `runtime_root/<run_id>/`；需预留模型、optimizer 镜像和日志空间。

## 启动与查看状态

```bash
DATA_PICKING_GPU_IDS=0,1,2 .venv/bin/python scripts/domain_retrieval.py run
```

只使用明确允许且空闲的物理 GPU，不停止其他任务。
单卡顺序执行；多卡时先完成 probe 和评分，再分配独立的 arm/seed 任务。
通过运行目录的 `status.json`、`pretraining/*/train.jsonl` 和 `logs/` 查看进度。
`EXPERIMENT_COMPLETED.json` 只在全部训练、评估和报告完成后写入。

默认从当前 blobfuse 挂载配置读取 checkpoint 传输设置，使用分块上传、服务端校验和读回检查。
也可指定外部挂载配置：

```bash
export DATA_PICKING_BLOB_CONFIG=/path/outside/repo/blobfuse2.yaml
```

该传输方式需要配置中的 `azstorage.endpoint`、`container` 和 `sas`；凭据留在仓库外。
其他持久化文件系统可显式设置 `DATA_PICKING_CHECKPOINT_IO=filesystem`。

## 自动恢复与跨节点迁移

同节点重复执行启动命令即可：已完成阶段会复用，训练从最近完整 checkpoint 恢复。
checkpoint 包含模型、optimizer、scheduler、RNG 和 epoch/文档/token 游标。
配置中的 `save_optimizer: true`、`auto_resume: true` 必须保持开启。

跨节点时，先确认旧节点及其子进程已经退出，再转交该 run 的 `OWNER.json` 所属节点。
同一 run 只能由一个节点写入；程序不会仅凭本机查不到旧 PID 就自动接管。
保持持久化路径、配置和数值实现一致；本地 RAM mirror 不存在时，会从持久化 checkpoint 恢复。

`domain_retrieval/data.py`、`domain_retrieval/training.py` 和训练参数受指纹保护。
改变数据或数值实现时使用新 `run_id`，不能修改完成标记或绕过身份校验。

## 结果

报告、逐篇统计、有效配置、训练日志及 notebook 一起保存到
`/home/v-guanhechen/experiments/YYYY-MM-DD_HH-MM-<topic>/`。
目录时间使用 `America/Los_Angeles`，总索引从旧到新排列。
图表在 notebook 的代码单元中生成，执行输出保存在 notebook 内。
