# 100k 候选与 32 组 A100 实验

本地 H100 负责扩池与 M0/M1 前向评分；集群训练由用户自行提交。原八组小预算实验保持暂停。
新实验不恢复已删除的梯度选数方法，也不读取旧集群包的 M1／目标梯度分片。

## 固定协议

- 候选池：100,000 条 `base/8k/git_diff`，从 512 个固定随机分片按原 ID-hash reservoir 采样。
  实际 Qwen token 长度不超过 8192，保留相同去重、监督边界和显式污染排除规则。它不是整个原始仓库的全量数据。
- 排序组：全文／答案 × 绝对／相对下降量 × Top 10%、Top 20%、Top 50%、Bottom 10%，共 16 组。
- Random：两种监督各 8 个 seed（42–49），共 16 组；从完整 100k 池随机取样。相同 seed 的两种目标共用输入顺序。
- 排序组训练 seed 固定为 42；四个百分比都以同一个 100k 池为分母，Top 子集嵌套。
- 每组从原始 Qwen3.5-9B M0 出发，使用一个 8×A100 80GB 节点；不从 M1 或其他训练组续接。
- 100 次 optimizer 更新，目标为每步 **256,000 输入 tokens**，不是答案监督 tokens。
  为保留完整 PR，按最接近预算的整条记录组批，每步偏差最多约 4096 tokens；保存每步与总计的实际输入、监督 tokens。
  若子集用完则重新打乱继续，实际 epoch／唯一 PR 覆盖随子集大小变化。
- 原始文本和 loss mask 与评分一致；不做 PR 拼接后的跨样本注意力，不截断记录。
- 全语言模型参数和 Adam 状态 FP32，矩阵计算 BF16；FSDP FULL_SHARD 数据并行、activation checkpointing、分块输出 head loss。
  每步 loss 按全局实际监督 tokens 归一化；不同 rank 的 microbatch 数用零梯度 dummy 对齐。
- AdamW LR 2e-6，betas=(0.9,0.95)，eps=1e-8，weight decay=0，warmup=2，clip=1。
- 每 5 步保存完整模型、optimizer、scheduler、RNG、step 游标和配置身份，100 步共 20 份；自动恢复。
- T 在 step 0 和 step 100 评测，沿用固定 53 个任务、135 条完整轨迹及语义 mask。中间 checkpoint 先保存，不用 T 选择 checkpoint。

每份完整 checkpoint 约 107 GB，每组 20 份约 2.15 TB，32 组约 **68.8 TB**；默认全部保留。
节点本地只暂存当前一份，确认全部 rank 上传并发布 pointer 后删除本地临时文件。

## 本地准备

```bash
DATA_PICKING_GPU_IDS=0,1,2,3 bash scripts/run_pr_scores.sh configs/pr_selection_100k.yaml
```

`src/picking/pr_selection/` 的模型、mask 和前向评分口径保持固定。分布式训练和集群传输位于
`src/picking/pr_cluster/`，避免集群开发改变正在进行的本地评分身份。

便携环境在旧固定 runtime 上增加 `requirements-cluster.txt` 的 Azure 认证依赖，不修改旧 `.venv` 或 `.venv-pr`。
发布使用仓库外的 `DATA_PICKING_BLOB_CONFIG`，凭据不进入代码、YAML 或压缩包。

```bash
# 可先发布仅供环境/恢复检查的 smoke；使用三个已有 PR，与正式 100k 训练数据分开。
DATA_PICKING_BLOB_CONFIG=/path/to/storage.yaml \
  .venv-pr/bin/python scripts/prepare_pr_cluster.py --smoke-only --publish --staging /tmp/pr-cluster-smoke-v1

# 100k 完成后发布完整输入；可让该进程等待评分完成。
DATA_PICKING_BLOB_CONFIG=/path/to/storage.yaml \
  .venv-pr/bin/python scripts/prepare_pr_cluster.py --wait-for-pool --publish

.venv-pr/bin/python scripts/make_pr_cluster_yaml.py
```

包位于 `data-picking-v2/cluster_bundles/pr-100k-a100-100step-v1-20261126`，smoke 包另加 `-smoke` 后缀。
只有全部输入、代码和 runtime 上传后才发布 `READY.json`。发布包不可覆盖；改变数值代码或 runtime 时使用新版本。

## Clone 仓库后直接提交

任务 YAML 统一放在 `yamls/guanheng/pr_selection_100k/`，目录结构参考已有的 `rl-sopd/yamls/guanheng/`。
仓库的 `.amltconfig` 沿用该仓库已注册的 `llm_new` 项目元数据，只包含项目、存储和 registry 标识；认证仍使用提交机已有登录状态。
在已登录 AMLT 的机器执行：

```bash
git clone --branch experiment/pr-selection-qwen35 https://github.com/hhhjerry58/data-picking-v2.git
cd data-picking-v2
amlt run yamls/guanheng/pr_selection_100k/smoke_a100.yaml pr100k-smoke-v1 -y --sla Premium
```

YAML 的 `code.local_dir` 指向仓库根目录，AMLT 自动上传代码，并执行 `scripts/pr_cluster_bootstrap.py`。
上传规则排除本地虚拟环境、缓存和凭据文件；不需要额外的 `submission/` 目录或手工下载提交 ZIP。

之前提供的提交 ZIP 只是 YAML 和启动脚本的便携副本，适用于提交机没有本仓库的情况，不是数据集或模型。
按当前 clone 工作流，这个 ZIP 不再作为提交入口。

节点自动读取的 `runtime.tar.gz`、`code.tar.gz` 和数据 manifest 是另一回事：它们固定依赖和数值代码，
保证与已验证的 smoke 及 checkpoint 身份一致，且避免每个节点重新安装环境。用户不需要手工下载这些文件。
实验超参仍在 `configs/pr_cluster.yaml`；修改数值代码或超参后应发布新版本，而不是让旧 checkpoint 混用新代码。

新 smoke 会验证八卡 FSDP 的全局梯度归一化、optimizer 恢复、FLA 对照、真实模型的前向/反向、
Blob 上传读回恢复，以及三条长短 T 的前向。smoke 与正式任务输出隔离。

成功依据是控制台 `SMOKE_PASSED.json`，以及 smoke 任务 `STATE.json` 中 `completed: true`。
正式节点还会核对其中的数值配方身份、runtime、模型和 T 输入，拒绝未经匹配 smoke 验证的训练。
该校验独立于候选池成员，允许先检查环境，再等待 100k 评分；它不等于完整 100 步实验已完成。

```bash
# 每批 8 个独立节点，按现有节点容量分批提交。
amlt run yamls/guanheng/pr_selection_100k/train_batch_1_a100.yaml pr100k-batch1-v1 -y --sla Premium
amlt run yamls/guanheng/pr_selection_100k/train_batch_2_a100.yaml pr100k-batch2-v1 -y --sla Premium
amlt run yamls/guanheng/pr_selection_100k/train_batch_3_a100.yaml pr100k-batch3-v1 -y --sla Premium
amlt run yamls/guanheng/pr_selection_100k/train_batch_4_a100.yaml pr100k-batch4-v1 -y --sla Premium
```

也提供 `train_a100.yaml`，包含全部 32 个 job。生成脚本只写配置，不替用户提交。

## 存储和旧 smoke 故障

旧第一次失败是挂载路径读取 M1 分片的 `EIO`。新 bootstrap 对 runtime/code 的挂载读取增加有界重试和完整校验，
模型、PR、T 和 checkpoint 则直接使用 Azure Blob API、分块传输、ETag 和校验和，不依赖 mmap 或大文件复制穿过 FUSE。
集群现在只需要 M0、PR 和 T；M1 留在本地用于评分。

旧第二次失败的 worker 日志明确为跨 H100/A100 数值检查失败，loss 最大差约 0.059；不能当作下载错误或成功。
新实验是在 A100 上训练和评测独立模型，并使用 A100 的 M0 基线；不把旧 H100/A100 分数混合。
新 smoke 检查的是同设备参考与分布式实现的数学一致性，未放宽旧包的失败阈值。
`NCCL_SOCKET_IFNAME='=eth0'` 是合法的精确接口匹配写法，AMLT 的提示本身不是上述错误原因。

Blob API 需要 job managed identity 对容器拥有读写及 lease 权限。也可通过平台 secret 注入
`DATA_PICKING_BLOB_SAS`；不要将 SAS 写进 YAML 或 git。认证失败会在模型加载前明确报错。
本地已经验证了传输和 lease；实际 A100 上的身份权限仍以用户提交的新 smoke 为准。

每个任务在 `runs/<campaign>/<job>/STATE.json` 上持有可续期租约，只有租约持有者能发布最新 checkpoint。
checkpoint 上传使用唯一 attempt 路径，全部 rank 完成后才更新 pointer；抢占后可自动取得过期租约并恢复最后完整 step。
每个 rank 的错误由 torchrun tee 到 AMLT 控制台，失败时还会打印 stderr 尾部，避免只看到笼统的 “worker failed”。

## 观察状态

```bash
DATA_PICKING_BLOB_CONFIG=/path/to/storage.yaml .venv-pr/bin/python scripts/pr_cluster_status.py
```

`STATE.json` 同时记录最新进度、已提交 checkpoint step、M0 基线、最终评测引用和完成状态。
看到 step 100 的进度不等于 checkpoint/评测已经完整发布，应以 `completed: true` 为准。
