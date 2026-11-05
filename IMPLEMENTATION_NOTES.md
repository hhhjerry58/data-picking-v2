执行说明与实现核验
==================

实验范围保持为用户批准的 100,000 篇正式标注、两种 loss 变化标签、两个 fastText、
同一约 16B-token 候选池、三组 160M / 3.2B 预训练。

- 现有评分模型和数学 SFT step-64 权重保持固定。评分采用 FP32 / SDPA math，
  在 H100 80GB 上将计算 batch 从起点 4 调至 16；正式写入评分前检查实际 batch 与
  单样本的 NLL 一致性，容差仍为 2e-5。输入窗口、mask 和两种指标定义不变。
  正式采用按长度排序的实际 batch；少于 128 tokens 的窗口单条评分，避免把极短文本
  填充到 2,048 tokens 带来的数值偏移。三个 reference 分片的最大实际 batch/单条差为
  2.83e-6、2.88e-6、3.12e-6，均通过原容差。初次把长短窗口混在一起的检查失败记录保留，
  未用于正式分数；当前结果位于 `scores/fp32_sorted_short_single_v2/`。
- 预训练 micro batch 从起点 8 调至 32，梯度累积为 4；全局 batch 仍为 128。
  同一设置应用于所有实验臂。预训练启用确定性算法和编译后的 causal SDPA Flash，
  与常规 SDPA math 的因果行为/FP32 logits 对照通过；评分阶段不使用此快捷路径。
- 本地 PyTorch 编译器会让 GPT-NeoX 两条并行 residual 分支的 bias 梯度共享缓冲区。
  直接进行累积和原地裁剪会修改两次同一缓冲区。现已在每次 backward 后分离共享的
  梯度缓冲区，再进行后续累积和裁剪。相应回归检查覆盖了独立累积与裁剪。
- 真实 160M、2,048-token 输入的连续/中断恢复对照已通过：三步结束时模型与
  optimizer 最大差均为 0，scheduler、RNG 和 token/data 游标恢复。完整记录位于
  运行目录的 `preflight/training_resume_verification.json`。
- fastText 源码初始化遗漏与完整训练状态适配见
  [VENDOR_PATCHES.md](vendor/fasttext/VENDOR_PATCHES.md)。恢复后的模型和状态文件与
  连续训练逐字节一致，并与同一固定后端的 CLI 输出一致。
- 所有大产物持久化到批准的挂载。重复读取使用检查过 SHA-256 的 POSIX RAM 副本，
  以避免 blobfuse 重复下载。checkpoint 复用旧项目已验证的同一 blob 分块上传实现，
  检查服务端 block MD5、提交顺序、大小及 ETag；每个 run 的首次保存做完整读回。
- 数据读取遇到挂载等待后重启协调器。原始处理逻辑完整冻结在
  [源码快照](/home/v-guanhechen/data-picking-v2/src/picking/_data_processing_v1.py)，
  内容哈希与已完成分片一致；新协调器读取相同的已校验分片，不改变过滤、去重或抽样规则。
  最终索引使用显式 `uint64` 保存文档采样哈希。初次自动推断有符号整数发生溢出后，
  已从全部校验过的分片与完整去重决策日志恢复相同的保留集合；文档数和 token 总数逐项一致。
- Qwen 少于两个 tokens 的正文无法计算既定的文档 loss。若正式标注集合出现此类
  文档，在任何模型 forward 前，按固定哈希从与 P/通用评估隔离的 unused 储备中替补，
  保持 100,000 篇；排除、替补和最终划分写入 `label_validity.json` 与
  `final_label_manifest.parquet`，候选池 P 保持固定。

以上检查属于实现完整性与数值正确性检查，不计入正式三组预训练结果。

2026-11-05 UTC 的运行恢复记录：

- 发现 2026-11-03T12:07 UTC 后流水线停止。控制进程写 `pipeline_progress.json`
  遇到挂载 EIO 后退出，watchdog 重启时写 `watchdog.json` 也遇到 EIO 并退出，旧心跳因此过期。
- 已完成 seed 42 三组训练及评估、seed 43 的 Random / FT-delta 训练；FT-relative
  的最后日志约为 3.142B tokens，可恢复的完整 checkpoint 是 step 11,500 / 3.013184B tokens。
- 已复核该 checkpoint 的 SHA-256、model / optimizer / scheduler / RNG / 数据游标和配置指纹。
  数值训练代码与训练配置保持原样。
- 控制状态改为 POSIX RAM 与工作区本地磁盘的原子记录，由独立进程镜像到挂载；
  加入挂载 EIO、RAM 状态目录不可用与完成阶段跳过的回归测试。新的控制进程和 watchdog
  已重新启动，等待 GPU 可用后继续剩余任务。
- 本次检查时四张 GPU 均由已有 vLLM 服务占用，没有停止或改动这些服务。
  持久化细节位于运行目录 `recovery_20261105.json`。
