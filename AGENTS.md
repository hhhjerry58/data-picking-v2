# 项目范围

- 当前主实验是 Qwen3.5-9B 的 PR 六个评分分布，协议见 `docs/PR_SELECTION.md`，配置见 `configs/pr_selection.yaml`；第一阶段不选数、不继续训练 PR、不计算 T loss。
- 各选数方法放在独立的 `methods/<name>/` 目录，新实验公共实现位于 `src/picking/pr_selection/`。`src/picking/domain_retrieval/` 保留已完成 arXiv pilot 的数值实现及恢复身份。
- Qwen3.5 使用 `.venv-pr`；保留旧 `.venv` 的固定依赖。新增逻辑使用新的 run_id，不改写历史实验。
- 项目讨论与 review 文档使用中文。
- 切换节点前先读 `README.md` 和 `docs/DEVNODE.md`。大数据、权重和 checkpoint 位于 `/mnt/tianzhu_hot/data-picking-v2`。
- 数值代码与训练配置必须兼容 checkpoint 中的指纹。变更数值实现或数据时使用新 `run_id`，不得绕过身份检查或覆盖已审阅结果。
- 用 `DATA_PICKING_GPU_IDS` 明确指定允许使用的 GPU；不停止其他任务。同一 run 同时只允许一个节点写入。
- 新建训练实验必须确认保存 optimizer 状态并启用自动恢复；交接或启动前检查有效配置。

# 实验分析归档

- 除非用户明确要求，`/home/v-guanhechen/experiments/` 只保存实际训练实验的分析。讨论解释、文献调研、baseline 建议、实现方案和源码快照不自动放入该目录；普通对话也不默认创建归档文档。
- 训练实验分析保存到 `/home/v-guanhechen/experiments/`；`experiments` 指这个绝对目录。
- 新分析目录使用 `YYYY-MM-DD_HH-MM-<topic>/`，时间为 `America/Los_Angeles`；同一分析可复用目录。
- 报告、图表、统计、复现 notebook 和分析专用采集 helper 放在同一分析目录。
- 绘图代码放在 `.ipynb` 单元中，并执行、保存内嵌输出；不使用独立 `.py` 绘图脚本。
- 维护该目录的 `README.md` 索引，按时间从旧到新排列；迁移分析时更新链接和复现命令。

# 验证

- 检查范围与改动相称，优先使用针对性 diff、配置/语法检查和相关测试。
- 不默认执行全量文件哈希清单或前后哈希比较；只对具体的完整性或 checkpoint 身份问题使用哈希。
- 测试入口为 `.venv/bin/python -m pytest -q`；GPU 测试必须显式选择空闲卡。
