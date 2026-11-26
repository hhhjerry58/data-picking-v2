# 选数方法

每种方法独立放在一个目录：`<name>/method.py` 和 `README.md`。
公共流水线按配置中的方法名自动发现目录；新增方法不用修改注册表。

`method.py` 提供 `score(rows, *, seed)`，按输入顺序返回有限实数分数，越大越优先。
`rows` 只包含 `id`、`loss_before`、`loss_after`、`prediction_tokens`；类别和
ground truth 不传入方法。两份 loss 使用完全相同的文本、tokenizer、上下文和有效 token mask。
相同分数用与类别无关的固定 ID hash 打破排序并列；PR 和 AP 按共同阈值合并并列分数。

- [绝对下降量](absolute_loss_drop/README.md)
- [相对下降量](relative_loss_drop/README.md)
- [随机对照](random/README.md)
- [GradDot](grad_dot/README.md)：新 PR 实验的全语言模型梯度内积。

PR 评分阶段复用绝对／相对公式，并为 GradDot 提供独立的准备和逐样本评分接口。
公共可求导引擎负责 token loss 与梯度计算，GradDot 方法负责其目标梯度及内积评分。
`full_text`、`answer_only` 是监督 mask 配置，不复制成两套方法目录。
当前阶段只输出三个方法 × 两个监督目标的分数，不自动选 top-K。

方法输出位于新运行的 `methods/<name>/`，包含排名、PR 阈值和指标。
`oracle_target`、`other` 是使用已知类别的继续训练对照，不是可用标签的检索方法。
