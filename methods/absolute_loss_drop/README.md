# 绝对 loss 下降量

`score(x) = L(M0, x) - L(M1, x)`，按分数降序排列。
`L` 是逐篇有效预测 token 的平均交叉熵（nats/token），不是整篇 loss 的总和。
保留负分，完整候选池都进入 PR 评估。
