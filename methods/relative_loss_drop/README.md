# 相对 loss 下降量

`score(x) = (L(M0, x) - L(M1, x)) / max(L(M0, x), 1e-8)`，降序排列。
两份 loss 都是逐篇平均 token NLL。分母只用微调前 loss；不除以微调后 loss，
不使用 perplexity 变化，也不截断负分。
