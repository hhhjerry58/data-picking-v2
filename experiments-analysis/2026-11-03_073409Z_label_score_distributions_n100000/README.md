标注集分数分布
================

打开 [label_score_distributions.ipynb](label_score_distributions.ipynb) 查看已执行的图表与统计。

包含：中央 99% 直方图、全范围对数直方图、文档/token 加权 CDF、train/dev/test 对照和联合分布。

- 数据：完整的 100,000 篇标注集，来源与校验和见 [manifest.json](manifest.json)。
- relative delta 在图表中显示为百分比，快照保留原始比例。
- 主体视图明确标注显示区间，全范围图保留所有尾部数据。
- 主要图：[主体分布](figures/01_central_distributions.png) · [全范围](figures/02_full_range_log_histograms.png)。
- 表格：[总体统计](summary.csv) · [分位数](quantiles.csv) · [划分统计](split_summary.csv)。

本目录为独立分析快照，未修改主实验的评分、配置、状态或其他分析目录。
