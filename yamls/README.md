# 集群任务 YAML

在已登录 AMLT 的机器 clone 本仓库后，从仓库根目录运行 `amlt run`。
任务按 `yamls/guanheng/<experiment>/` 组织，实验超参保留在 `configs/`。

| 实验 | 入口 |
|---|---|
| 100k PR：32 组八卡 A100 对照 | [提交说明](guanheng/pr_selection_100k/README.md) |

不需要下载额外提交包。AMLT 会上传仓库中的启动脚本，节点自动读取共享存储上的固定运行环境、代码快照和数据。
