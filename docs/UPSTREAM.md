# 上游源码快照

与 `wmp_lab` 同级的 `../wmp_go1/` 来自 https://github.com/haozhang04/wmp_go1 ，固定提交
`65cf1afd13e4792bce7b9def6f44372ecbb3cdf0`。

之前保存在 `/tmp/wmp_go1_source_20261013` 的部分源码已补齐并移至该目录，
缺失文件按上述提交补齐。目录保存完整跟踪文件快照，没有 Git 历史；
同级的 `../wmp_go1.tree.json` 保存 GitHub 原始树清单，可用 Git blob SHA-1 校验各文件。
全部 158 个跟踪文件（66,389,436 字节）已逐项通过校验，结果保存在 `../wmp_go1.snapshot.json`。
上游 LICENSE 和各模块 LICENSE 保持原样。

Go1 原始参考动作在 `../wmp_go1/datasets/mocap_motions_collected/{flat,hop,bound}`。
共 19 条：flat 6 条、hop 6 条、bound 7 条，每条 500 帧。
`../wmp_go1/tool_mujoco/collect_reference_motions.py` 明确使用 Go1 MJCF 与 Go1 策略，
以 50 Hz 保存 61 维 AMP 帧，是仿真采集数据。
该脚本使用 FL/FR/RL/RR 腿顺序，上游内置旧 mocap/Go2 文件也已经是该顺序：
`rsl_rl/datasets/motion_loader.py` 中 PyBullet 到 Isaac Gym 的重排被注释，
三套数据脚部位置的 y 坐标也按左/右/左/右排列。
转换保留该顺序并将四元数统一为 wxyz。
额外传入未重排的 PyBullet 动作时，默认源顺序仍为 FR/FL/RR/RL，
可通过 `--source-leg-order FL FR RL RR` 显式指定已重排的数据。

`scripts/convert_motions.py --dataset go1` 转换全部 Go1 采集片段到包内
`source/wmp_lab/wmp_lab/data/motions_go1`，文件名包含 gait 前缀。
manifest 记录原始文件、源顺序、仓库、提交、帧数、采样率与权重。
`scripts/train.py --robot go1` 默认使用该目录。

上游 Go1 训练配置原本混合了旧 mocap 与 Go2 动作，Go1 采集项被注释；
本项目 Go1 默认数据集改为 Go1 采集数据。这是数据选择变更，原始参考配置保留。

v0.3.0 增加 `scripts/convert_motions.py --dataset upstream`，直接解析该固定提交的
`MOTION_FILES` 列表，转换 2 条旧 mocap trot 与 7 条 Go2 片段到 `data/motions_upstream`。
训练可用 `--motions source/wmp_lab/wmp_lab/data/motions_upstream` 显式选择。
该选项用于固定数据的对照，不改变 Go1/Go2 默认动作集。

v0.5.0 按原 `MOTION_FILES` 顺序保存原 9 条，并通过 `--amp-set upstream` 选择。
本机不运行 Isaac Gym；原源码用于采样数学验证，性能对照全部限定在 IsaacLab 内。
数据契约、审计及四组对照方案见 [AMP_ALIGNMENT.md](AMP_ALIGNMENT.md)。
