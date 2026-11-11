# Go2 环境数量与 motion 数据验证

## 2026-10-15：v3 完整上游网络复测

| 环境数 | 真实相机数 | 64 帧序列、50 次深度/10 次 WM 更新 | 整卡峰值 GiB | 完整更新耗时 s |
|---:|---:|---|---:|---:|
| 512 | 128 | 通过 | 7.71 | 4.90 |
| 4096 | 1024 | 通过 | 11.12 | 10.48 |

记录为 `logs/benchmarks/v3_go2_{512,4096}.json`。使用新版完整 Encoder/Decoder/RSSM，
保留真实 CUDA/全部默认地形比例及物理随机化。为在固定步测试中稳定覆盖 64 帧更新，
规模脚本显式使用浮点环形缓存并提前启动 WM；它验证计算/显存链路，
不验证正式训练的完成 episode 采样，也不是收敛结果。正式短训另使用 `replay_mode=episodes`。

旧版数据记录如下，不能直接作为 v3 的显存预算或世界模型 loss 参考。

## 2026-10-14：v2 历史记录

2026-10-14，本机 RTX 5070 12 GB（驱动报告 12,227 MiB）、32 GB 系统内存，
IsaacLab 2.3.2 / Isaac Sim 5.1 / `env_isaaclab`，Warp 深度后端。
使用真实 CUDA 模拟器，经共享 IsaacLab smoke wrapper 启动。

## 环境规模

保留默认地形比例、16 行 × 40 列、物理随机化、完整网络、24 步 rollout，
PPO 5 epoch / 4 minibatch、AMP replay 1,000,000、世界模型 replay 300,000。
短测将深度/世界模型更新次数各设为 1，并提前触发更新；深度 batch 仍为 1024。
首次更新时实际可用世界模型序列只有 5 帧，因此另做预热后的完整更新。

| 环境数 | 真实相机数 | 短更新结果 | 整卡显存峰值 GiB | 短更新耗时 s |
|---:|---:|---|---:|---:|
| 64 | 64 | 通过 | 4.83 | 1.72 |
| 256 | 256 | 通过 | 5.14 | 1.95 |
| 1024 | 1024 | 通过 | 6.90 | 2.98 |
| 2048 | 1024 | 通过 | 7.59 | 4.49 |
| 4096 | 1024 | 通过 | 10.15 | 7.34 |

整卡峰值由 CUDA `mem_get_info()` 每 0.2 秒采样，包含桌面及模拟器/PhysX/Warp 分配；
JSON 另记录 PyTorch 分配与保留峰值。时间包含采集与更新，不含场景启动。

4096 环境另做预热：累计 15 个迭代、1,474,560 环境步；缓存预热到 68 帧后，
最后一次更新使用完整的 64 帧 × 16 条世界模型序列、50 次深度训练、10 次世界模型训练。
仅将世界模型启动门槛提前，其他最终学习参数与默认完整配置一致。
最终完整更新通过，耗时 **10.09 秒**，整卡峰值 **10.57 GiB**。
AMP discriminator loss = 0.2000、depth loss = 0.0321、world loss = 0.7747，指标均有限。
成功保存 `logs/benchmarks/go2_4096_warm.pt`，该文件是短时验证 checkpoint。

当时验证默认的 4096 环境 / 1024 相机可运行；较保守的起步配置为 2048 / 1024。
这些数字是已测试范围，不是最大环境数量或长期性能保证。

## Motion 数据

训练读取 `source/wmp_lab/wmp_lab/data/motions_go2`，源数据为同级
`../wmp_go1/datasets/go2_motion`，固定提交 `65cf1afd13e4792bce7b9def6f44372ecbb3cdf0`。
17 条片段共 **3740 帧 / 50 Hz**，全部参与验证：

- 原始 JSON 与转换后关节位置/速度逐项匹配，顺序与模拟器 FL/FR/RL/RR 一致。
- 全部数组有限、形状匹配、四元数归一化；关节位置超出 Go2 硬限制的样本为 0。
- GPU 抽取 8192 对 30 维 AMP 转移，覆盖全部 17 条序列，当前与下一时刻状态均有限。
- 运动片段的记录关节速度与位置有限差分相关系数为 0.959–0.998；
  站立片段速度很小，RMSE 约 0.006 rad/s。
- 将参考关节角放入包内 Go2 MuJoCo 模型做足端正运动学检查，
  与原始数据足坐标的坐标分量 RMSE 为 12.9–14.1 mm；未做运动学重定向。
  当前 30 维 AMP 输入使用关节位置/速度和机身线/角速度，足坐标不参与判别器。
- 真实 PPO/AMP 更新完成，判别器 loss、梯度惩罚和 expert/policy score 均有限。

结论是这批数据可以作为当前 AMP 训练输入；行走/越障质量仍需长期训练评估。

## 复测与文件

```bash
cd /home/re/re_lab/wmp/wmp_lab
env WMP_BENCH_ENVS=4096 WMP_BENCH_WARM=1 \
  WMP_BENCH_OUTPUT="$PWD/logs/benchmarks/go2_4096_warm.json" \
  ISAACLAB_SMOKE_SCRIPT="$PWD/scripts/benchmark_go2.py" \
  ISAACLAB_SMOKE_STEPS=24 ISAACLAB_SMOKE_TIMEOUT=360 \
  /home/re/IsaacLab/tools/run_codex_isaaclab_smoke.sh

# 正常训练入口；默认使用 Go2 motion 数据
./bash/leggedskill.sh -t --robot go2 --num-envs 4096 --camera-envs 1024 --iterations 20000
```

`scripts/benchmark_go2.py` 提供可重复的真实模拟器检查，`WMP_BENCH_ROWS` 默认 16，
`WMP_BENCH_CAMERAS` 默认 `min(envs, 1024)`。省略 `WMP_BENCH_WARM` 为短测。
各档结果、完整更新结果、motion 审核及原始日志位于 `logs/benchmarks/`。
项目检查：pytest 14 项通过，1 个地形模块需 Isaac Sim 启动；Ruff 通过。
