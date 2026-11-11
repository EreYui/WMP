# 验证记录

2026-10-22 / v0.6.0 参考环境迁移：50项离线测试通过，另5项地形测试在模拟器内补跑通过，Ruff通过。
修复Go1资产质量、足端索引、gamma/软std、高度与边缘、原地形、reset/命令/timeout、感知与导出时序。
Go1/Go2通过完整网络和18类地形的联合更新、优化器恢复及导出；实际训练CLI跨进程恢复通过，
两种新策略各通过10步MuJoCo。Go1的4100/1100、Go2的2048/550完整更新通过，
整卡峰值分别10.79/10.86GiB；Go2的4100/1100超过本机显存。
规模检查额外修复PhysX碰撞栈/patch溢出和推理临时显存峰值。
细节与规模限制见[REFERENCE_ALIGNMENT_RESULT.md](REFERENCE_ALIGNMENT_RESULT.md)。本轮未启动正式长训练。

以下为历史版本记录，默认配置以当前README为准。

2026-10-15 / v0.5.0 固定 AMP：31 项测试通过，1 项需 Isaac Sim 的地形模块跳过，Ruff 通过。
原采样函数差分、文件移动/变更、RNG 与 checkpoint 契约均验证。
Go1/Go2 共用 9 条数据通过真实 CUDA smoke；4096 环境/1024 相机加 200 万对样本的完整更新通过，
峰值 11.46 GiB。仅在 IsaacLab 内验证，不运行或宣称 Isaac Gym 性能基线，详见 [AMP_ALIGNMENT.md](AMP_ALIGNMENT.md)。

2026-10-15 / v0.4.0 控制链对齐：27 项离线测试通过，1 项需 Isaac Sim 的地形模块跳过，Ruff 通过。
Go1/Go2 均通过真实 CUDA 的逐关节限制、原版 PD 力矩差分、子步延迟及完整学习/导出 smoke；
MuJoCo 两种机器人通过 10 个策略控制步。旧第 200 轮模型重放 20 秒，位置/动作/关节/力矩与历史记录差异均为 0。
旧 v2 第 9500 轮模型导出误差为 0。详见 [CONTROL_ALIGNMENT.md](CONTROL_ALIGNMENT.md)，
结构化记录位于 `logs/validation/control_alignment_20261015/summary.json`。训练保持暂停。

2026-10-15 / v0.3.0：23 项离线测试通过，1 项需要 Isaac Sim 启动的地形模块跳过；Ruff 通过。
差分测试实际执行固定上游源码，对照 Actor 前向与熵梯度、PPO/AMP 全部参数的一步更新、
世界模型完整损失与所有参数梯度、深度预测器布局和特殊速度奖励。
新增完成 episode 隔离/替换、长度加权、稀疏相机及浮点数据保持测试。

Go1/Go2 均通过真实 CUDA 的完整网络、18 类地形、联合更新、checkpoint 与 TorchScript smoke。
Go2 完整 64 帧序列/默认更新次数规模复测：512/128 相机及 4096/1024 相机通过，
整卡峰值分别 7.71/11.12 GiB。正式短训与回放结果见 [UPSTREAM_ALIGNMENT_RESULT.md](UPSTREAM_ALIGNMENT_RESULT.md)。
旧 v2 第 9500 轮 checkpoint 读取与导出验证通过，数值误差为 0；原长训练保持暂停。

以下为 v2 历史记录。

环境：2026-10-13，IsaacLab 2.3.2 / Isaac Sim 5.1 / env_isaaclab，RTX 5070 12 GB。

## 已完成

- 13 项离线测试通过，1 个要求 Isaac Sim 导入的地形测试模块在普通 pytest 中跳过。
- Ruff 检查通过。
- Go1、Go2 固定步 CUDA smoke：传感器、真实/预测深度、reset、连续训练、checkpoint、TorchScript。
- Go1、Go2 全部18类地形及物理随机化场景 smoke：18环境、2级地形、8个真实相机。
- 完整网络配置 smoke：RSSM512、32×32分类状态、Actor256/128/64、AMP1024/512、PPO5epoch/4minibatch。
  只缩短 rollout/序列与世界模型更新次数，确认完整网络链路可以更新和导出。
- RTX 轴向深度、RGB渲染和MP4录制 smoke，检查相机世界位姿跟随机器人。
- 同内参/同位姿 Go1 stool 场景：Warp/RTX 深度 MAE约4.08e-7 m，最大差异约7.87e-6 m。
  仅代表这一静态采样场景，不推广为任意场景精度指标。
- 训练CLI：Go1两轮更新、每轮保存、JSONL/TensorBoard写入。
- 恢复CLI：从周期 `model_1.pt` 恢复迭代1，继续到迭代2。
- 播放CLI：10步推理，导出策略/JSON元数据，生成MP4。
- MuJoCo Go1/Go2：10步视觉推理、动作采集、采集格式转换。
- 上游快照位于同级目录 `../wmp_go1`：158个跟踪文件逐项通过Git blob SHA-1校验。
- Go1采集数据：19条50Hz/500帧片段，保留FL/FR/RL/RR顺序及步态文件名。
- 数据更新后：pytest 14项通过，1项需Isaac Sim启动；Ruff通过。
  Go1、Go2均通过共享IsaacLab wrapper的真实CUDA smoke，使用各自默认动作数据完成PPO/AMP/RSSM更新。
- 地形等级图、日志压缩/删除预览、深度比较工具运行验证。
- 2026-10-14 Go2规模测试：64/256/1024/2048/4096环境均通过；
  4096环境/1024相机完成64帧世界模型序列、50次深度与10次世界模型的完整更新。
  整卡峰值10.57 GiB，17条motion共3740帧通过采样与实际AMP更新；详见 [GO2_BENCHMARK.md](GO2_BENCHMARK.md)。
- 2026-10-15训练暂停回放：9500轮策略的左后膝均值/方差退化，动作长期饱和，
  确定性、采样、限制std及移除障碍的对照均未达到gap/climb课程升级条件。
  训练规模验证不代表长期策略质量；详见 [GO2_TRAINING_DIAGNOSIS.md](GO2_TRAINING_DIAGNOSIS.md)。

## 验证边界

这些测试不证明两万轮收敛、越障成功率、长期训练吞吐量或实机安全性。
没有运行实际 ROS2/Gazebo或机器人驱动；ROS2节点已实现数据桥，外部驱动需按部署契约接入。
以下数据选择是v2历史配置：Go1采用上游Go1 MuJoCo采集动作，Go2采用上游Go2动作；
v0.6新实验默认共用原9条AMP。转换仅统一关节顺序与四元数格式，不做运动学重定向。

Ray-cast 与 RTX 深度比较应使用一致的轴向深度、相机标定和机器人姿态。
RTX渲染包含机器人自身；ray-cast默认只包含地形。
不将不同场景/位姿的误差比较当成后端精度证明。
