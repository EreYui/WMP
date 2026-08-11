# WMP Go2 IsaacLab 扩展

这是一个独立于 IsaacLab 主仓库的外部扩展，面向 IsaacLab 2.3.2、Isaac Sim
5.1 和 Python 3.11。它把 WMP 的 285 维观测契约、AMP-PPO、Dreamer 风格
RSSM 世界模型、深度预测器以及联合训练/推理链路迁移到 `wmp_go2` 命名空间，
不会覆盖环境内的 RSL-RL 3.1.2。

`WMP/` 和 `legged_mpc_amp/` 是只读上游参考；扩展代码位于
`source/wmp_go2/`，IsaacLab 主仓库不需要修改。

## 任务

- `Isaac-WMP-Go2-Rough-Direct-v0`：4096 环境，论文规模网络、混合地形、
  domain randomization 和 curriculum。
- `Isaac-WMP-Go2-Rough-Direct-Smoke-v0`：4 环境，其中前 2 个逻辑环境使用
  真实 ray-cast 深度，另外 2 个使用深度预测器；关闭高成本随机化和课程。

仿真步长是 0.005 s，decimation 为 4，策略周期为 0.02 s。关节数据契约固定为：

```text
[FL, FR, RL, RR] × [hip, thigh, calf]
```

285 维观测按 `[privileged 53, proprioception 33, previous_action 12,
body_height 187]` 拼接；历史输入是去掉 3 维速度命令后的 `42 × 5 = 210`；
AMP 状态是 `[joint_pos 12, root_lin_vel_b 3, root_ang_vel_b 3,
joint_vel 12] = 30`。世界模型固定使用 `prop[33]`、归一化
`depth[64,64,1]`、5 步动作历史 `[60]`、reward 和 `is_first`。

## 安装与动作转换

以下命令都使用 IsaacLab 的 `env_isaaclab`，不要使用 base Python：

```bash
conda run -n env_isaaclab pip install -e source/wmp_go2
conda run -n env_isaaclab python scripts/convert_motions.py
```

默认转换 `WMP/datasets/mocap_motions` 的四段 501 帧 JSON。输出是一段动作一个
`.npz`，以及保留序列边界的 `manifest.json`。转换器把 WMP 的 `xyzw` 转成
IsaacLab 的 `wxyz`，并把源 `[FR, FL, RR, RL]` 显式重排为项目关节顺序。
这些是 A1 动作，只用于验证 AMP 接口与训练链路；转换不等同于 A1→Go2
运动学重定向，也不代表 Go2 训练质量。

同一转换器可接收 `legged_mpc_amp` 的 CSV：

```bash
conda run -n env_isaaclab python scripts/convert_motions.py \
  legged_mpc_amp/amp_data/*.csv --source-fps 50
```

如果未来没有动作数据，需要独立进入 `legged_mpc_amp` 按其 ROS/Gazebo 文档
采集，例如运行 `bash scripts/collect_amp_data.sh`；本扩展不会复制、启动或管理
ROS、NMPC、WBC 或 Gazebo。

## 训练、恢复和推理

论文规模训练（显存需求远高于单张 RTX 5070）：

```bash
/home/re/IsaacLab/isaaclab.sh -p scripts/train.py \
  --headless --task Isaac-WMP-Go2-Rough-Direct-v0 --iterations 20000
```

RTX 5070 上的低显存正确性运行：

```bash
/home/re/IsaacLab/isaaclab.sh -p scripts/train.py \
  --headless --smoke --iterations 1 --output logs/wmp_go2_smoke.pt
```

恢复训练时 replay buffer 按设计不进入 checkpoint，会重新预热：

```bash
/home/re/IsaacLab/isaaclab.sh -p scripts/train.py \
  --headless --smoke --checkpoint logs/wmp_go2_smoke.pt --iterations 1
```

确定性 checkpoint 推理：

```bash
/home/re/IsaacLab/isaaclab.sh -p scripts/play.py \
  --checkpoint logs/wmp_go2_smoke.pt --task Isaac-WMP-Go2-Rough-Direct-Smoke-v0
```

新的 `.pt` checkpoint 包含格式版本、配置、策略/价值网络、世界模型、深度
预测器、AMP 判别器、优化器、normalizer 和迭代数；不承诺兼容旧 WMP 权重键。

## 验证

```bash
conda run -n env_isaaclab pytest -q
conda run -n env_isaaclab ruff check .

env ISAACLAB_SMOKE_SCRIPT=$PWD/scripts/wmp_go2_smoke.py \
  ISAACLAB_SMOKE_STEPS=4 ISAACLAB_SMOKE_TIMEOUT=300 \
  /home/re/IsaacLab/tools/run_codex_isaaclab_smoke.sh
```

真实 smoke 必须在能看到 `/dev/nvidia*` 的环境中运行。它验证离线 Go2 USD、
地形、两个高度 ray-caster、接触传感器、深度相机、reset/step、真实/预测深度
两条分支、一次 AMP-PPO 更新、一次世界模型与深度更新以及 checkpoint 往返。
短烟测不用于判断 reward、步态稳定性、吞吐量、策略收敛或论文性能复现。

## 资产与许可

项目代码采用 Apache-2.0。包内 Go2 USD 保留 Unitree Robotics 的 BSD-3-Clause
声明；WMP/Dreamer/Parkour/legged-gym 的适配来源及 MIT/BSD/Apache 声明见
`THIRD_PARTY_NOTICES.md` 和资产目录内的许可证文件。
