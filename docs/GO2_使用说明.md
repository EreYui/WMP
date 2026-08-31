# Go2 离线资产与训练使用说明

## 变更概览

本项目已内置 Unitree Go2 的 URDF 和 URDF 实际引用的 DAE 网格，训练和可视化时不会再从网络、Nucleus 或其他外部资产目录下载 Go2 模型。

新增任务：

- `go2_amp`：使用 Go2 资产运行本项目的 WMP + AMP 训练和可视化流程。
- `go2`：Go2 的基础粗糙地形配置，供标准 Legged Gym runner 或二次开发使用。

Go2 URDF 的关节顺序为 `FL, FR, RL, RR`，原动作数据的顺序为 `FR, FL, RR, RL`。`go2_amp` 会自动重排 AMP 动作数据；该行为仅对 Go2 开启，不影响原有 `a1` 和 `a1_amp` 任务。

## 环境安装

Python、PyTorch、Isaac Gym Preview 3 及其他 Python 依赖的安装方式没有改变，仍按项目根目录的 `README.md` 执行。

Go2 不需要额外安装资产，也不需要执行下载脚本。完整离线资产位于：

```text
resources/robots/go2/
├── urdf/go2.urdf
└── dae/*.dae
```

资产许可见 `resources/robots/go2/go2_license.txt`。资产来自 Unitree Go2 开源模型，按 BSD 3-Clause License 随项目分发。

## 启动 Go2 训练

在项目根目录执行：

```bash
python legged_gym/scripts/train.py \
  --task=go2_amp \
  --headless \
  --sim_device=cuda:0
```

如需降低并行环境数量或先进行短训练，可增加参数：

```bash
python legged_gym/scripts/train.py \
  --task=go2_amp \
  --headless \
  --sim_device=cuda:0 \
  --num_envs=512 \
  --max_iterations=100
```

默认日志及模型目录为：

```text
logs/go2_amp_example/WMP/
```

## Go2 可视化

先确保上述目录中已有训练检查点，再在项目根目录执行：

```bash
python legged_gym/scripts/play.py \
  --task=go2_amp \
  --sim_device=cuda:0 \
  --terrain=climb
```

`--terrain` 可选值为：`slope`、`stair`、`gap`、`climb`、`crawl`、`tilt`、
`plum_piles`。

## 障碍课程与相机修正

- Go2 深度相机位置改为官方 `front_camera_joint` 相对 base 的
  `[0.32715, -0.00003, 0.04297]`。
- 地形课程改为 16 行，并增加梅花桩：桩径随难度从 0.48 m 减小到 0.26 m，
  桩间隙从 0.12 m 增大到 0.30 m，坑深从 0.25 m 增大到 0.60 m。
- 最高等级不再随机回收；升级要求沿 `+X` 穿越 3 m，梅花桩要求 3.4 m 以确认
  进入落地区，且横向偏移小于 1 m。
  日志会分别输出 gap、tilt、crawl 和 plum-piles 的平均等级。
- gap/梅花桩使用 0.45--0.8 m/s 的前进命令，降低跳跃时的竖直速度惩罚并增加
  有符号前进奖励；梅花桩启用落脚边缘惩罚。tilt 通道改为 0.42--0.34 m，给
  Go2 保留可学习的碰撞余量。
- 相机环境优先分配给 tilt、crawl、梅花桩，并修正 depth buffer reset 使用错误
  环境索引的问题。

上述配置会改变旧 run 的训练分布。建议创建新的日志目录从头训练，并观察：
`terrain_level_gap`、`terrain_level_tilt`、`terrain_level_crawl`、
`terrain_level_plum_piles`，不要只根据总 `terrain_level` 判断是否卡住。

## 配置位置

- Go2 基础配置：`legged_gym/envs/go2/go2_config.py`
- Go2 WMP/AMP 配置：`legged_gym/envs/go2/go2_amp_config.py`
- 任务注册：`legged_gym/envs/__init__.py`

Go2 的初始高度、默认关节角、PD 增益、机身高度目标、相机位置和碰撞过滤均在上述 Go2 配置中独立定义，修改 Go2 参数不会覆盖 A1 配置。

## 验证建议

首次训练前建议先使用较小的 `--num_envs` 和 `--max_iterations` 检查资产、相机
和显存，再开始完整的新 run。短时仿真只能验证执行链路，不能证明策略会收敛或
宽 gap/tilt 已经学会。
