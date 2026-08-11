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

`--terrain` 可选值与原 A1 流程一致：`slope`、`stair`、`gap`、`climb`、`crawl`、`tilt`。

## 配置位置

- Go2 基础配置：`legged_gym/envs/go2/go2_config.py`
- Go2 WMP/AMP 配置：`legged_gym/envs/go2/go2_amp_config.py`
- 任务注册：`legged_gym/envs/__init__.py`

Go2 的初始高度、默认关节角、PD 增益、机身高度目标、相机位置和碰撞过滤均在上述 Go2 配置中独立定义，修改 Go2 参数不会覆盖 A1 配置。

## 本次验证范围

本次改动只进行静态检查，不在未安装 Isaac Gym 的机器上运行仿真或训练测试。首次在完整 Isaac Gym 环境中运行时，建议先使用较小的 `--num_envs` 和 `--max_iterations` 做显存及资产加载检查。
