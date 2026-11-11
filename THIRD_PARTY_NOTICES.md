# 第三方来源与许可

- 本项目参考 [haozhang04/wmp_go1](https://github.com/haozhang04/wmp_go1)，提交
  `65cf1afd13e4792bce7b9def6f44372ecbb3cdf0`，上游整体 Apache-2.0；
  WMP 原始算法来自 [World Model-based Perception](https://wmp-loco.github.io/)。
  当前代码为 IsaacLab 适配，学习核心的复用范围如下。完整上游源码（含 RSL-RL/Dreamer）另外保存在
  同级目录 `../wmp_go1` 作为迁移参考，当前扩展不导入该目录。
- v0.3 起，世界模型直接复用上述提交的 Dreamer networks/tools，保存于
  `learning/upstream_dreamer`，保留 MIT 文件头与 LICENSE。仅将未指定设备的 MLP 默认构造设备改为 CPU，
  便于离线测试；实际计算随模块及输入迁移设备，不修改网络数学定义。
- Actor/PPO/AMP 与奖励适配参考上游 RSL-RL/LeggedGym，保留 NVIDIA、ETH 与 Bytedance 声明；
  BSD-3-Clause 文本位于 `learning/UPSTREAM_RSL_LICENSE.txt`。
- 原 v0.2 学习实现归档于 `learning/legacy_v2`，用于原 checkpoint 回放；新版本不续训该优化器状态。
- v0.4 控制适配使用外部 IsaacLab 的 IdealPDActuator（BSD-3-Clause），保留历史 DC 控制配置用于回放。
- v0.7 的 `sensors/upstream_wmp` 来自上述固定提交的
  `legged_gym/sensors/warp/warp_cam.py` 与 `warp_camera_kernels.py`。保留源文件原文，
  仅修改 WarpCam 的相对导入；kernel 逐字节一致，出处见同目录 `PROVENANCE.json`。
  新增的 `wmp_warp_camera.py` 负责 IsaacLab 位姿、选中相机缓冲和 CUDA 流同步适配。
- IsaacLab API 与环境模式：BSD-3-Clause。IsaacLab 是外部依赖，不复制主仓库。
- v0.6 的 `reference/terrain.py` 来自上述固定提交的 LeggedGym 地形实现，保留
  NVIDIA、ETH 与 Bytedance 文件头和完整 BSD-3-Clause 条款；仅调整依赖入口及 Go2 通道宽度参数。
  `reference/meshes.py` 来自同一提交，保留 Ziwen Zhuang 的完整 MIT 许可。
  `reference/primitives.py` 是本项目独立实现的 NumPy 基础地形算法；测试读取本机 Isaac Gym
  安装中的函数进行差分验证，项目不分发该安装中的专有源码。
- Go1 URDF/STL/MJCF 来自上述 WMP Go1 仓库，机器人原始描述为 Unitree Robotics BSD-3-Clause；
  原始及标准化 URDF 同时保留。Go1 描述的 BSD 声明位于
  `source/wmp_lab/wmp_lab/assets/robots/go1/LICENSE.txt`。
- Go2 USD 来源于原工作区 `legged_mpc_amp`，保留资产目录内 Unitree 与源项目的 BSD 声明。
- Go2 MJCF/OBJ 来自本机 Unitree 官方 `unitree_mujoco`，保留
  `source/wmp_lab/wmp_lab/assets/robots/go2/MUJOCO_LICENSE.txt`。
- 包内 A1/Go1/Go2 参考动作来自 WMP / WMP Go1 仓库，原始运动学和出处保留；
  Go1 的 flat/hop/bound 数据由上游 Go1 模型及策略在 MuJoCo 中采集，原始数据与采集脚本一并保留。
  转换仅重排关节与四元数格式，不构成运动学重定向。
- TensorBoard 操作说明参考上游 `doc/TENSORBOARD_REMOTE.md`。

项目代码采用 Apache-2.0，机器人资产保留各自许可。ROS2/Gazebo/电机驱动未复制进扩展。
