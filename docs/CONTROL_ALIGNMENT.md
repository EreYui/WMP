# 第一步：执行器与控制链对齐

2026-10-15，v0.4.0，控制版本 `upstream_pd_v1`。

以同级原项目固定提交 `65cf1afd13e4792bce7b9def6f44372ecbb3cdf0` 为依据。
这一步只调整控制链与相应的历史模型兼容，不改变 AMP 数据、奖励或学习网络，不恢复训练。

## 对齐契约

- 按关节名称读取/核对原 URDF 限制，避免 12 个关节共用一个常数。
  Go1 为 33.5 N·m / 30 rad/s；Go2 髋/大腿为 23.7 / 30.1，小腿为 35.55 / 20.07。
- 使用显式 PD 与固定力矩裁剪，移除新增的 DC 力矩速度曲线。
- 电机强度在裁剪之后相乘，每个物理子步独立采样 [0.8,1.2]，沿用上游顺序。
- 延迟沿用原版每个策略步统一采样 0/1/2 个物理子步，替代概率性延迟整个策略步。
- 控制版本写入 checkpoint 与导出元数据；历史模型回放使用旧控制配置，禁止静默跨控制版本续训。

## 验证

先运行离线关节限制/历史版本测试，再通过共享 IsaacLab wrapper 验证两种机器人：
实际加载的关节参数、与原 `_compute_torques` 的固定输入差分、三个延迟值及仿真联合更新/导出。
本阶段不把冒烟更新视作步态或越障质量验收。后续单独固定 AMP 数据建立对照。

## 已完成的结果

- Go1 按原训练实际使用的 `go1_new_2.urdf` 对齐，不套用其他 Go1 描述文件的力矩值。
  Go2 按原仓库 `go2.urdf` 对齐；它是资产参数参考，原仓库没有已验证的独立 Go2 训练任务。
- 每个关节均按名称校验力矩和速度限制；原版 PD 固定限幅替代 DC 力矩速度曲线。
- 电机强度乘在限幅之后。与上游相同，1.2 倍强度时指令可达到标称力矩的 1.2 倍；
  这不能误读为限幅失效。模拟器不重复裁剪已处理的显式力矩。
- checkpoint 格式仍为 v3，新增 `extra.control_revision`；导出元数据格式为 v4，
  `torque_limits` / `velocity_limits` 按 `joint_order` 保存 12 维值。
- 缺少控制版本的历史 checkpoint 自动归为 `legacy_dc_v3`，播放/诊断重建旧 DC 控制。
  原 v2 学习实现保持归档。旧控制 checkpoint 不允许直接续训到新控制版本，需从零建立新基线。
- MuJoCo 使用同一组逐关节力矩限制，新策略同步修正 MJCF 的控制/输出力矩范围；
  旧导出的控制版本用于选择原限制。此项不代表 MuJoCo 与 PhysX 动力学一致。

验证结果：**27 项离线测试通过，1 项需 Isaac Sim 启动的地形模块跳过，Ruff 通过**。
Go1/Go2 真实 CUDA 完整网络 smoke 均通过：18 类地形、逐关节限制、原控制函数精确差分、
0/1/2 子步延迟、reset、PPO/AMP/RSSM 更新、checkpoint 和 TorchScript。
固定输入覆盖正负高速，PD 力矩与原 `_compute_torques` 乘以 0.8/1/1.2 后的输出逐元素相等。
MuJoCo 两种机器人均通过 10 个策略控制步及逐关节限制检查。

历史第 200 轮模型按旧控制重放 20 秒：位置、原始动作、关节角、施加力矩的最大逐元素差异均为 **0**。
该模型导出误差也为 0。这证明此次控制更新保留了历史比较口径，不能作为新控制性能提升证据。
记录位于 `logs/validation/control_alignment_20261015`，包括两种机器人 smoke、测试与历史回放对照。

## 下一步顺序

第二步的数据/采样固定已完成，见 [AMP_ALIGNMENT.md](AMP_ALIGNMENT.md)，训练尚未启动。

1. 固定上游原版 9 条混合 AMP 数据，与机器人专用数据分开记录；核对相对关节范围和采样分布。
2. 对齐资产质量/惯量、碰撞、自碰撞和接触参数，量化相同指令下的动力学响应。
3. 对齐相机、地形几何和脚边掩码，再建立从零开始的平地与 gap/climb 训练基线。

本步骤未改 AMP 数据、奖励、网络和地形。旧长训练与第 200 轮短测都保持结束/暂停状态，未启动新训练。


## 2026-10-31：Go2 官方 Unitree Lab 控制适配

新 Go2 使用独立控制版本 unitree_go2_v1；Go1 继续使用 upstream_pd_v1。
旧 Go2 checkpoint 仍按已保存版本恢复 40/1、旧固定限矩和小腿 20.07 rad/s，
不会静默切换为新执行器。源码/环境契约仍禁止跨配置直接续训。

参考固定的官方 unitree_rl_lab 提交
4960b84732b0c2ec593dccbfe963fda1bcd7b1e3：
PD 改为 Kp=25、Kd=0.5，关节摩擦系数 0.01；关节名到默认姿态、动作缩放的配置
现在实际写入 ArticulationCfg。官方默认姿态与原值一致，因此不额外改角度。
关节软位置比例仍为 0.9，出生高度 0.4 m，动作缩放 0.25，物理步长/策略步长仍为 5/20 ms。

Go2HV 执行器使用官方同向/反向峰值和线性扭速曲线：
X1=13.5 rad/s、X2=30 rad/s、Y1=20.2 Nm、Y2=23.4 Nm；
全部十二关节沿用官方同一曲线，不增加小腿传动倍率。
WMP 的 0/1/2 物理子步延迟及曲线裁剪后的 [0.8,1.2] 电机强度随机化保留，
未叠加官方 DelayedPD 的第二套延迟。求解器全局和机器人均设为 position=8、velocity=4。
显式 PD 下引擎隐式驱动刚度/阻尼为零，避免双重 PD。

完整 Go2 资产保留 19 刚体、27 碰撞形状和自碰撞。
源四个 USD 文件与官方 unitree_model 提交
323e350252b9c3aee9c40acbfdad84f6ce46a5ac 逐字节一致；
也与 unitree_ros 提交 3ae893d905d11795ddf1075825ee5010364a1109 的
19 刚体质量/COM/惯量、18 保留关节、27 形状一致。16.087 kg 包括官方
12 个 0.089 kg rotor；此前与旧 URDF 的 1.068 kg 差值不是额外虚假质量。
小腿 simulator 速度恢复为资产的 15.7 rad/s。奖励公式和权重未改，
依赖速度上限的原奖励门槛因此随资产恢复为 14.13 rad/s；不能把它误报为奖励数值完全不变。

原课程、地形生成、任务奖励函数、PPO/AMP/世界模型配置均保持。
降低起步台阶的候选补丁未应用。MuJoCo、导出和 ROS 位置桥按保存控制版本路由；
MuJoCo 的控制律验证不代表与 USD 完整动力学相同，ROS 外部驱动需匹配导出执行器参数。

验证记录位于 gpufree 的 .work/go2_unitree_lab_20261031：
包括官方源锁定、资产审计、CPU 差分、完整网络有限更新/保存恢复/导出、
旧第 30000 轮 Go2 回放、Go1 回归、4096 环境/1100 相机单批验证及 5 秒静止支撑验证。
验证产生的 checkpoint 仅用于验证，不作为正式训练成果；未启动长期训练。


## 2026-11-01：WMP Go2 小腿传动映射试验

独立控制版本 unitree_go2_joint_v2。保留官方直接曲线版本 unitree_go2_v1 的全部数值路径，
并保留 upstream_pd_v1 和 legacy_dc_v3。v2 是基于当前资产参数的 WMP 适配假设，
不是声称官方源码遗漏了 calf 分支。

7300 批同采样量比较中，新 v1 的 gap/bream 近期课程为 0，climb 也接近 0；
climb 近 500 批抬脚比例约 2.92%，上轮约 34.36%，而动作 std 没有坍缩。
climb 的速度与碰撞惩罚比上轮更轻，回报反而更高，说明仅凭全局速度罚不能解释停滞。
本轮只尝试修正 calf 关节输出空间与共享电机曲线的映射，暂不改奖励。

资产原生 calf/hip 最大力矩比 g=45.43/23.7=1.916877637；
速度比 30.1/15.7 与其接近（受参数舍入影响）。v2 对 calf 采用该 g，
hip/thigh 保持 g=1：
请求电机力矩 = 请求关节力矩/g，电机速度 = 关节速度*g；
原 Go2HV 电机曲线裁剪后，输出关节力矩 = 输出电机力矩*g。
因此不虚增功率，也不只是放大力矩。

calf 关节曲线变为：全力矩拐点约 7.043 rad/s，空载速度约 15.650 rad/s，
同向峰值约 38.721 Nm，静止/反向峰值约 44.855 Nm。
低速可用输出增强，15 rad/s 时同向上限反而降至约 2.926 Nm。
保持原有曲线后的 [0.8,1.2] 电机强度随机化，因此随机化命令可超过名义峰值；
这与此前 WMP 控制顺序一致，不用于真实硬件直接输出。

PD25/.5、solver8/4、engine calf15.7、原速度罚门槛14.13、站姿、相机、原课程、
奖励公式/权重、PPO/AMP/WM更新比例与完整19刚体/27碰撞形状及自碰撞均保持。
实际齿比按关节名称传入，MuJoCo、导出和ROS元数据按保存版本区分；v1 g=1 输出位级保持。
旧 v1 已在第 7500 批保存完整 checkpoint 后按用户授权停止，新 v2 从零建立独立训练，
不绕过 checkpoint 契约续训。

试验/验证/备份记录在 .work/go2_joint_drive_v2_20261101。
此修改仍需通过重新训练检验跨越能力，数值和链路验证不等于已解决 climb。


## Go2 forward-progress reward revision (2026-11-03)

New Go2 reference-environment runs select frontier_progress_v1. Go1 reference
runs retain upstream_v3, and playback explicitly selects the reward revision
saved in the checkpoint (missing fields mean the historical reference reward).
The full environment/source contract continues to reject incompatible resumes.

This addresses bounded forward/backward motion receiving locomotion rewards
without crossing obstacles. On gap, continuous_gap, bream and climb, an
episode-local tracker records the furthest position reached in the commanded
world-X direction. Its advance over a 2-second window is compared with the
existing 0.1 m/s stuck threshold. Reset and command stop/start or reversal get a
fresh window. Command magnitude changes in the same direction do not restart
the window. Small new maxima do not refresh the clock.

After sustained insufficient progress, positive task terms and AMP style reward
are gated off; negative task terms retain their original values. The historical
per-step task clamp and all numeric reward coefficients remain unchanged.
For active movement on the selected terrains, tracking uses signed frontier
speed with the original tracking kernel and overspeed clipping. The initial
window uses actual elapsed time. The command curriculum accumulates the same gated tracking score that the task
actually receives. Its rules and terrain advancement rules are unchanged.
Metrics include stalled fraction and frontier speed. Terminal-transition gates
own their tensor storage and survive automatic reset; per-episode frontier
history is reset when restoring curriculum into a freshly spawned environment.

This revision does not alter observations, policy/network dimensions, PD,
torque-speed mapping, robot assets, self-collision, camera or PPO/world-model
update settings. It is a reward-logic change and requires a separate training
experiment. Do not SIGCONT the frozen historical run as an application of this
fix; it holds the old in-memory implementation.

Validation: 134 CPU tests passed; 8 environment-dependent tests skipped. In the
180-episode GPU fixed-weight matrix, the successful Go1 policy retained 4/4
climb and 3/4 gap completions with current Go2 hardware. Normal flat rewards and
all physical trajectories matched the historical matrix; the known stationary
oscillation received no task/style rewards in its final 5 seconds. Full network
updates, checkpoint restore and export were verified in a bounded 64-env check,
plus a 4096-lane CUDA tracker check. These checks do not demonstrate newly learned
climbing or training convergence; the long run remains paused.
