# Go1 / Go2 部署接口

## 输入与输出

部署文件是 B=1 的有状态 TorchScript，不包含 Critic、AMP 判别器或训练用深度预测器。
使用 `torch.jit.load` 后先 `reset()`；每 0.02 s 调用一次。
若机器人重置、传感器中断或更换控制阶段，需要重新 reset。
v0.6 的 `go1_reference_v6` 策略默认对 RSSM categorical 状态采样，与训练一致；
Actor 使用动作均值。旧环境版本继续使用历史推理方式，具体以导出 JSON 为准。
导出数值校验临时使用 categorical mode，保存前恢复采样；因此校验误差不表示两次采样轨迹相同。

`obs45` 顺序为：角速度3×0.25、投影重力3、机体速度指令3×[1,1,0.25]、
关节角减 nominal_q 的12维、关节速度12×0.05、上一实际应用动作12维。
所有关节都按 FL/FR/RL/RR × hip/thigh/calf。
命令不是实测速度；线速度估计由历史编码器完成。

深度为 `[1,64,64,1]`；也接受 NCHW `[1,1,64,64]`、BHW 和二维单帧。
输入应是轴向深度，`clip(depth_m,0,2)/2-0.5`。
无效距离应处理为 2 m，并按训练相机 FOV/安装位姿裁剪/标定。
Go1 相机 (0.28,0,0.12)，Go2 相机 (0.32715,-0.00003,0.04297)，默认下俯20度、水平FOV58度。
训练默认延迟一个相机周期。新环境版本的部署桥将初始归一化深度设为0，
从0开始计数，在控制调用3、8、13…结束后采图，在调用4、9、14…更新 RSSM。
更新读取上一周期图像，因此调用4读取初始0，调用9读取调用3采集的图像。
`ReferenceDepthBuffer` 实现这一缓冲时序。外部桥需要根据导出 JSON 的相位和延迟字段实现相同流程。

输出是关节动作：`q_target = nominal_q + 0.25*clip(action,-6,6)`。
使用导出 JSON 中的 nominal_q、PD 增益和 torque_limit，不将动作直接作为力矩。

## MuJoCo

`tools/mujoco_deploy.py` 使用包内机器人 MJCF，显式解析关节、qpos、qvel 和 actuator 映射；
每个控制周期运行4步 PD 力矩控制，深度每5步更新，按策略环境版本选择对应缓冲时序。
`--collect FILE` 保存带 `JointOrder` 和 `Robot` 标签的 WMP JSON。
`--motion FILE.npz` 可回放转换动作；采集与转换不做机器人间运动学重定向。
内置 MuJoCo 场景为平地；复杂地形 S2S 需在 MJCF 中增加相同障碍，不能直接把平地结果当越障验证。

## ROS2 / Gazebo / 实机

`ros2_policy_node.py` 订阅 `/joint_states`、`/imu/data`、`/cmd_vel` 和 `/camera/depth/image_raw`，
按关节名重排输入，用 CameraInfo 内参将32FC1米或16UC1毫米深度重采样到58度FOV，发布 `/wmp/joint_targets` JointState。
已预处理的64×64归一化深度可用 `--normalized-depth`，不再重复归一化。
输入缺失/超过0.3秒未更新时停止发布；恢复后重置策略状态。
需要 ROS2、rclpy、对应 message 包和 OpenCV，运行环境与 IsaacLab 训练环境可以分开。

该节点提供推理/数据桥，不管理电机驱动、机器人模式切换或 Gazebo控制器。
使用 [LeggedSkillDeploy/wmp_parkour](https://github.com/haozhang04/LeggedSkillDeploy/tree/wmp_parkour)
或已有 Go1/Go2 驱动接收关节目标，按 JSON 契约适配输入/输出顺序和相机格式。
上游 Isaac Gym 部署的 FR/FL/RR/RL 顺序与本项目不同，不能直接替换 `.pt` 后忽略此差异。
本项目没有对实际 ROS2/Gazebo或机器人完成运行验证。
