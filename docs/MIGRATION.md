# WMP Go1/Go2 IsaacLab 重构设计

参考：haozhang04/wmp_go1，提交 65cf1afd13e4792bce7b9def6f44372ecbb3cdf0。
后端固定为现有 IsaacLab 2.3.2 / Isaac Sim 5.1 / env_isaaclab。

v0.4.0 单独恢复上游显式 PD、逐关节限制、限幅后电机强度及物理子步延迟，
历史 checkpoint 使用旧控制版本回放，详见 [CONTROL_ALIGNMENT.md](CONTROL_ALIGNMENT.md)。

2026-10-15：0.3.0 已恢复主要上游学习语义，并对固定输入/权重的前向、损失、梯度及
PPO/AMP 参数更新执行差分验证，详见 [UPSTREAM_PARITY.md](UPSTREAM_PARITY.md)。
功能清单不能视作收敛或原版效果验收；真实短训结果见 [修复验证](UPSTREAM_ALIGNMENT_RESULT.md)。

## 模块边界

- `source/wmp_lab/wmp_lab`：公共张量契约、机器人配置、环境、学习与导出。
- `wmp_go2`：历史导入兼容入口；Gym Go2 任务 ID 保留，新增 Go1 ID。
- `tasks/direct`：统一环境、机器人差异配置、完整地形与三组课程。
- `learning`：AMP-PPO、离散 RSSM、连续 rollout、序列回放、checkpoint、部署策略。
- `scripts`：训练/恢复、播放/录制、地形查看、动作转换、TorchScript 导出、真实 smoke。
- `tools`：批量播放、分地形等级绘图、深度误差对比、日志预览/压缩/清理。

## 功能验收清单

1. 离线 Go1/Go2 资产、关节顺序 FL/FR/RL/RR、独立相机和执行器参数。
2. 上游 17 类地形，保留本地梅花桩；全地形和单地形配置、难度课程。
3. 五组独立速度课程；脚边惩罚奖励课程；真实物理随机化及动作延迟。
4. IsaacLab Warp ray-cast 深度、64x64、[0,2]m→[-0.5,0.5]；3-D 障碍强制真实相机。
5. 策略 50 Hz / 世界模型 10 Hz；episode reset 清除状态；相机帧延迟可配置。
6. AMP 终止前状态、PPO GAE/timeout bootstrap、多 epoch/minibatch/clipped value。
7. 离散 RSSM 重建/平衡 KL；有界 CPU 序列缓存，预测深度监督排除 3-D 障碍。
8. 断点带 robot/config/迭代/课程与优化器；恢复不包含大回放；原子保存。
9. 有状态 TorchScript：obs45+depth→action12，reset、双频率、数值一致性验证。
10. CLI / TensorBoard / JSONL / 视频 / 地形等级 / 深度对比 / 日志管理 / 动作工具。
11. 离线测试先于真实模拟器 smoke；两种机器人均测试 reset、sensor、联合更新、导出与恢复。

## 部署边界

迁移上游统一策略部署接口，提供 MuJoCo 推理工具和外部 Gazebo/Go1 实机部署接入说明。
无硬件时不宣称实机验证；采集的数据保留来源与机器人标签，不把动作重排当运动学重定向。
新架构 checkpoint 使用版本标记，不伪装兼容旧 Gaussian RSSM 或 Isaac Gym checkpoint。
训练、短 smoke 和收敛评估分别说明；短 smoke 不证明步态质量。

## 实现差异与结果

- 全部训练/感知/导出功能在公共模块实现，旧 `wmp_go2` 名称仅为兼容导入。
- 世界模型直接复用固定上游 Dreamer 的 Encoder/Decoder、自定义 GRU/RSSM、归一化和分布；
  推理导出使用经数值对照的可脚本化适配器，不承诺原 checkpoint 键直接兼容。
- IsaacLab 的轴向深度对应 `distance_to_image_plane`；不能用射线长度 `distance_to_camera` 替代。
- RTX 后端显式同步独立相机位姿、刷新渲染帧和 sensor buffer；ray-cast 为默认训练后端。
- 地形按 IsaacLab 原生配置/网格重建，出生区与跨单元障碍做了修正；不承诺逐像素几何一致。
- v3 地形课程恢复原版距离判定和最高级后的随机等级，五组速度课程独立升级；
  移除迁移时增加的 Go2 助跑速度下限，恢复前向低速命令过滤。
- CPU 缓存保留浮点深度及最近完成 episode，世界模型按长度加权采样；仅 smoke 使用环形缓存。
  渲染/PPO/深度/RSSM更新全包含在 CLI 的 iteration_seconds。
- v3 与 v2 checkpoint 隔离，v2 归档实现用于历史回放/导出，原长训练保持暂停。
- ROS2提供 CameraInfo/FOV 重采样、IMU/关节顺序/状态新鲜度桥；驱动管理留在外部部署包。
- MuJoCo内置平地推理、动作回放和采集；复杂障碍场景和实机收敛不计入短 smoke 的验收。
