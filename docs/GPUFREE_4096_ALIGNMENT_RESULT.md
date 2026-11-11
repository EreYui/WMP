# gpufree 4096 基准对齐：修改与验证结果

所有代码、实验和文档均位于 wp。未在 Mac 写代码，未启动长期训练。

## 基准与更新单位

基准是 gpufree 已成功的 Go1 从零训练：4096 环境、1100 真实相机、24 控制步、5 epoch×4 minibatch，世界模型 10 次、深度预测器 50 次。算法标识为 `gpufree_4096_v1`。

| 项目 | 1024 环境的新正式配置 |
|---|---|
| 物理环境 / 相机 | 1024 / 275 |
| 每个完整更新批次的采集 | 4 段×24 步，模型在采集阶段保持冻结 |
| 每段 GAE | 独立 24 步；每段有各自末状态 bootstrap |
| 有效 PPO 样本数 | 98304，与4096×24一致 |
| PPO epoch / minibatch | 5 / 4，共20次Adam更新 |
| 有效 minibatch / 显存微批次 | 24576 / 6144；梯度累积后统一KL/LR、裁剪和Adam |
| 世界模型 | 每完整批次10次；batch16、序列最多64 |
| 深度预测器 | 每完整批次50次；batch1024 |
| 世界模型启动门槛 | 10000个完成回合帧，保持参考规则 |
| 完成回合缓存 | 每物理环境最近4个完成回合，共4096个完成槽 |
| 地形课程状态 | 每物理环境轮换4份独立状态，共4096逻辑槽 |

现在日志与checkpoint的 `iteration` 表示完整更新批次。1024环境的新一轮对应旧24步日志的4轮。
2048环境为2段；4096环境为1段。正式环境数必须整除4096，无法精确组成参考批次时明确报错。

## 课程时钟与状态

- 奖励课程使用累计环境步/98304作为时钟。系数仍是0.1到1.0，节点仍为参考第4000到10000批次，即累计393216000到983040000条样本。折算为旧1024×24单位，是第16000到40000轮。
- 速度课程按参考4096环境的分组完成回合数评分：omni=410、rough_flat=204、stairs_discrete=1024、forward=2253、slope_cliff=205。分数阈值0.8、每次增加0.2、初始值和上限不变。
- 每次地形回合结束，仍用原版距离/命令条件更新对应课程槽，然后轮换到下一槽；等级统计覆盖全部逻辑槽，避免1024物理环境的平均等级按采样量推进过快。
- checkpoint保存全部地形槽、当前槽、速度课程累计分数/回合数、完整配置及累计样本。恢复后时钟与课程不重置；大回放仍需重新预热。
- 预测深度改为env.step后缓存，跨PPO与感知更新边界消费旧参数生成的帧，与Go1流程一致。

## 不变项与兼容性

通过AST比较确认 `_get_rewards` 和 `update_reward_curriculum` 的公式与修改前完全相同；修改的是调用时钟和课程证据累计方式。
奖励权重未调整，20个USD/URDF等资产描述文件的哈希未变化。Go2保留19刚体、27碰撞形状和开启的自碰撞，控制及相机安装参数不变。

旧 `upstream_reference_v6` checkpoint仍可回放/导出，禁止直接切换到新更新制度续训。
已用原Go2 `model_10000.pt` 验证网络加载、有限动作输出和旧训练入口拒绝行为。
完整网络benchmark的诊断checkpoint也不能误用于正式训练；新实验需从新配置独立开始。

## 数值与真实GPU验证

- pytest：71通过，1个依赖Isaac Sim启动的模块在普通Python中跳过；实际GPU检查覆盖其地形功能。
- ruff：全部通过。
- 数值测试直接抽取原Go1 `compute_returns` 和地形课程函数作独立对照；覆盖GAE、时间/逻辑环境合并顺序、完整/微批次梯度与Adam、KL调度、AMP统计、课程评分门槛、完成槽隔离、深度缓存及保存恢复。
- 两种机器人均通过env_isaaclab和共享GPU wrapper：全18类地形、完整网络、原PD/限矩/延迟、自碰撞、Warp深度、终止观测、联合更新、checkpoint与TorchScript。
- 最终1024/275验证固定8个完整批次后退出：786432条采样，PPO=160次，预热后世界模型=60次、深度预测器=300次。每完整批次的有效minibatch为24576，GAE为24步。
- 完整64帧显存验证通过，实际更新计数20/10/50，整卡采样峰值11.06 GiB。此项用有界ring预热验证完整模型内存，正式实验使用完成回合缓存。
- 最终checkpoint从第8批恢复并固定执行2批，到第10批、983040条样本；课程时钟依次为8、9。实际加载函数对全部课程槽、活动槽与累计速度评分逐项恢复一致。
- 原始预检8批次保留在go2_1024目录；最终源码对应go2_1024_final与go2_resume，避免混用中间版本checkpoint。

## 文件与复查入口

- 设计：/home/re/re_lab/wmp/wmp_lab/docs/GPUFREE_4096_ALIGNMENT_PLAN.md
- 数值/资源汇总：/home/re/re_lab/wmp/wmp_lab/logs/validation/gpufree_alignment_20261024/validation_summary.json
- 最终完整配置：/home/re/re_lab/wmp/wmp_lab/logs/validation/gpufree_alignment_20261024/go2_1024_final/run.json
- 最终短验证checkpoint：/home/re/re_lab/wmp/wmp_lab/logs/validation/gpufree_alignment_20261024/go2_1024_final/model_8.pt
- 恢复验证checkpoint：/home/re/re_lab/wmp/wmp_lab/logs/validation/gpufree_alignment_20261024/go2_resume/model_10.pt
- 64帧检查：/home/re/re_lab/wmp/wmp_lab/logs/validation/gpufree_alignment_20261024/world64.json
- 两种机器人：/home/re/re_lab/wmp/wmp_lab/logs/validation/gpufree_alignment_20261024/go1_final.json、/home/re/re_lab/wmp/wmp_lab/logs/validation/gpufree_alignment_20261024/go2_final.json
- 修改前文件与资产描述哈希：/home/re/re_lab/wmp/wmp_lab/.work/gpufree_alignment_20261024

## 边界

1024物理环境的连续采样不能复制4096个同时独立的物理轨迹、随机物理参数和随机数序列；本次对齐的是有效批次、24步GAE、优化步数、缓存规模和课程计量。
Isaac Gym/Isaac Sim接触信号差异及Go2资产动力学疑点未在本次调整。短验证证明流程、计数、恢复和显存可用，climb收敛仍需后续训练确认。
