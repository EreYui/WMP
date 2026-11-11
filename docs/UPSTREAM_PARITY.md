# 原项目与迁移实现的训练语义对照

核查时间：2026-10-15。基线为同级 `../wmp_go1` 的完整源码快照，提交
`65cf1afd13e4792bce7b9def6f44372ecbb3cdf0`。
本页前半记录 v2 问题；v3 修正状态见下节。原长训练保持暂停。

v0.4.0 进一步恢复逐关节力矩/速度、显式 PD、限幅后电机强度及物理子步延迟，
详见 [控制链对齐](CONTROL_ALIGNMENT.md)。此变更与 v0.3.0 历史短测通过控制版本隔离。

## v3 修复状态（0.3.0）

- Actor 恢复直接 std；AMP 恢复两层 ReLU、共享分组 Adam、weight decay、归一化裁剪与更新顺序。
  PPO 各 epoch 复用每轮 rollout 的同一次 minibatch 洗牌，与原版 storage 一致。
  差分测试直接执行原版 PPO/AMP `update`，核对所有策略/判别器参数及归一化统计。
- 世界模型复用固定上游 Dreamer 核心，恢复 GRU、CNN/MLP、LayerNorm、symlog、类别分布、
  图像/本体事件维求和损失。固定随机种子下，原版 `_train` 与新适配的总损失及每个参数梯度一致。
- 深度预测器恢复原版初始化、NHWC 种子布局、无界输出以及 Adam 优化器，移除迁移时的 tanh 输出。
- CPU 完成 episode 缓存恢复原版数据选择规则；不量化图像，每个环境保留最近完成序列，
  WM 按序列长度加权，深度监督按有效相机环境均匀。无相机样本用当前预测器重新生成图像。
- 恢复已核实的速度裁剪、脚接触过滤、cheat/stuck/stool/stumble 奖励；移除额外前进奖励与跳跃竖直速度减罚。
  关节加速度惩罚改为策略步速度差分，避免使用物理子步的瞬时加速度改变惩罚口径。
  恢复原版出生随机范围、关节/根速度随机化、跌落/翻转/高级别速度违反终止、地形课程与前向命令过滤。
- 相机分配恢复强制真实 3-D 相机后优先障碍组 80% 的规则。默认关闭观测噪声，与上游一致。
- 新增精确读取原版 `MOTION_FILES` 的 `--dataset upstream`，保留 9 条混合动作供对照；
  Go1/Go2 日常默认动作集仍按机器人选择，不能把这些实验称为同数据上游对照。
- v3 checkpoint 与 v2 隔离；旧第 9500 轮模型已通过历史实现加载及 TorchScript 导出，误差为 0。

仍有后端适配差异：IsaacLab 资产/接触/PD 执行、重建地形几何、足边掩码近似、深度采样实现及
终止帧时序。没有完成同机器人、同数据、同地形的原版长训对照，不能宣称性能等价。
具体运行证据见 [UPSTREAM_ALIGNMENT_RESULT.md](UPSTREAM_ALIGNMENT_RESULT.md)。

## v2 核查记录

v2 代码属于重写与适配，未满足原版训练算法等价性。相同的张量尺寸、网络层宽和可运行的
训练链路不足以证明训练行为等价。此前短时验收没有覆盖这些语义差异与长期策略退化。

## 确认存在的偏差

| 项目 | 原项目 | v2 wmp_lab |
|---|---|---|
| PPO 动作噪声 | ActorCriticWMP 直接优化 `nn.Parameter(std)` | 优化 `log_std` 并指数化，两者优化动力学不同 |
| AMP 网络 | 1024、512 两个隐藏层后均接 ReLU | 通用 MLP helper 将 512 作为输出层，缺少该层后的 ReLU |
| AMP 优化 | Actor 与判别器参数共用分组 Adam；判别器 trunk/head weight decay 分别为 0.001/0.1 | 独立 Adam，判别器没有对应 weight decay；自适应学习率仅改 Actor 优化器 |
| AMP 归一化 | Normalizer 将归一化结果裁剪到 ±10，按原更新顺序执行 | 无 ±10 裁剪，统计更新时机与数据处理顺序改变 |
| 世界模型结构 | 原 Dreamer RSSM、自定义带归一化的 GRU、MLP/CNN 归一化 | 独立简化实现、PyTorch GRUCell，若干归一化结构不同 |
| 世界模型图像损失 | 每个样本先按像素求和，再平均 batch/time | `F.mse_loss` 对像素及 batch/time 一起取均值 |
| 世界模型本体损失 | symlog MSE 按本体维度求和 | symlog MSE 按本体维度取均值 |
| 障碍速度奖励 | gap/bream/continuous_gap/climb 对超出命令方向的速度作特殊裁剪 | 统一普通平方速度误差，超出指令速度时的奖励行为不同 |
| 其他任务奖励 | cheat 为偏航阈值，stuck 基于机体前向速度，stool 为高度阈值 | cheat 为横向距离，stuck 改为世界 x 速度并限定方向地形，stool 为目标高度平方误差；另新增前进项 |
| 地形与课程 | 原 Isaac Gym 地形生成、出生区、升降级规则 | 多处按 IsaacLab 重建或改写，尚未逐项证明任务等价 |
| 默认训练任务与动作集 | Go1；默认 9 条，混合 trot1/trot2 与 7 条 Go2 动作 | 本轮为 Go2；全部 17 条 Go2 动作 |

这些既包含有意的机器人/后端适配，也包含不应混入等价迁移的学习算法变更。
尚未完成同机器人、同数据、同任务的上游对照训练，不能把本轮退化归因到某一个差异，
也不能据此量化原项目与当前实现的性能差距。

## 世界模型损失尺度的直接复核

从上游 `dreamer/tools.py` 提取原始 `MSEDist` 类，使用形状
`[2, 3, 64, 64, 1]` 的全零预测与全一目标进行数值比较：

- 原版 `-MSEDist(pred).log_prob(target).mean()` = **4096**。
- 当前 `F.mse_loss(pred, target)` = **1**。

因此在相同预测/目标下，当前图像重建项相对未改尺度的 KL 项被缩小 **4096 倍**；
本体重建项相应缩小 **33 倍**。这改变了训练目标中各项的相对权重。
当前世界模型 loss 约 0.61 且贴近 KL 下限，不能与原项目的 loss 数值直接比较，
也不能仅以该数字小或下降认定已复现原世界模型学习效果。

## 对上一轮诊断的订正

以下行为在原项目中也存在，不能称为迁移独有缺陷或直接认定为本次失败的根因：

- Actor 动作 std 只设置下限，没有上限。
- 环境裁剪到 ±6，而 WMPRunner 的动作历史保存裁剪前动作。
- 总任务奖励在 `only_positive_rewards=True` 时截断为非负。

本轮回放已经确认的事实仍然成立：左后膝原始动作与 std 发散、长期饱和，基础步态退化，
移除障碍后仍然无法有效前进。上面的共同行为可能与其他改动发生相互作用，
但它们的存在本身不能解释上游与本轮实现的表现差别。
尤其需要区分直接优化 std 与指数化 log_std，这才是已确认的动作分布参数化差异。

## 后续基线原则

优先恢复原项目的 Actor/PPO/AMP、归一化、Dreamer 网络及损失语义，
IsaacLab 适配集中在环境、资产、传感器、接触数据与 reset 接口。
先固定机器人、数据与简单地形建立可重复的步态基线，再验证 gap/climb 与混合地形。
任何新增动作约束、奖励塑形或算法改动应作为独立消融，不能未经验证地当作原版迁移。

主要依据：

- `../wmp_go1/rsl_rl/modules/actor_critic_wmp.py`
- `../wmp_go1/rsl_rl/algorithms/amp_ppo.py`、`amp_discriminator.py`
- `../wmp_go1/rsl_rl/utils/utils.py`、`runners/wmp_runner.py`
- `../wmp_go1/dreamer/{configs.yaml,models.py,networks.py,tools.py}`
- `../wmp_go1/legged_gym/envs/base/{legged_robot.py,legged_robot_config.py}`
- `../wmp_go1/legged_gym/envs/a1/a1_amp_config.py`
- 当前 `source/wmp_lab/wmp_lab/learning` 与 `tasks/direct/wmp_env.py`
