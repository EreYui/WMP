# v0.3.0 上游语义修复与短测

后续 v0.4.0 已单独修正执行器与控制链，见 [CONTROL_ALIGNMENT.md](CONTROL_ALIGNMENT.md)。
以下是修正前的历史短测；其 checkpoint 回放会自动使用旧控制版本，数值与视频结论保留。

2026-10-15。上游固定提交 `65cf1afd13e4792bce7b9def6f44372ecbb3cdf0`。
旧长训练 `logs/go2/20261014_113404` 保持暂停，未恢复第 9500 轮失效策略。

## 修正范围

- 策略恢复直接 std，AMP 两层 ReLU、共享分组 Adam、weight decay、归一化裁剪/更新顺序，
  PPO 各 epoch 复用每轮 rollout 的 minibatch 划分。
- 世界模型复用上游 Encoder/Decoder、自定义 GRU/RSSM、类别分布、归一化与求和损失，
  恢复本体/图像重建项与 KL 的相对尺度。深度预测器恢复原版布局、初始化、无界输出及 Adam。
- 浮点 CPU 缓存保存每个环境最近完成的 episode，WM 按有效长度加权，深度监督均匀选有效相机环境；
  未完成 episode 不参与训练，无相机图像按当前预测器重建。
- 恢复特殊速度奖励、脚接触过滤、cheat/stuck/stool/stumble、策略步关节速度差分惩罚，
  移除额外前进项/跳跃减罚；恢复出生随机范围、终止、课程和前向命令过滤。
- 新增原版默认 9 条混合动作的转换选项。Go1/Go2 原默认动作集保留，避免混淆机器人数据选择。
- checkpoint 升至 v3，v2 用归档代码回放/导出；新增逐关节 std、均值、裁剪率和训练停止保护。
  停止保护不裁剪策略分布，也不修改原版梯度。

## 验证证据

`env_isaaclab`：23 项测试通过，1 个需 Isaac Sim 启动的地形模块在普通 pytest 中跳过，Ruff 通过。
差分测试执行同级固定原源码：Actor 前向/熵梯度、完整 PPO/AMP 单步参数更新、WM 总损失/全部参数梯度、
深度预测器和特殊速度奖励。另检查完成 episode 隔离、替换、加权采样、稀疏相机及浮点保持。

Go1/Go2 均经共享 wrapper 完成最终代码的真实 CUDA 完整网络 smoke：
18 类地形、随机化、相机、reset、PPO/AMP/RSSM、checkpoint 和 TorchScript。
记录：`logs/validation/v3_alignment_20261015/wmp_v3_{go1,go2}_verified_smoke.log`。
v2 `model_9500.pt` 读取/导出通过，误差为 0。

新版完整网络额外执行 64 帧序列、50 次深度及 10 次 WM 更新：
512/128 相机通过，整卡峰值 7.71 GiB；4096/1024 相机通过，峰值 11.12 GiB。
规模测试用环形缓存稳定构造 64 帧，不替代正式 episode 采样验证。
详见 [规模记录](GO2_BENCHMARK.md)。

## 分阶段短训

Go2、512 环境、128 相机、16 行默认混合地形，使用原 17 条 Go2 动作，完整网络和默认更新次数。
本次同时做源码复核，检查点和源码 SHA-256 均保留；阶段边界如下：

| 迭代 | 目录 | 说明 |
|---|---|---|
| 0–50 | `logs/go2/v3_alignment_20261015` | 核心网络/损失、AMP、episode 采样预检 |
| 51–100 | `logs/go2/v3_alignment_final_20261015` | 恢复 PPO 各 epoch 的 minibatch 复用 |
| 101–200 | `logs/go2/v3_alignment_verified_20261015` | 补正策略步加速度惩罚后继续验证 |

边界处明确停止进程，从已完整保存的检查点恢复；恢复时不加载大回放，按规则重新预热。
阶段目录可能含停止前多执行的少数迭代；后续模型只从表中的检查点恢复。
这是一轮分阶段调试短测，不能当作从零、全程同配置的 200 轮受控实验。

短测已完成并退出，累计有效环境步 **2,457,600**，98 轮记录了 WM 更新。
最后 std 范围 **0.303–0.658**，左后膝 std **0.658**；最大逐关节平均绝对动作均值 **0.856**。
200 轮记录及四组回放均未出现超出 ±6 的动作，未触发健康停止保护。
这仅覆盖本次短测，不构成长期不发散保证。

末轮深度 MSE 0.00451，WM loss 9.427（本体 0.359、图像 5.064、KL 6.674，按原版权重组合）。
这些是训练批次指标，不是独立验证集精度或感知质量证明，也不能与 v2 的不同尺度 loss 直接比较。
gap/climb 末轮训练等级均为 **0**。

- [训练曲线](../logs/go2/v3_alignment_verified_20261015/health.png)，虚线标记第 50/100 轮开发阶段边界。
- [训练汇总](../logs/go2/v3_alignment_verified_20261015/training_summary.json)。
- [最终 checkpoint](../logs/go2/v3_alignment_verified_20261015/model_200.pt)。

## 固定指令回放

真实相机、0.45 m/s、20 秒，关闭随机化和课程，gap/climb/rough_flat/stairs_up 每类 4 个环境。
表中是同类全部样本的最大有符号前进距离，不能当作平均值或成功率。

| 模型/模式 | gap 最大前进 m | climb 最大前进 m | 达到 4 m 条件 |
|---|---:|---:|---|
| 初始模型，确定性 | 0.000 | 0.000 | 均为 0/4 |
| 200 轮，确定性 | 0.296 | 0.297 | 均为 0/4 |
| 200 轮，采样策略和 RSSM | 1.021 | 0.586 | 均为 0/4 |
| 200 轮，移除两类障碍、确定性 | 0.292 | 0.330 | 均为 0/4 |

确定性 rough_flat 最大前进仅 0.051 m；视频主要表现为站立、低姿态和小幅调整。
移除障碍后前进仍有限，因此不能把失败仅解释为障碍太难。
本次修正后尚未形成有效前进步态，也没有证明已复现原版越障效果。
不能把不同环境数、训练步数、默认数据及开发阶段的结果用作 v2/v3 的受控性能排名。

- [gap 视频](../logs/go2/v3_alignment_verified_20261015/diagnostics/mean_real/gap.mp4)
- [climb 视频](../logs/go2/v3_alignment_verified_20261015/diagnostics/mean_real/climb.mp4)
- [确定性记录](../logs/go2/v3_alignment_verified_20261015/diagnostics/mean_real/summary.json)
- [采样记录](../logs/go2/v3_alignment_verified_20261015/diagnostics/sample_real/summary.json)
- [移除障碍对照](../logs/go2/v3_alignment_verified_20261015/diagnostics/mean_flat_control/summary.json)
- 各目录 `trace.npz` 保存逐步位置、速度、关节、力矩、动作和精确奖励分解。

原 v2 长训练保持暂停，新的短测已结束，没有后台训练进程。

## 剩余差异与结论边界

IsaacLab 的资产、接触、执行器/电机模型、动作延迟与随机化仍是后端适配；
地形按网格重建，脚边掩码仍是高度射线近似，未达到原 heightfield 全局边缘网格等价。
终止帧时序、相机采样/缓存时序也未证明逐步等价。默认 Go2 动作集与原版 Go1 混合动作集不同。
测试证明修正的模块能更新且满足所覆盖的数学契约，不证明原版长训效果或长期稳定性。
