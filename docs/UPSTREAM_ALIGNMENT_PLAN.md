# 上游训练语义恢复方案

2026-10-15，基线提交 `65cf1afd13e4792bce7b9def6f44372ecbb3cdf0`。
旧训练保持暂停，修复后的短训练使用新目录和新 checkpoint 版本。

## 模块边界

- Actor/PPO/AMP：恢复直接 std、共享分组 Adam、两层 ReLU、归一化裁剪与原版更新顺序。
- 世界模型：在包内保留上游 Dreamer networks/tools 原文件与 MIT 声明，包装现有接口，
  复用原版 Encoder/Decoder、RSSM/GRU、归一化、分布及逐维求和损失。
- 导出：使用独立的可脚本化推理适配器，逐步对照原版 RSSM，保留双频率与 reset 验证。
- 环境：恢复已核实的速度裁剪、cheat/stuck/stool/stumble 等奖励，以及可选择的原版课程/重置语义。
  Go1/Go2 资产和 IsaacLab 传感器适配保留；尚未逐项等价的几何明确列出。
- 旧版：将 v2 学习实现归档，保留旧模型回放能力；v3 不直接恢复 v2 优化器和世界模型。
- 监控：增加逐关节 std/动作均值/裁剪率、回报、终止及实际前进记录，及时识别有限值退化。

## 验证顺序

1. 原版与新适配的固定输入/权重差分测试：Actor/PPO/AMP、RSSM reset、损失及梯度尺度。
2. 原有项目测试、checkpoint/导出/旧版读取兼容检查。
3. 共享 wrapper 真实 CUDA：Go1/Go2 更新、完整规模显存、相机及 reset。
4. 新目录短训练，保存早期 checkpoint；平地和 gap/climb 固定指令回放。
5. 报告实际进展及剩余差异。短训练不作为两万轮收敛保证，不自动恢复旧长训练。
