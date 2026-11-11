# WMP Lab：Go1 参考迁移与 Go2 支持

IsaacLab 2.3.2 / Isaac Sim 5.1 / Python 3.11 独立扩展，运行环境固定为 `env_isaaclab`。
参考同级 `wmp_go1` 的固定提交 `65cf1afd13e4792bce7b9def6f44372ecbb3cdf0`。
版本 **0.7.1** 默认复用 wmp_go1 的 Warp 相机核心，仍使用 `go1_reference_v6` 环境；
相机实现和全部源码哈希写入完整契约。相机与 Go2 复核见
[WMP_CAMERA_GO2_RESULT.md](docs/WMP_CAMERA_GO2_RESULT.md)，前次迁移记录见
[REFERENCE_ALIGNMENT_RESULT.md](docs/REFERENCE_ALIGNMENT_RESULT.md)。本轮未启动长期训练。采样与更新改为 [gpufree 4096 基准](docs/GPUFREE_4096_ALIGNMENT_PLAN.md)。

## 参考默认值

| 项目 | 默认值 |
|---|---|
| 机器人 | Go1；`--robot go2` 使用 Go2 资产、关节限制和相机 |
| 参考环境 / 真实相机 | gpufree 成功 Go1：4096 / 1100；低显存使用1024 / 275 |
| 地形 | 上游17类原比例、10行×20列、原始heightfield与附加障碍 |
| AMP | 上游固定9条混合动作，原时间采样、200万对预加载 |
| 策略 / 物理 / 世界模型 | 50 / 200 / 10 Hz |
| PPO | gamma=0.99、GAE lambda=0.95、每段GAE 24步、完整批次98304样本、5 epoch×4 minibatch |
| 最小探索噪声 | 按90%软关节范围计算 |
| 种子 | 1；可显式覆盖 |
| 训练入口上限 / 保存间隔 | 100000 / 1000迭代；短验证必须显式指定迭代数 |

Go2 保留独立相机位置、初始高度和逐关节力矩/速度限制；tilt 宽度适配为0.34–0.42m，
Go1为0.28–0.32m。机器人适配属于明确差异，不宣称Go1/Go2具有相同动力学或训练效果。
梅花桩保留为显式扩展，不加入参考默认比例；`--terrain all` 启用全部18类。

## 按 gpufree 成功训练对齐完整更新

新正式训练使用 `gpufree_4096_v1`。4096环境每轮采24步；2048环境采2段×24步；
1024环境采4段×24步。所有段采集完成前冻结模型，GAE分别以24步计算，不跨段延长到96步。
合并后的有效PPO批次固定98304样本，5epoch×4minibatch，每个minibatch为24576；
内部以6144样本微批次累积梯度，每个有效minibatch只执行一次KL调度、裁剪、Adam和AMP统计更新。

每个完整批次固定更新世界模型10次、深度预测器50次，不再按环境数取整为2/12。
1024环境每个保留4个完成回合槽，2048环境每个2槽，匹配4096个完成回合的缓存规模；
只采样完成回合，仍按长度加权，深度监督均匀采样有效相机槽。
预测深度在env.step后缓存，跨模型更新边界保留原参数生成的帧。

奖励课程按累计环境步/98304推进，4000～10000个参考批次的0.1→1.0系数不变。
速度课程累计到参考4096环境的各组回合数后评分，原0.8阈值、0.2增量和速度上限不变。
每个物理环境轮换独立地形课程状态（1024为4份），每次完成回合只更新对应槽，日志统计所有逻辑槽。
地形升级条件及奖励公式、权重均未改动。全部课程槽、当前槽和累计评分都保存到checkpoint。

日志和checkpoint中的`iteration`表示一次完整更新批次：1024环境的一轮包含96个物理控制步。
`collection_steps`、`batch_samples`、`gae_horizon`、有效minibatch、真实优化器步数和课程时钟均写入日志。
旧的1024-env/24步更新checkpoint保留回放/导出能力；禁止静默切换时钟续训，需开始独立实验。
正式环境数必须整除4096；不满足时明确报错。显式smoke/benchmark保留有界诊断配置。

物理并行度仍是1024，连续采样不能复制4096个独立环境的轨迹；对齐的是批次、GAE窗口、
优化更新和课程计量，不承诺逐轨迹或收敛效果相同。

## 安装与资产

```bash
cd /home/re/re_lab/wmp/wmp_lab
conda run -n env_isaaclab pip install -e source/wmp_lab

# 仅首次安装或主动重建资产时执行；生成独立reference资产，保留历史USD。
ISAACLAB_SMOKE_SCRIPT=$PWD/scripts/convert_robot.py ISAACLAB_SMOKE_STEPS=1 \
  /home/re/IsaacLab/tools/run_codex_isaaclab_smoke.sh
```

Go1 转换前合并非脚部固定链接，显式守恒质量/COM/惯量，移除空肩链接。
参考资产位于 `source/wmp_lab/wmp_lab/assets/robots/{go1,go2}/reference/`。
旧资产保留在各机器人原目录；Go2参考入口复用原几何，并显式设置碰撞属性。
不修改 IsaacLab 源码，不覆盖系统 RSL-RL，不依赖在线机器人资产。

## 使用

```bash
# 低显存、固定迭代的功能检查；不用于评估步态。
./bash/leggedskill.sh -t --robot go1 --smoke --iterations 2 --output /tmp/go1_smoke.pt
./bash/leggedskill.sh -t --robot go2 --smoke --iterations 2 --output /tmp/go2_smoke.pt

# 独立地形和机器人专用数据均为显式选择。
./bash/leggedskill.sh -t --robot go2 --terrain rough_flat --amp-set robot \
  --num-envs 512 --camera-envs 128 --iterations 2 --log-dir /tmp/go2_check

# 仅v4 checkpoint可续训；完整环境/源码/资产/AMP契约不同会拒绝。
./bash/leggedskill.sh -t --resume --path /path/to/model.pt --iterations 2

# 播放默认Actor均值+采样RSSM，与上游play语义相同；mode用于确定性对照。
./bash/leggedskill.sh -p --path /path/to/model.pt --terrain climb --headless
./bash/leggedskill.sh -p --path /path/to/model.pt --terrain gap --rssm-mode mode --headless
./bash/leggedskill.sh -v --path /path/to/model.pt --num-envs 18
./bash/leggedskill.sh -e --checkpoint /path/to/model.pt --output /tmp/policy.pt
```

上述仅列功能检查命令，不会自动启动正式长训练。完整网络正式规模需要足够显存和CPU内存。
低环境数时也必须覆盖所有tilt/stool/crawl的真实相机，否则明确报错。
本机12GB显卡的Go2参考自碰撞任务需要显式降低规模，例如`--num-envs 1024 --camera-envs 275`；
旧4100/1100配置在本机曾OOM；历史0.7配置的2048/550短验证峰值约11.03GiB，
该历史显存数据不能代替新完整批次的验证。
规模检查使用有界CPU环形回放，正式训练默认使用完成episode回放，
长时间回放的CPU内存占用需要另行观测。
`--amp-set robot` 为Go1的19条或Go2的17条专用数据；不做运动学重定向。
自定义数据使用 `--motions`；`--amp-set upstream` 显式选择原9条。

## 实现与边界

- 公共观测为特权53+本体33+上一动作12+高度187=285；特权字段顺序与上游一致。
  历史去掉指令，42×5=210；AMP为q12+机体线速度3+角速度3+dq12=30。
- 动作按FL/FR/RL/RR×hip/thigh/calf；四元数wxyz。目标为nominal_q+0.25×clip(action,-6,6)。
  显式PD先限幅，再乘物理子步电机强度；支持0/1/2子步动作延迟。
- 使用原heightfield三邻格最小值查询187/525维高度，保留原点顺序和投影四元数旋转规则。
  脚边惩罚使用全局x边缘掩码和按名称映射的真实足端位置。
- 默认复用 wmp_go1 的 WarpCam 和原版 kernel，轴向64×64、58°水平视场，0–2m归一化至[-0.5,0.5]。
  射线使用原整数像素坐标，只为实际相机分配渲染缓冲；环境双帧输出接口仍按全部环境保存。
  相机10Hz更新，延迟一帧；位置±1cm、pitch15–25°、roll/yaw±2°在初始化时随机一次。
  有相机用真实深度，其余用高度525+本体33预测；tilt/stool/crawl排除出预测器监督。
  可选RTX后端用于同场景对照。
- Actor、AMP-PPO、上游Dreamer网络/损失、完成episode缓存与AMP采样均有数值对照测试。
  reset零动作步、初始episode长度随机化、命令边界、timeout和RSSM调度按参考流程处理。
- 导出为有状态TorchScript，输入obs45和真实深度，输出action12；含历史、RSSM和双频调度。
  默认采样RSSM。导出误差检查在确定性分支执行，不能把该误差当作实机性能证据。
- checkpoint为v4，保存优化器、AMP统计/抽样状态、课程和完整环境/源码/资产契约。
  大型回放与完整物理/RNG状态不保存，恢复后重新reset和预热，属于语义一致的续训，不是逐帧复跑。
- 缺少相机实现字段的历史模型回放自动选择旧 `isaaclab_sparse`；新模型记录 `wmp_go1`。
  完整契约或源码哈希不同的 checkpoint 会拒绝续训，不能将新相机静默混入旧实验。
- v2/v3历史模型保留回放/导出入口，不能跨到新环境直接续训。旧Go1资产仍在；
  通用索引修复也适用于历史环境，因此不承诺历史轨迹逐元素不变。

Isaac Gym与Isaac Sim/PhysX版本、接触张量/force sensor、GPU非确定性仍有后端差异。
本机未运行Isaac Gym性能基线，没有长训收敛或实机验证结论。

## 验证

```bash
conda run -n env_isaaclab pytest -q
conda run -n env_isaaclab ruff check .

# 两种机器人：全地形、完整网络、短更新、保存/恢复/导出；固定步数退出。
for robot in go1 go2; do
  WMP_ROBOT=$robot WMP_SMOKE_FULL=1 WMP_SMOKE_PAPER=1 \
    WMP_VALIDATION_OUTPUT=$PWD/logs/validation/reference_alignment_20261022/$robot.json \
    ISAACLAB_SMOKE_SCRIPT=$PWD/scripts/wmp_smoke.py ISAACLAB_SMOKE_STEPS=10 \
    ISAACLAB_SMOKE_TIMEOUT=300 /home/re/IsaacLab/tools/run_codex_isaaclab_smoke.sh
done
```

仅CPU测试不能覆盖实际资产质量、碰撞、索引和传感器。真实CUDA检查必须通过共享wrapper，
不以CPU回退替代。验证结果和剩余差异见[修复结果](docs/REFERENCE_ALIGNMENT_RESULT.md)。

## 部署与工具

MuJoCo平地推理/动作采集：`tools/mujoco_deploy.py`；ROS2传感器到关节目标桥：
`scripts/ros2_policy_node.py`。部署元数据记录相机相位和初始深度；参考部署桥保留一相机周期延迟。
驱动、模式管理和Gazebo控制器由外部部署包提供，详见[部署契约](docs/DEPLOYMENT.md)。

训练写入TensorBoard、metrics.jsonl、run.json及周期checkpoint，并监控逐关节std/均值/饱和率。
`tools/` 保留批量录制、等级图、深度比较、日志管理和AMP数据审计。
TensorBoard见[远程访问](docs/TENSORBOARD_REMOTE.md)。历史诊断见
[Go2训练诊断](docs/GO2_TRAINING_DIAGNOSIS.md)；它们对应旧环境和控制版本。

许可证及上游来源见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
