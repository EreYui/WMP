#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo 部署/采集/动作回放。

用法：MUJOCO_GL=egl python tools/mujoco.py --robot go1 --policy policy.pt --steps 1000。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from wmp_lab.deploy_mujoco import MujocoRobot
from wmp_lab.deployment_depth import ReferenceDepthBuffer
from wmp_lab.robots import (
    ENVIRONMENT_REVISION,
    LEGACY_CONTROL_REVISION,
    LEGACY_ENVIRONMENT_REVISION,
    default_control_revision,
)


def deployment_options(robot, metadata=None):
    """策略遵从导出版本；无策略的动作回放使用目标机器人的当前默认。"""
    if metadata is None:
        return dict(control_revision=default_control_revision(robot), environment_revision=ENVIRONMENT_REVISION)
    if metadata["robot"]["name"] != robot or metadata["batch_size"] != 1:
        raise ValueError("policy robot/batch metadata does not match deployment")
    return dict(
        control_revision=metadata.get("control_revision", LEGACY_CONTROL_REVISION),
        environment_revision=metadata.get("environment_revision", LEGACY_ENVIRONMENT_REVISION),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("go1", "go2"), required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--motion", type=Path, help="Replay a converted NPZ sequence instead of a policy")
    parser.add_argument("--collect", type=Path, help="Save a collected sequence in WMP JSON format")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--command", type=float, nargs=3, default=(0.6, 0, 0))
    parser.add_argument("--view", action="store_true")
    args = parser.parse_args()
    if args.policy is None and args.motion is None:
        parser.error("provide --policy or --motion")
    metadata = json.loads(args.policy.with_suffix(".json").read_text()) if args.policy else {}
    options = deployment_options(args.robot, metadata if args.policy else None)
    robot = MujocoRobot(args.robot, **options)
    print(
        f"[CONTROL] revision={options['control_revision']}, actuator={robot.profile.actuator_model}, "
        f"joint_gear_ratios={getattr(robot.profile, 'joint_gear_ratios', (1.0,) * 12)}; "
        "control-law mapping only, not a USD/MJCF dynamics equivalence claim",
        flush=True,
    )
    viewer = None
    frames = []
    try:
        policy = torch.jit.load(str(args.policy)) if args.policy else None
        if policy:
            metadata = json.loads(args.policy.with_suffix(".json").read_text())
            if metadata["robot"]["name"] != args.robot or metadata["batch_size"] != 1:
                raise ValueError("policy robot/batch metadata does not match deployment")
            policy.reset()
        motion = np.load(args.motion) if args.motion else None
        if args.view:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
        previous = np.zeros(12, dtype=np.float32)
        depth = np.full((1, 64, 64, 1), 0.5, dtype=np.float32)
        reference_depth = (
            ReferenceDepthBuffer() if metadata.get("environment_revision") == ENVIRONMENT_REVISION else None
        )
        with torch.inference_mode():
            for step in range(args.steps):
                if motion:
                    index = min(int(step * float(motion["fps"]) * 0.02), len(motion["joint_pos"]) - 1)
                    robot.data.qpos[:3] = motion["root_pos"][index]
                    robot.data.qpos[3:7] = motion["root_quat_wxyz"][index]
                    robot.data.qpos[robot.q_ids] = motion["joint_pos"][index]
                    robot.data.qvel[robot.v_ids] = motion["joint_vel"][index]
                    robot.mj.mj_forward(robot.model, robot.data)
                else:
                    if reference_depth is not None:
                        observation = robot.observation(args.command, previous)
                        action = policy(
                            torch.from_numpy(observation[None]), torch.from_numpy(reference_depth.delayed)
                        ).numpy()[0]
                    elif step % 5 == 0:
                        # 默认保持一个相机周期延迟，与 IsaacLab 训练一致。
                        next_depth = robot.depth().reshape(1, 64, 64, 1).astype(np.float32)
                        observation = robot.observation(args.command, previous)
                        action = policy(torch.from_numpy(observation[None]), torch.from_numpy(depth)).numpy()[0]
                        depth = next_depth
                    else:
                        action = policy(
                            torch.from_numpy(robot.observation(args.command, previous)[None]), torch.from_numpy(depth)
                        ).numpy()[0]
                    robot.step(action)
                    if reference_depth is not None and reference_depth.capture_due(step):
                        reference_depth.capture(robot.depth())
                    previous = np.clip(action, -6, 6)
                if args.collect:
                    frames.append(robot.motion_frame())
                if viewer:
                    viewer.sync()
        if args.collect:
            if args.collect.exists():
                raise FileExistsError(args.collect)
            args.collect.parent.mkdir(parents=True, exist_ok=True)
            # 采集结果本身已使用目标顺序，显式写入顺序以供转换器读取。
            args.collect.write_text(
                json.dumps(
                    dict(
                        FrameDuration=0.02,
                        MotionWeight=1.0,
                        Frames=frames,
                        JointOrder=list(robot.profile.joint_order),
                        Robot=args.robot,
                    )
                )
            )
        print(f"[PASS] MuJoCo {args.robot}: {args.steps} steps")
    finally:
        if viewer:
            viewer.close()
        robot.close()


if __name__ == "__main__":
    main()
