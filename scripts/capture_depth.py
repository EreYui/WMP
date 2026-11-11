#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""采集 IsaacLab 深度和网络输入。

用法：isaaclab.sh -p scripts/capture_depth.py --robot go1 --terrain stool --headless --output depth.npz。"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--robot", choices=("go1", "go2"), default="go2")
parser.add_argument("--terrain", default="climb")
parser.add_argument("--backend", choices=("warp", "isaac"), default="warp")
parser.add_argument("--output", required=True, type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.backend == "isaac":
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa:E402
import numpy as np  # noqa:E402
import torch  # noqa:E402
import wmp_lab  # noqa:E402,F401
from wmp_lab.tasks.direct.wmp_env_cfg import make_env_cfg  # noqa:E402

try:
    cfg = make_env_cfg(args.robot, smoke=True, terrain=args.terrain, num_envs=1, evaluation=True)
    cfg.terrain.terrain_generator.num_rows = 1
    cfg.sim.device = args.device
    cfg.depth_backend = args.backend
    cfg.camera_delay_frames = 0
    cfg.command_x_range = cfg.command_y_range = cfg.command_yaw_range = (0.0, 0.0)
    cfg.jump_command_x_range = cfg.tilt_command_x_range = (0.0, 0.0)
    env = gym.make(f"Isaac-WMP-{args.robot.capitalize()}-Rough-Direct-Smoke-v0", cfg=cfg).unwrapped
    try:
        observation = env.reset()[0]["policy"]
        for _ in range(5):
            observation = env.step(torch.zeros(1, 12, device=env.device))[0]["policy"]
        # 对齐最终机器人位姿与渲染/射线采样时刻，避免比较旧 RTX 帧。
        if args.backend == "isaac":
            env.sim.render()
        env._depth_camera.update(env.step_dt, force_recompute=True)
        env._depth_frame_step = -1
        depth = env.get_depth().cpu().numpy()
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output,
            depth_m=(depth + 0.5) * 2,
            depth_normalized=depth,
            observation=observation.cpu().numpy(),
            forward_height_map=env.get_forward_height_map().cpu().numpy(),
            backend=args.backend,
            robot=args.robot,
            intrinsics=env._depth_camera.data.intrinsic_matrices.cpu().numpy(),
            camera_position=env._depth_camera.data.pos_w.cpu().numpy(),
            camera_quaternion=env._depth_camera.data.quat_w_world.cpu().numpy(),
            root_pose=env._robot.data.root_state_w[:, :7].cpu().numpy(),
        )
        print(f"Captured {args.output}", flush=True)
    finally:
        env.close()
finally:
    app.close()
