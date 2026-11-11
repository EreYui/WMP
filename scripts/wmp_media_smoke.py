#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""固定步 RTX 深度/视频测试，使用共享 IsaacLab smoke wrapper。"""

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa:E402
import imageio.v2 as imageio  # noqa:E402
import torch  # noqa:E402
import wmp_lab  # noqa:E402,F401
from isaaclab.utils.math import quat_apply  # noqa: E402
from wmp_lab.tasks.direct.wmp_env_cfg import make_env_cfg  # noqa:E402


def main():
    cfg = make_env_cfg("go2", smoke=True, terrain="stool", num_envs=1, evaluation=True)
    cfg.terrain.terrain_generator.num_rows = 1
    cfg.sim.device = args.device
    cfg.depth_backend = "isaac"
    cfg.camera_delay_frames = 0
    env = gym.make("Isaac-WMP-Go2-Rough-Direct-Smoke-v0", cfg=cfg, render_mode="rgb_array").unwrapped
    try:
        env.reset()
        with tempfile.TemporaryDirectory(prefix="wmp_media_") as directory:
            path = Path(directory) / "video.mp4"
            with imageio.get_writer(path, fps=50, codec="libx264") as writer:
                for _ in range(max(args.steps, 5)):
                    env.step(torch.zeros(1, 12, device=env.device))
                    writer.append_data(env.render())
            depth = env.get_depth()
            offset = torch.tensor(cfg.depth_camera.offset.pos, device=env.device).expand(1, -1)
            expected = env._robot.data.root_pos_w + quat_apply(env._robot.data.root_quat_w, offset)
            torch.testing.assert_close(env._depth_camera.data.pos_w, expected, atol=1e-5, rtol=0)
            assert depth.shape == (1, 64, 64, 1) and torch.isfinite(depth).all()
            assert -0.5 <= depth.min() <= depth.max() <= 0.5
            assert path.stat().st_size > 0
        print("[PASS] RTX depth, rgb_array rendering and MP4 recording", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
