#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Train WMP Go2 with the external IsaacLab direct environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "source" / "wmp_go2"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-WMP-Go2-Rough-Direct-v0")
parser.add_argument("--smoke", action="store_true", help="Use the 4-environment low-memory configuration")
parser.add_argument("--iterations", type=int, default=1)
parser.add_argument("--num-envs", type=int, default=None)
parser.add_argument("--checkpoint", type=Path, default=None, help="Resume checkpoint")
parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "logs" / "wmp_go2.pt")
parser.add_argument(
    "--motions",
    type=Path,
    default=REPOSITORY_ROOT / "source" / "wmp_go2" / "wmp_go2" / "data" / "motions",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import wmp_go2  # noqa: E402, F401
from wmp_go2.learning import AMPMotionDataset, WMPRunner  # noqa: E402
from wmp_go2.learning.config import get_learning_cfg  # noqa: E402
from wmp_go2.tasks.direct.go2_env_cfg import WMPGo2RoughEnvCfg, WMPGo2RoughSmokeEnvCfg  # noqa: E402


def main() -> None:
    smoke = args.smoke or args.task.endswith("Smoke-v0")
    task_id = "Isaac-WMP-Go2-Rough-Direct-Smoke-v0" if smoke else args.task
    env_cfg = WMPGo2RoughSmokeEnvCfg() if smoke else WMPGo2RoughEnvCfg()
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    env = gym.make(task_id, cfg=env_cfg).unwrapped
    cfg = get_learning_cfg(smoke)
    runner = WMPRunner(cfg, env.device)
    if args.checkpoint is not None:
        runner.load(args.checkpoint)
        print(f"Resumed iteration {runner.iteration} from {args.checkpoint}; replay buffers will warm up again.")
    expert = AMPMotionDataset(args.motions, env.device)
    try:
        for _ in range(args.iterations):
            metrics = runner.collect_and_update(env, expert)
            print(f"iteration={runner.iteration} " + " ".join(f"{key}={value:.5g}" for key, value in metrics.items()))
            runner.save(args.output)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
