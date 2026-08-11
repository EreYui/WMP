#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run deterministic WMP Go2 checkpoint inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "source" / "wmp_go2"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-WMP-Go2-Rough-Direct-Smoke-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--num-envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import wmp_go2  # noqa: E402, F401
from wmp_go2.constants import ACTION_DIM, WORLD_MODEL_ACTION_HISTORY_STEPS  # noqa: E402
from wmp_go2.learning.config import get_learning_cfg  # noqa: E402
from wmp_go2.learning.observations import HistoryBuffer, proprioception  # noqa: E402
from wmp_go2.learning.runner import WMPRunner  # noqa: E402
from wmp_go2.tasks.direct.go2_env_cfg import WMPGo2RoughEnvCfg, WMPGo2RoughSmokeEnvCfg  # noqa: E402


def main() -> None:
    smoke = args.task.endswith("Smoke-v0")
    cfg = WMPGo2RoughSmokeEnvCfg() if smoke else WMPGo2RoughEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = gym.make(args.task, cfg=cfg).unwrapped
    runner = WMPRunner(get_learning_cfg(smoke), env.device)
    runner.load(args.checkpoint, load_optimizers=False)
    runner.modules["policy_value"].eval()
    observation, _ = env.reset()
    observation = observation["policy"]
    history_buffer = HistoryBuffer(env.num_envs, env.device)
    history = history_buffer.append(observation)
    action_history = torch.zeros(env.num_envs, WORLD_MODEL_ACTION_HISTORY_STEPS, ACTION_DIM, device=env.device)
    is_first = torch.ones(env.num_envs, device=env.device)
    state = None
    try:
        for _ in range(args.steps):
            with torch.inference_mode():
                prop = proprioception(observation)
                depth, _, _ = runner._mixed_depth(env, prop)
                state, _ = runner.world_model.observe_step(prop, depth, action_history.flatten(1), is_first, state)
                feature = runner.world_model.policy_feature(state)
                action = runner.policy.act_inference(observation, history, feature)
            observation, _, terminated, truncated, _ = env.step(action)
            observation = observation["policy"]
            done = terminated | truncated
            action_history = torch.cat((action_history[:, 1:], action.unsqueeze(1)), dim=1)
            history_buffer.reset(done.nonzero(as_tuple=False).flatten())
            history = history_buffer.append(observation)
            is_first = done.float()
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
