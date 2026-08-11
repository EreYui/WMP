#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Real CUDA/Isaac Sim smoke accepted by ``run_codex_isaaclab_smoke.sh``."""

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "source" / "wmp_go2"))

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import wmp_go2  # noqa: E402, F401
from wmp_go2.constants import AMP_OBSERVATION_DIM, FORWARD_HEIGHT_DIM, OBSERVATION_DIM  # noqa: E402
from wmp_go2.learning import AMPMotionDataset, WMPRunner  # noqa: E402
from wmp_go2.learning.config import SMOKE_CFG  # noqa: E402
from wmp_go2.learning.motions import convert_motions  # noqa: E402
from wmp_go2.tasks.direct.go2_env_cfg import WMPGo2RoughSmokeEnvCfg  # noqa: E402


def assert_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise RuntimeError(f"{name} contains non-finite values")


def main() -> None:
    print("[SMOKE] preparing motions", flush=True)
    motion_dir = REPOSITORY_ROOT / "source" / "wmp_go2" / "wmp_go2" / "data" / "motions"
    if not (motion_dir / "manifest.json").is_file():
        convert_motions(sorted((REPOSITORY_ROOT / "WMP" / "datasets" / "mocap_motions").glob("*.txt")), motion_dir)
    cfg = WMPGo2RoughSmokeEnvCfg()
    print("[SMOKE] creating environment", flush=True)
    env = gym.make("Isaac-WMP-Go2-Rough-Direct-Smoke-v0", cfg=cfg).unwrapped
    print("[SMOKE] creating learning stack", flush=True)
    expert = AMPMotionDataset(motion_dir, env.device)
    runner = WMPRunner(SMOKE_CFG, env.device)
    try:
        print("[SMOKE] reset and sensor checks", flush=True)
        observation, _ = env.reset()
        policy_obs = observation["policy"]
        if policy_obs.shape != (4, OBSERVATION_DIM):
            raise RuntimeError(f"unexpected observation shape {policy_obs.shape}")
        if env.get_forward_height_map().shape != (4, FORWARD_HEIGHT_DIM):
            raise RuntimeError("forward ray-caster shape mismatch")
        if env.get_amp_observations().shape != (4, AMP_OBSERVATION_DIM):
            raise RuntimeError("AMP observation shape mismatch")
        if env.camera_mask.sum().item() != 2 or (~env.camera_mask).sum().item() != 2:
            raise RuntimeError("real/predicted depth logical branches are not both active")
        assert_finite("observation", policy_obs)
        assert_finite("depth", env.get_depth())
        env.episode_length_buf[0] = env.max_episode_length - 1
        reset_observation, _, _, truncated, _ = env.step(torch.zeros(4, 12, device=env.device))
        if not truncated[0]:
            raise RuntimeError("forced timeout did not execute the termination/reset branch")
        assert_finite("post-reset observation", reset_observation["policy"])
        print("[SMOKE] joint rollout/update", flush=True)
        metrics = runner.collect_and_update(env, expert, rollout_steps=max(4, args.steps))
        if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise RuntimeError("joint update produced non-finite metrics")
        with tempfile.TemporaryDirectory(prefix="wmp_go2_smoke_") as directory:
            print("[SMOKE] checkpoint roundtrip", flush=True)
            checkpoint = runner.save(Path(directory) / "checkpoint.pt")
            clone = WMPRunner(SMOKE_CFG, env.device)
            clone.load(checkpoint)
            torch.manual_seed(7)
            observation = torch.randn(2, OBSERVATION_DIM, device=env.device)
            history = torch.randn(2, 210, device=env.device)
            feature = torch.randn(2, SMOKE_CFG.world_feature_dim, device=env.device)
            with torch.inference_mode():
                expected = runner.policy.act_inference(observation, history, feature)
                actual = clone.policy.act_inference(observation, history, feature)
            torch.testing.assert_close(expected, actual, rtol=0.0, atol=0.0)
        print("[PASS] WMP Go2 real simulator smoke")
        print("[METRICS] " + " ".join(f"{key}={value:.6g}" for key, value in metrics.items()))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
