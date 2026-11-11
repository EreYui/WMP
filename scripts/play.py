#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""播放/导出/录制 Go1/Go2 策略。用法：isaaclab.sh -p scripts/play.py --checkpoint model.pt --terrain climb。"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", "--path", type=Path)
parser.add_argument("--robot", choices=("go1", "go2"))
parser.add_argument("--task")
parser.add_argument("--terrain", default=None)
parser.add_argument("--terrain-view", action="store_true")
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--rssm-mode", choices=("sample", "mode"))
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--level", type=int, default=0)
parser.add_argument("--command", nargs=3, type=float, default=(0.6, 0.0, 0.0))
parser.add_argument("--video", type=Path)
parser.add_argument("--export", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.checkpoint is None or args.checkpoint.is_dir():
    search = args.checkpoint or (Path(__file__).resolve().parents[1] / "logs")
    if args.robot and args.checkpoint is None:
        search = search / args.robot
    candidates = list(search.rglob("model_*.pt"))
    if not candidates:
        parser.error(f"no checkpoint found under {search}; train first or pass --checkpoint")
    args.checkpoint = max(candidates, key=lambda path: path.stat().st_mtime_ns)
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import wmp_lab  # noqa: E402,F401
from wmp_lab.learning.deployment import export_policy  # noqa: E402
from wmp_lab.learning.runner import WMPRunner  # noqa: E402
from wmp_lab.robots import (  # noqa: E402
    ENVIRONMENT_REVISION,
    LEGACY_CONTROL_REVISION,
    LEGACY_ENVIRONMENT_REVISION,
    robot_metadata,
)
from wmp_lab.tasks.direct.wmp_env_cfg import make_env_cfg, terrain_config  # noqa: E402
from wmp_lab.learning.progress_rewards import checkpoint_reward_revision  # noqa: E402


def checkpoint_environment_options(runner):
    """回放只认 checkpoint 保存的控制版本，不套用当前机器人的新默认。"""
    contract = getattr(runner, "environment_contract", None) or {}
    saved = contract.get("config", {})
    revision = getattr(runner, "control_revision", LEGACY_CONTROL_REVISION)
    if saved.get("control_revision", revision) != revision:
        raise ValueError("checkpoint control revision disagrees with its environment contract")
    return dict(
        control_revision=revision,
        environment_revision=getattr(runner, "environment_revision", LEGACY_ENVIRONMENT_REVISION),
        camera_implementation=saved.get("camera_implementation", "isaaclab_sparse"),
        reward_revision=checkpoint_reward_revision({"extra": {
            "environment_contract": contract, "environment": getattr(runner, "_resume_environment", None),
        }}),
    )


def main():
    runner = WMPRunner.from_checkpoint(args.checkpoint, args.device)
    robot = args.robot or runner.cfg.robot
    if robot != runner.cfg.robot:
        raise ValueError("checkpoint robot differs from selected robot")
    options = checkpoint_environment_options(runner)
    selected_profile = robot_metadata(robot, options["control_revision"], options["environment_revision"])
    print(
        f"[PLAY] robot={robot}, control_revision={options['control_revision']}, "
        f"actuator={selected_profile['actuator_model']}, "
        f"joint_gear_ratios={selected_profile.get('joint_gear_ratios', [1.0] * 12)}",
        flush=True,
    )
    terrain = "all" if args.terrain_view else args.terrain
    cfg = make_env_cfg(
        robot,
        smoke=args.smoke,
        terrain=terrain,
        num_envs=args.num_envs,
        evaluation=True,
        **options,
    )
    if terrain is None and cfg.environment_revision == ENVIRONMENT_REVISION:
        contract = getattr(runner, "environment_contract", None) or {}
        saved_generator = (contract.get("config", {}).get("terrain") or {}).get("terrain_generator") or {}
        saved_mix = saved_generator.get("sub_terrains")
        if saved_mix:
            available = terrain_config(robot, "all")
            unknown = set(saved_mix) - set(available.sub_terrains)
            if unknown:
                raise ValueError(f"unsupported saved terrain names: {sorted(unknown)}")
            generator = cfg.terrain.terrain_generator
            generator.sub_terrains = {name: available.sub_terrains[name] for name in saved_mix}
            for name, item in saved_mix.items():
                generator.sub_terrains[name].proportion = item["proportion"]
            generator.num_cols = saved_generator["num_cols"]
            generator.num_rows = saved_generator["num_rows"]
    if not hasattr(runner.cfg, "algorithm_revision"):
        cfg.reward_semantics = cfg.curriculum_semantics = "legacy_v2"
    cfg.sim.device = args.device
    cfg.command_x_range = (args.command[0], args.command[0])
    cfg.command_y_range = (args.command[1], args.command[1])
    cfg.command_yaw_range = (args.command[2], args.command[2])
    cfg.jump_command_x_range = cfg.tilt_command_x_range = cfg.command_x_range
    cfg.terrain.max_init_terrain_level = args.level
    task = f"Isaac-WMP-{robot.capitalize()}-Rough-Direct-{'Smoke-v0' if args.smoke else 'v0'}"
    env = gym.make(task, cfg=cfg, render_mode="rgb_array" if args.video else None).unwrapped
    try:
        cfg_rows = cfg.terrain.terrain_generator.num_rows
        if args.level < 0 or args.level >= cfg_rows:
            raise ValueError(f"level must be within [0,{cfg_rows - 1}]")
        env._terrain.terrain_levels[:] = args.level
        env._terrain.env_origins[:] = env._terrain.terrain_origins[
            env._terrain.terrain_levels, env._terrain.terrain_types
        ]
        output = args.export or args.checkpoint.parent / "exported" / f"{robot}_policy.pt"
        path, _ = export_policy(runner, output, batch_size=1)
        # 单机器人部署文件固定 B=1；播放使用独立的向量化模块。
        from wmp_lab.learning.deployment import DeploymentPolicy

        policy = DeploymentPolicy(
            runner, env.num_envs, None if args.rssm_mode is None else args.rssm_mode == "sample"
        ).to(env.device)
        observation = env.reset()[0]["policy"]
        video = None
        if args.video:
            import imageio.v2 as imageio

            args.video.parent.mkdir(parents=True, exist_ok=True)
            video = imageio.get_writer(args.video, fps=50, codec="libx264")
        try:
            with torch.inference_mode():
                for _ in range(args.steps):
                    action = policy(observation[:, 53:98], env.get_depth())
                    observation, _, terminated, truncated, _ = env.step(action)
                    observation = observation["policy"]
                    policy.reset_envs(terminated | truncated)
                    if video or not args.headless:
                        position = env._robot.data.root_pos_w[0].tolist()
                        if args.terrain_view:
                            extent = (
                                max(cfg.terrain.terrain_generator.num_cols, cfg.terrain.terrain_generator.num_rows)
                                * 8.0
                            )
                            env.sim.set_camera_view((extent, extent, extent), (0, 0, 0))
                        else:
                            env.sim.set_camera_view((position[0] + 3, position[1] + 3, position[2] + 2), position)
                    if video:
                        video.append_data(env.render())
            print(f"Played {robot}; exported {path}", flush=True)
        finally:
            if video:
                video.close()
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
