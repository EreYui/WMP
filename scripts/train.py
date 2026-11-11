#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""训练/恢复 Go1/Go2 WMP。用法：isaaclab.sh -p scripts/train.py --robot go1 --headless。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--robot", choices=("go1", "go2"), default=None)
parser.add_argument("--task", default=None)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--terrain", default=None)
parser.add_argument("--iterations", type=int, default=100000)
parser.add_argument("--num-envs", type=int)
parser.add_argument("--camera-envs", type=int)
parser.add_argument("--rows", type=int)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--checkpoint", "--path", type=Path)
parser.add_argument("--output", type=Path)
parser.add_argument("--log-dir", type=Path)
parser.add_argument("--save-interval", type=int, default=1000)
motion_args = parser.add_mutually_exclusive_group()
motion_args.add_argument("--motions", type=Path)
motion_args.add_argument("--amp-set", choices=("upstream", "robot"))
parser.add_argument("--amp-sampling", choices=("upstream_v1", "legacy_uniform"))
parser.add_argument("--amp-preload", type=int)
parser.add_argument("--seed", type=int)
parser.add_argument("--health-max-std", type=float, default=3.0)
parser.add_argument("--health-max-mean", type=float, default=6.0)
parser.add_argument("--control-revision", default=None, help="Fresh runs only; resume keeps the saved revision")
parser.add_argument("--reward-revision", default=None, help="Fresh runs only; resume keeps the saved revision")
parser.add_argument("--camera-revision", default=None, help="Fresh runs only; resume keeps the saved revision")
parser.add_argument("--collision-revision", default=None, help="Fresh runs only; resume keeps the saved revision")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.resume and args.checkpoint is None:
    parser.error("--resume requires --checkpoint/--path")
if args.iterations < 1 or args.save_interval < 1:
    parser.error("iterations and save interval must be positive")
if args.amp_preload is not None and args.amp_preload < 0:
    parser.error("AMP preload must be nonnegative")
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import wmp_lab  # noqa: E402,F401
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
from wmp_lab.learning import AMPMotionDataset, WMPRunner  # noqa: E402
from wmp_lab.learning.config import ALGORITHM_REVISION, LearningCfg, get_learning_cfg  # noqa: E402
from wmp_lab.learning.contracts import environment_contract  # noqa: E402
from wmp_lab.robots import (  # noqa: E402
    CAMERA_REVISION_NATIVE,
    COLLISION_REVISION_NATIVE,
    ENVIRONMENT_REVISION,
    LEGACY_CONTROL_REVISION,
    checkpoint_control_revision,
    default_control_revision,
    robot_metadata,
    robot_profile,
)
from wmp_lab.tasks.direct.wmp_env_cfg import make_env_cfg, terrain_config  # noqa: E402
from wmp_lab.learning.progress_rewards import checkpoint_reward_revision  # noqa: E402


def require_training_schedule(cfg, *, smoke):
    cfg.validate_schedule()
    if not smoke and not cfg.aligned_schedule:
        raise ValueError("diagnostic checkpoints support playback only; start a fresh aligned training run")


def resolve_training_control_revision(robot, payload, requested=None):
    """新实验按机器人或显式参数选版本；续训严格保留保存的版本与完整契约。"""
    revision = checkpoint_control_revision(payload) if payload else (requested or default_control_revision(robot))
    if payload and requested is not None and requested != revision:
        raise ValueError("resume cannot change control revision; start a separate experiment")
    if revision == LEGACY_CONTROL_REVISION:
        raise ValueError("legacy actuator checkpoints support playback only; start a fresh training run")
    robot_profile(robot, control_revision=revision)
    if payload:
        extra = payload.get("extra", {})
        full_config = (extra.get("environment_contract") or {}).get("config", {})
        mini_config = (extra.get("environment") or {}).get("config", {})
        if any(options.get("control_revision", revision) != revision for options in (full_config, mini_config)):
            raise ValueError("checkpoint control revision disagrees with its environment contract")
    return revision


def main():
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.checkpoint else None
    if payload and payload.get("version") != 4:
        raise ValueError("historical checkpoints support playback only; start a new reference training run")
    if payload and payload["config"].get("algorithm_revision") != ALGORITHM_REVISION:
        raise ValueError("historical update schedule supports playback only; start a fresh gpufree alignment run")
    saved_amp = (payload or {}).get("extra", {}).get("amp_dataset")
    if payload and not saved_amp:
        raise ValueError("checkpoint has no fixed AMP contract; use playback or start a fresh experiment")
    robot = args.robot or (
        payload["config"]["robot"] if payload else ("go2" if args.task and "-Go2-" in args.task else "go1")
    )
    if args.task and (f"-{robot.capitalize()}-" not in args.task):
        raise ValueError("task and robot selection disagree")
    smoke = (
        args.smoke
        or bool(args.task and args.task.endswith("Smoke-v0"))
        or bool(payload and payload["config"]["world_hidden"] <= 64)
    )
    control_revision = resolve_training_control_revision(robot, payload, args.control_revision)
    saved_environment_revision = (payload or {}).get("extra", {}).get("environment_revision", ENVIRONMENT_REVISION)
    reward_revision = checkpoint_reward_revision(payload) if payload else args.reward_revision
    if payload and args.reward_revision is not None and args.reward_revision != reward_revision:
        raise ValueError("resume cannot change reward revision; start a separate experiment")
    saved_camera_revision = (
        ((payload or {}).get("extra", {}).get("environment_contract") or {}).get("config", {}).get("camera_revision")
        or ((payload or {}).get("extra", {}).get("environment") or {}).get("config", {}).get("camera_revision")
        or CAMERA_REVISION_NATIVE
    ) if payload else (args.camera_revision or CAMERA_REVISION_NATIVE)
    if payload and args.camera_revision is not None and args.camera_revision != saved_camera_revision:
        raise ValueError("resume cannot change camera revision; start a separate experiment")
    camera_revision = saved_camera_revision
    saved_collision_revision = (
        ((payload or {}).get("extra", {}).get("environment_contract") or {}).get("config", {}).get("collision_revision")
        or ((payload or {}).get("extra", {}).get("environment") or {}).get("config", {}).get("collision_revision")
        or COLLISION_REVISION_NATIVE
    ) if payload else (args.collision_revision or COLLISION_REVISION_NATIVE)
    if payload and args.collision_revision is not None and args.collision_revision != saved_collision_revision:
        raise ValueError("resume cannot change collision revision; start a separate experiment")
    collision_revision = saved_collision_revision
    env_cfg = make_env_cfg(
        robot, smoke=smoke, terrain=args.terrain, num_envs=args.num_envs,
        control_revision=control_revision, environment_revision=saved_environment_revision,
        reward_revision=reward_revision, camera_revision=camera_revision,
        collision_revision=collision_revision,
    )
    saved_env = (payload or {}).get("extra", {}).get("environment") or {}
    saved_options = saved_env.get("config", {})
    if payload:
        env_cfg.camera_implementation = saved_options.get(
            "camera_implementation",
            (payload.get("extra", {}).get("environment_contract") or {})
            .get("config", {})
            .get("camera_implementation", "isaaclab_sparse"),
        )
    if args.seed is None:
        args.seed = saved_amp["seed"] if saved_amp else 1
    if saved_options and args.terrain is None:
        available = terrain_config(robot, "all")
        env_cfg.terrain.terrain_generator.sub_terrains = {
            name: available.sub_terrains[name] for name in saved_options["terrain_proportions"]
        }
        for name, proportion in saved_options["terrain_proportions"].items():
            env_cfg.terrain.terrain_generator.sub_terrains[name].proportion = proportion
        env_cfg.terrain.terrain_generator.num_cols = saved_options["num_cols"]
        env_cfg.terrain.terrain_generator.num_rows = saved_options["num_rows"]
    if saved_options and args.num_envs is None:
        env_cfg.scene.num_envs = saved_options["num_envs"]
    if saved_options and args.camera_envs is None:
        env_cfg.logical_camera_envs = saved_options["logical_camera_envs"]
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    if args.rows is not None:
        env_cfg.terrain.terrain_generator.num_rows = args.rows
    if args.camera_envs is not None:
        env_cfg.logical_camera_envs = args.camera_envs
    cfg = (
        LearningCfg(**payload["config"])
        if payload
        else replace(get_learning_cfg(smoke, num_envs=env_cfg.scene.num_envs), robot=robot)
    )
    if payload and cfg.num_envs != env_cfg.scene.num_envs:
        raise ValueError("resume cannot change physical environment count; start a separate experiment")
    cfg = replace(cfg, num_envs=env_cfg.scene.num_envs)
    require_training_schedule(cfg, smoke=smoke)
    if cfg.robot != robot:
        raise ValueError("checkpoint robot differs from selected robot")
    print(
        f"[CONFIG] envs={cfg.num_envs}, control_revision={env_cfg.control_revision}, "
        f"reward_revision={env_cfg.reward_revision}, camera_revision={env_cfg.camera_revision}, "
        f"collision_revision={env_cfg.collision_revision}, "
        f"camera_pos={env_cfg.depth_camera.offset.pos}, "
        f"collection={cfg.collection_chunks}x{cfg.rollout_steps}, "
        f"batch_samples={cfg.batch_samples}, gae_horizon={cfg.rollout_steps}, "
        f"world_updates={cfg.world_train_steps}, depth_updates={cfg.depth_train_steps}, "
        f"ppo_epochs={cfg.learning_epochs}, minibatches={cfg.mini_batches}, "
        f"ppo_micro_batch={cfg.ppo_micro_batch_size}; iteration means one complete update batch",
        flush=True,
    )
    selected_robot_profile = robot_metadata(
        robot, env_cfg.control_revision, env_cfg.environment_revision, camera_revision=env_cfg.camera_revision
    )
    print(
        f"[ACTUATION] model={selected_robot_profile['actuator_model']}, "
        f"joint_gear_ratios={selected_robot_profile.get('joint_gear_ratios', [1.0] * 12)}",
        flush=True,
    )
    task = f"Isaac-WMP-{robot.capitalize()}-Rough-Direct-{'Smoke-v0' if smoke else 'v0'}"
    data_root = Path(wmp_lab.__file__).parent / "data"
    if args.motions:
        motion_dir = args.motions
    elif args.amp_set:
        motion_dir = data_root / ("motions_upstream" if args.amp_set == "upstream" else f"motions_{robot}")
    elif saved_amp:
        motion_dir = saved_amp["source"]
    else:
        motion_dir = data_root / "motions_upstream"
    sampling = args.amp_sampling or (saved_amp["sampling_mode"] if saved_amp else "upstream_v1")
    preload = (
        args.amp_preload
        if args.amp_preload is not None
        else (
            saved_amp["preload"] if saved_amp else (0 if sampling == "legacy_uniform" else (64 if smoke else 2_000_000))
        )
    )
    expert = AMPMotionDataset(
        motion_dir,
        args.device,
        seed=args.seed,
        sampling_mode=sampling,
        preload=preload,
        dt=env_cfg.sim.dt * env_cfg.decimation,
    )
    if saved_amp and saved_amp["fingerprint"] != expert.contract["fingerprint"]:
        raise ValueError("AMP files, order or sampling parameters differ from checkpoint; use a separate experiment")
    env = gym.make(task, cfg=env_cfg).unwrapped
    if cfg.aligned_schedule:
        env.configure_training_schedule(cfg.reference_num_envs)
    log_dir = args.log_dir or ROOT / "logs" / robot / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    runner = WMPRunner(cfg, env.device)
    if not args.checkpoint:
        runner.control_revision = env_cfg.control_revision
        runner.environment_revision = env_cfg.environment_revision
    if args.checkpoint:
        runner.load(args.checkpoint)
        print(f"Resumed iteration {runner.iteration}; sequence replay will warm up again.", flush=True)
    runner.bind_expert(expert)
    current_contract = environment_contract(env.cfg)
    if runner.environment_contract and runner.environment_contract["fingerprint"] != current_contract["fingerprint"]:
        raise ValueError("environment/source/asset contract differs from checkpoint; use a separate experiment")
    runner.environment_contract = current_contract
    if runner._resume_environment is None:
        runner._resume_environment = env.curriculum_state_dict()
    (log_dir / "run.json").write_text(
        json.dumps(
            dict(
                robot=robot,
                control_revision=env_cfg.control_revision,
                robot_profile=selected_robot_profile,
                environment_revision=env_cfg.environment_revision,
                environment_contract=current_contract,
                task=task,
                args={k: str(v) for k, v in vars(args).items()},
                learning=runner.cfg.to_dict(),
                motions=str(motion_dir),
                amp_dataset=expert.contract,
                source_sha256={
                    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (ROOT / "source/wmp_lab/wmp_lab").rglob("*.py")
                },
            ),
            indent=2,
        )
    )
    try:
        if not args.checkpoint:
            runner.save(log_dir / "model_0.pt")
        with (log_dir / "metrics.jsonl").open("a") as stream:
            for _ in range(args.iterations):
                started = time.monotonic()
                metrics = runner.collect_and_update(env, expert)
                metrics["iteration_seconds"] = time.monotonic() - started
                row = dict(iteration=runner.iteration, total_steps=runner.total_steps, **metrics)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                for name, value in metrics.items():
                    writer.add_scalar(name, value, runner.iteration)
                print(
                    f"iteration={runner.iteration} "
                    + " ".join(f"{k}={v:.5g}" for k, v in metrics.items() if not k.startswith("action_")),
                    flush=True,
                )
                max_std = max(v for k, v in metrics.items() if k.startswith("action_std/"))
                max_mean = max(v for k, v in metrics.items() if k.startswith("action_mean_abs/"))
                if max_std > args.health_max_std or max_mean > args.health_max_mean:
                    runner.save(log_dir / f"model_unhealthy_{runner.iteration}.pt")
                    writer.flush()
                    raise FloatingPointError(f"policy health guard: std={max_std:.3f}, mean_abs={max_mean:.3f}")
                if runner.iteration % args.save_interval == 0:
                    runner.save(log_dir / f"model_{runner.iteration}.pt")
                    writer.flush()
            runner.save(log_dir / f"model_{runner.iteration}.pt")
            if args.output:
                runner.save(args.output)
            print(f"Saved run to {log_dir}", flush=True)
    finally:
        writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
