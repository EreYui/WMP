#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Go1/Go2 环境规模与 AMP 数据的真实模拟器检查。

用法：WMP_BENCH_ENVS=64 ISAACLAB_SMOKE_SCRIPT=$PWD/scripts/benchmark_go2.py
      ISAACLAB_SMOKE_STEPS=24 /home/re/IsaacLab/tools/run_codex_isaaclab_smoke.sh。
WMP_BENCH_WARM=1 预热到64帧后执行完整默认训练更新。
"""

import argparse
import json
import os
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=24)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import wmp_lab  # noqa: E402,F401
from wmp_lab.learning import AMPMotionDataset, WMPRunner  # noqa: E402
from wmp_lab.learning.config import PAPER_CFG, get_learning_cfg  # noqa: E402
from wmp_lab.tasks.direct.wmp_env_cfg import make_env_cfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
robot_name = os.environ.get("WMP_BENCH_ROBOT", "go2")
num_envs = int(os.environ.get("WMP_BENCH_ENVS", "64"))
num_cameras = int(os.environ.get("WMP_BENCH_CAMERAS", str(min(num_envs, 1100))))
rows = int(os.environ.get("WMP_BENCH_ROWS", "10"))
warm = os.environ.get("WMP_BENCH_WARM") == "1"
output = Path(os.environ.get("WMP_BENCH_OUTPUT", str(ROOT / "logs/benchmarks" / f"{robot_name}_{num_envs}.json")))
result = dict(
    robot=robot_name, num_envs=num_envs, requested_cameras=num_cameras, terrain_rows=rows, warm=warm, passed=False
)
phase = "startup"
stop = threading.Event()
usage = []


def memory_monitor():
    while not stop.wait(0.2):
        try:
            free, total = torch.cuda.mem_get_info()
            usage.append((total - free) / 1024**2)
        except Exception:
            return


def main():
    global phase
    torch.manual_seed(7)
    thread = threading.Thread(target=memory_monitor, daemon=True)
    thread.start()
    torch.cuda.reset_peak_memory_stats()
    phase = "environment"
    started = time.monotonic()
    cfg = make_env_cfg(robot_name, num_envs=num_envs)
    cfg.terrain.terrain_generator.num_rows = rows
    cfg.logical_camera_envs = num_cameras
    cfg.sim.device = args.device
    env = gym.make(f"Isaac-WMP-{robot_name.capitalize()}-Rough-Direct-v0", cfg=cfg).unwrapped
    result["environment_seconds"] = time.monotonic() - started
    result["real_cameras"] = int(env.camera_mask.sum())
    result["camera"] = env._depth_camera.buffer_report()
    result["supervision_cameras"] = int(env.depth_supervision_mask.sum())
    result["terrain_types"] = sorted(set(env.terrain_names))
    result["joint_names"] = [env._robot.joint_names[index] for index in env._joint_ids]
    phase = "motions"
    shared_amp = os.environ.get("WMP_MOTION_SET") != "robot"
    data_root = Path(wmp_lab.__file__).parent / "data" / ("motions_upstream" if shared_amp else f"motions_{robot_name}")
    manifest = json.loads((data_root / "manifest.json").read_text())
    assert result["joint_names"] == manifest["joint_order"]
    limits = env._robot.data.joint_pos_limits[0, env._joint_ids].cpu().numpy()
    clips = []
    for entry in manifest["sequences"]:
        with np.load(data_root / entry["file"]) as data:
            assert all(np.isfinite(data[key]).all() for key in data.files)
            assert data["joint_pos"].shape == data["joint_vel"].shape == (entry["frames"], 12)
            assert np.allclose(np.linalg.norm(data["root_quat_wxyz"], axis=1), 1, atol=1e-4)
            outside = (data["joint_pos"] < limits[:, 0]) | (data["joint_pos"] > limits[:, 1])
            clips.append(dict(name=entry["name"], frames=entry["frames"], outside_joint_limits=int(outside.sum())))
    expert = AMPMotionDataset(data_root, env.device, seed=7, sampling_mode="upstream_v1", preload=2_000_000)
    states, next_states, ids = expert.sample_transitions(8192)
    assert states.shape == next_states.shape == (8192, 30)
    assert torch.isfinite(states).all() and torch.isfinite(next_states).all()
    assert torch.unique(ids).numel() == len(clips) == (9 if shared_amp else (19 if robot_name == "go1" else 17))
    result["motions"] = dict(clips=clips, sampled_transitions=8192, sampled_all_clips=True, contract=expert.contract)
    print(f"[BENCH] envs={num_envs}, cameras={result['real_cameras']}, validated motions={len(clips)}", flush=True)
    phase = "runner"
    learning = replace(
        PAPER_CFG,
        reference_num_envs=None,
        robot=robot_name,
        replay_mode="ring",
        num_envs=num_envs,
        world_train_start=1,
        depth_train_steps=1,
        world_train_steps=1,
    )
    runner = WMPRunner(learning, env.device)
    result["network_config"] = runner.cfg.to_dict()
    phase = "short_update"
    started = time.monotonic()
    metrics = runner.collect_and_update(env, expert, rollout_steps=args.steps)
    torch.cuda.synchronize()
    result["short_update_seconds"] = time.monotonic() - started
    result["short_update_metrics"] = metrics
    assert {"discriminator_loss", "world_loss", "depth_loss"} <= metrics.keys()
    if warm:
        phase = "replay_warmup"
        runner.cfg = replace(learning, world_train_start=10**12)
        warmup_started = time.monotonic()
        while runner.world_replay.size < PAPER_CFG.world_sequence_length:
            runner.collect_and_update(env, expert, rollout_steps=args.steps)
            print(f"[BENCH] warmed {runner.world_replay.size} world frames", flush=True)
        result["warmup_seconds"] = time.monotonic() - warmup_started
        result["world_sequence_frames"] = runner.world_replay.size
        runner.cfg = replace(
            get_learning_cfg(num_envs=num_envs), robot=robot_name, reference_num_envs=None,
            replay_mode="ring", world_train_start=1
        )
        phase = "full_update"
        started = time.monotonic()
        metrics = runner.collect_and_update(env, expert)
        torch.cuda.synchronize()
        result["full_update_seconds"] = time.monotonic() - started
        result["full_update_metrics"] = metrics
        result["full_update_config"] = runner.cfg.to_dict()
        assert metrics["world_optimizer_steps"] == runner.cfg.world_train_steps
        assert metrics["depth_optimizer_steps"] == runner.cfg.depth_train_steps
        assert metrics["ppo_optimizer_steps"] == runner.cfg.learning_epochs * runner.cfg.mini_batches
        phase = "checkpoint"
        checkpoint = runner.save(output.with_suffix(".pt"))
        result["checkpoint"] = str(checkpoint)
    result["iterations"] = runner.iteration
    result["total_steps"] = runner.total_steps
    result["passed"] = True
    print(f"[PASS] {robot_name} scale {num_envs}: PPO/AMP/depth/RSSM update", flush=True)


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException as error:
        result["failed_phase"] = phase
        result["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        code = 1
    finally:
        stop.set()
        result["gpu_device_peak_mib"] = max(usage, default=0)
        result["torch_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        result["torch_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[BENCH] saved {output}", flush=True)
        # 与共享 smoke 一致，此工作站 Kit 的正常退出可能挂起。
        os._exit(code)
