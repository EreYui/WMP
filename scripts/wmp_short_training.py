#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""共享wrapper启动的有界Go2正式训练；仅旁路监控，不改策略/环境计算。"""

import argparse
import json
import os
import runpy
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=200, help="Maximum training iterations")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--checkpoint", type=Path, default=os.environ.get("WMP_TRAIN_CHECKPOINT"))
parser.add_argument("--save-interval", type=int, default=int(os.environ.get("WMP_TRAIN_SAVE_INTERVAL", "25")))
parser.add_argument("--num-envs", type=int, default=int(os.environ.get("WMP_SHORT_NUM_ENVS", "2048")))
parser.add_argument("--camera-envs", type=int, default=int(os.environ.get("WMP_SHORT_CAMERA_ENVS", "550")))
args = parser.parse_args()
if args.num_envs < 1 or not 1 <= args.camera_envs <= args.num_envs:
    parser.error("require num_envs >= camera_envs >= 1")
log_dir = Path(os.environ.get("WMP_SHORT_TRAIN_DIR", ROOT / f"logs/go2/short_{args.num_envs}_{args.camera_envs}"))
if (log_dir / "run.json").exists():
    parser.error("log directory already contains a run; choose a new WMP_SHORT_TRAIN_DIR")
log_dir.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "tools"))
from resource_memory import memory_report


def main():
    sys.argv = [
        str(ROOT / "scripts/train.py"),
        "--robot",
        os.environ.get("WMP_ROBOT", "go2"),
        "--num-envs",
        str(args.num_envs),
        "--camera-envs",
        str(args.camera_envs),
        "--iterations",
        str(args.steps),
        "--save-interval",
        str(args.save_interval),
        "--amp-set",
        "upstream",
        "--seed",
        "1",
        "--log-dir",
        str(log_dir),
        "--device",
        args.device,
        "--headless",
    ]
    for option, variable in (
        ("--control-revision", "WMP_CONTROL_REVISION"),
        ("--reward-revision", "WMP_REWARD_REVISION"),
        ("--camera-revision", "WMP_CAMERA_REVISION"),
        ("--collision-revision", "WMP_COLLISION_REVISION"),
    ):
        if os.environ.get(variable):
            sys.argv.extend((option, os.environ[variable]))
    if args.checkpoint is not None:
        sys.argv.extend(("--resume", "--checkpoint", str(args.checkpoint)))
    namespace = runpy.run_path(str(ROOT / "scripts/train.py"), run_name="wmp_bounded_training")
    import psutil
    import torch
    from training_monitor import EpisodeMonitor

    runner_class = namespace["WMPRunner"]
    original_collect = runner_class.collect_and_update
    monitors, reward_sums = {}, {}
    stop = threading.Event()
    started = time.time()
    learning = namespace["get_learning_cfg"](num_envs=args.num_envs)
    state = dict(
        pid=os.getpid(),
        started_unix=started,
        iterations=args.steps,
        num_envs=args.num_envs,
        camera_envs=args.camera_envs,
        world_updates=learning.world_train_steps,
        collection_chunks=learning.collection_chunks,
        collection_steps=learning.collection_steps,
        batch_samples=learning.batch_samples,
        iteration_unit="complete_update_batch",
        depth_updates=learning.depth_train_steps,
        replay_mode="episodes",
        resume_checkpoint=str(args.checkpoint) if args.checkpoint is not None else None,
        save_interval=args.save_interval,
        status="running",
    )
    (log_dir / "monitor_state.json").write_text(json.dumps(state, indent=2) + "\n")

    def resources():
        with (log_dir / "resources.jsonl").open("a") as stream:
            while not stop.is_set():
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        check=True,
                    )
                    used, total, utilization, temperature = map(float, result.stdout.splitlines()[0].split(","))
                    row = dict(
                        unix_time=time.time(),
                        elapsed_s=time.time() - started,
                        gpu_used_mib=used,
                        gpu_total_mib=total,
                        gpu_utilization=utilization,
                        gpu_temperature_c=temperature,
                        cpu_rss_mib=psutil.Process().memory_info().rss / 1024**2,
                        host_available_mib=psutil.virtual_memory().available / 1024**2,
                        swap_used_mib=psutil.swap_memory().used / 1024**2,
                    )
                    row.update(memory_report())
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                except Exception as error:
                    print(f"[MONITOR] resource poll failed: {error}", flush=True)
                stop.wait(1.0)

    def collect(runner, env, expert, rollout_steps=None):
        if id(env) not in monitors:
            monitor = monitors[id(env)] = EpisodeMonitor(env.num_envs, env.step_dt, env.device)
            original_step = env.step
            reward_sums[id(env)] = {}

            def step(actions):
                output = original_step(actions)
                if runner._env is not env:
                    return output  # reset 的零动作预热不计入正式采集批次。
                _, task, terminated, truncated, _ = output
                monitor.update(task, terminated, truncated, getattr(env, "_unclipped_task_reward", None))
                for name, values in getattr(env, "_reward_terms", {}).items():
                    value = values.detach().mean() * env.step_dt
                    reward_sums[id(env)][name] = reward_sums[id(env)].get(name, 0) + value
                return output

            env.step = step
        reward_sums[id(env)] = {}
        metrics = original_collect(runner, env, expert, rollout_steps)
        metrics.update(monitors[id(env)].metrics())
        for name, value in reward_sums[id(env)].items():
            metrics["task_reward_component/" + name] = float(value) / metrics["collection_steps"]
        levels = env.curriculum_levels.float()
        metrics.update(
            terrain_level_max=float(levels.max()),
            terrain_level_ge1_fraction=float((levels >= 1).float().mean()),
            terrain_level_ge3_fraction=float((levels >= 3).float().mean()),
            gpu_allocated_mib=torch.cuda.memory_allocated() / 1024**2,
            gpu_reserved_mib=torch.cuda.memory_reserved() / 1024**2,
            gpu_peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
            gpu_peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
            host_available_mib=psutil.virtual_memory().available / 1024**2,
            cpu_rss_mib=psutil.Process().memory_info().rss / 1024**2,
            training_elapsed_s=time.time() - started,
        )
        metrics.update(memory_report())
        print(
            f"[MONITOR] iteration={runner.iteration} task={metrics['task_return_mean']:.5f} "
            f"survival_s={metrics.get('episode_length_mean_s', 0):.3f} level={float(levels.mean()):.3f} "
            f"host_available_mib={metrics['host_available_mib']:.0f}",
            flush=True,
        )
        if metrics["host_available_mib"] < 2048:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            metrics.update(memory_report())
        if metrics["host_available_mib"] < 1024:
            runner.save(log_dir / f"model_host_memory_guard_{runner.iteration}.pt")
            raise MemoryError("host available memory is below 1 GiB; checkpoint saved")
        return metrics

    runner_class.collect_and_update = collect
    monitor_thread = threading.Thread(target=resources, daemon=True)
    monitor_thread.start()
    try:
        namespace["main"]()
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stop.set()
        monitor_thread.join(timeout=6)
        state.update(finished_unix=time.time(), elapsed_s=time.time() - started)
        (log_dir / "monitor_state.json").write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
