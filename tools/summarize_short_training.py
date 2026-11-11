#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""读取有界训练的原始日志，生成窗口统计和奖励/存活/课程/内存曲线。"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_rows(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    rows = read_rows(args.path / "metrics.jsonl")
    resources = read_rows(args.path / "resources.jsonl")
    if not rows:
        raise SystemExit("No completed training iterations")
    keys = (
        "task_return_mean",
        "episode_length_mean_s",
        "episode_length_median_s",
        "episode_length_p90_s",
        "episode_task_return_mean",
        "episode_unclipped_task_return_mean",
        "episode_timeout_fraction",
        "terrain_level_mean",
        "terrain_level_max",
        "terrain_level_ge1_fraction",
        "done_fraction",
        "world_loss",
    )

    def statistics(group):
        return {
            key: float(np.mean([row[key] for row in group if key in row]))
            for key in keys
            if any(key in row for row in group)
        }

    summary = dict(
        iterations=len(rows),
        last_iteration=rows[-1]["iteration"],
        total_steps=rows[-1]["total_steps"],
        first_20=statistics(rows[:20]),
        last_20=statistics(rows[-20:]),
        last=rows[-1],
        model_training_started_iteration=next(
            (r["iteration"] for r in rows if r.get("world_optimizer_steps", 0) > 0), None
        ),
        max_action_std=max(v for row in rows for k, v in row.items() if k.startswith("action_std/")),
        max_action_saturation=max(v for row in rows for k, v in row.items() if k.startswith("action_saturation/")),
        gpu_peak_device_gib=max((r["gpu_used_mib"] for r in resources), default=0) / 1024,
        min_host_available_gib=min((r["host_available_mib"] for r in resources), default=0) / 1024,
        peak_cpu_rss_gib=max((r["cpu_rss_mib"] for r in resources), default=0) / 1024,
        final_terrain_levels={k: v for k, v in rows[-1].items() if k.startswith("terrain_level_")},
    )
    (args.path / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    iterations = np.array([r["iteration"] for r in rows])

    def series(key):
        return np.array([r.get(key, np.nan) for r in rows])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    ax = axes[0, 0]
    ax.plot(iterations, series("task_return_mean"), color="#207060", alpha=0.4, label="Per-iteration task return")
    width = min(10, len(rows))
    average = np.convolve(series("task_return_mean"), np.ones(width) / width, mode="valid")
    ax.plot(iterations[width - 1 :], average, color="#207060", label=f"{width}-iteration mean")
    ax.set(title="Task reward (clipped, 24 steps)", xlabel="Iteration", ylabel="Return")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for key, label in (
        ("episode_length_mean_s", "Mean"),
        ("episode_length_median_s", "Median"),
        ("episode_length_p90_s", "P90"),
    ):
        ax.plot(iterations, series(key), label=label)
    ax.set(title="Completed episode survival (window=4096)", xlabel="Iteration", ylabel="Seconds")
    ax.legend(fontsize=8)
    ax = axes[0, 2]
    for key in (
        "terrain_level_mean",
        "terrain_level_gap",
        "terrain_level_climb",
        "terrain_level_stairs_up",
        "terrain_level_stool",
    ):
        ax.plot(iterations, series(key), label=key.removeprefix("terrain_level_"))
    ax.set(title="Terrain curriculum", xlabel="Iteration", ylabel="Level")
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    if resources:
        minutes = np.array([r["elapsed_s"] for r in resources]) / 60
        ax.plot(minutes, [r["gpu_used_mib"] / 1024 for r in resources], label="Whole device used")
        ax.axhline(resources[0]["gpu_total_mib"] / 1024, color="red", linestyle="--", label="Device capacity")
        ax.plot(minutes, [r["host_available_mib"] / 1024 for r in resources], label="Host RAM available", alpha=0.7)
    ax.set(title="GPU memory and host RAM", xlabel="Elapsed minutes", ylabel="GiB")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    std = np.array([[v for k, v in r.items() if k.startswith("action_std/")] for r in rows])
    ax.plot(iterations, std.mean(-1), label="Mean std")
    ax.fill_between(iterations, std.min(-1), std.max(-1), alpha=0.2, label="Joint range")
    ax.axhline(3, color="red", linestyle="--", label="Health guard")
    ax.set(title="Policy exploration", xlabel="Iteration", ylabel="Action std")
    ax.legend(fontsize=8)
    ax = axes[1, 2]
    for key in ("world_loss", "image_loss", "prop_loss"):
        ax.plot(iterations, series(key), label=key)
    ax.set(title="World model training losses", xlabel="Iteration", ylabel="Loss", yscale="log")
    ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    run = json.loads((args.path / "run.json").read_text())
    learning = run["learning"]
    camera_envs = run["environment_contract"]["config"]["logical_camera_envs"]
    summary["learning"] = learning
    summary["camera_envs"] = camera_envs
    (args.path / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    title = (
        f"Go2 | {learning['num_envs']} envs / {camera_envs} cameras | "
        f"{learning['world_train_steps']} WM + {learning['depth_train_steps']} depth updates | "
        f"seed {run['args']['seed']}"
    )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(args.path / "training_curves.png", dpi=150)
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != "last"}, indent=2))


if __name__ == "__main__":
    main()
