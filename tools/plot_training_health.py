#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""绘制逐关节动作健康、重建损失及课程；后一个日志覆盖重复迭代。"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--changes", nargs="*", type=int, default=[], help="Mark development-stage boundaries")
    args = parser.parse_args()
    combined = {}
    for path in args.paths:
        source = path / "metrics.jsonl" if path.is_dir() else path
        for line in source.read_text().splitlines():
            row = json.loads(line)
            combined[row["iteration"]] = row
    rows = [combined[key] for key in sorted(combined)]
    iterations = [row["iteration"] for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for prefix, axis, title in (
        ("action_std/", axes[0, 0], "Action std by joint"),
        ("action_mean_abs/", axes[0, 1], "Mean absolute action mean"),
        ("action_saturation/", axes[0, 2], "Raw action saturation fraction"),
    ):
        names = sorted({key for row in rows for key in row if key.startswith(prefix)})
        for name in names:
            axis.plot(iterations, [row.get(name, np.nan) for row in rows], label=name.split("/")[-1])
        axis.set_title(title)
        if prefix == "action_std/":
            axis.legend(fontsize=6, ncol=2)
    for name in ("prop_loss", "image_loss", "kl_loss"):
        axes[1, 0].plot(iterations, [row.get(name, np.nan) for row in rows], label=name)
    axes[1, 0].set(title="World model loss components", yscale="log")
    axes[1, 0].legend(fontsize=8)
    for name in ("task_return_mean", "rollout_return_mean"):
        axes[1, 1].plot(iterations, [row.get(name, np.nan) for row in rows], label=name)
    axes[1, 1].set_title("Return per rollout (24 steps)")
    axes[1, 1].legend(fontsize=8)
    for name in ("terrain_level_gap", "terrain_level_climb", "terrain_level_stairs_up"):
        axes[1, 2].plot(iterations, [row.get(name, np.nan) for row in rows], label=name.removeprefix("terrain_level_"))
    axes[1, 2].set_title("Terrain level (training curriculum)")
    axes[1, 2].legend(fontsize=8)
    for axis in axes.flat:
        for iteration in args.changes:
            axis.axvline(iteration, color="gray", linestyle="--", alpha=0.5)
        axis.set_xlabel("Iteration")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
