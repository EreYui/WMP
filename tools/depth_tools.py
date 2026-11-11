#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""比较以米为单位的深度 NPZ。用法：python tools/depth_tools.py warp.npz isaac.npz --output compare。"""

import argparse
import json
from pathlib import Path

import numpy as np


def depth_metrics(left, right):
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("depth arrays must have matching shapes and finite meter values")
    delta = np.abs(left - right)
    return dict(
        mae_m=float(delta.mean()), rmse_m=float(np.sqrt(((left - right) ** 2).mean())), max_m=float(delta.max())
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.inputs) < 2:
        parser.error("provide at least two captures")
    args.output.mkdir(parents=True, exist_ok=False)
    values = []
    calibration = []
    for path in args.inputs:
        with np.load(path) as data:
            if "depth_m" not in data:
                raise ValueError(f"{path}: depth_m is required; normalized/PNG units are ambiguous")
            values.append(data["depth_m"].copy())
            keys = ("intrinsics", "camera_position", "camera_quaternion")
            calibration.append({key: data[key].copy() for key in keys if key in data})
    rows = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            for key in set(calibration[i]) & set(calibration[j]):
                left, right = calibration[i][key], calibration[j][key]
                if key == "camera_quaternion" and np.allclose(left, -right, atol=1e-5):
                    continue
                if not np.allclose(left, right, atol=1e-5, rtol=1e-5):
                    raise ValueError(f"capture calibration mismatch: {key}")
            row = dict(left=str(args.inputs[i]), right=str(args.inputs[j]), **depth_metrics(values[i], values[j]))
            rows.append(row)
            print(json.dumps(row))
    (args.output / "metrics.json").write_text(json.dumps(rows, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(values), figsize=(4 * len(values), 4), squeeze=False)
    for axis, path, value in zip(axes[0], args.inputs, values, strict=True):
        axis.imshow(value.reshape(-1, 64, 64)[0], cmap="gray", vmin=0, vmax=2)
        axis.set_title(path.stem)
    fig.tight_layout()
    fig.savefig(args.output / "depth.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
