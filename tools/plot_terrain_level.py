#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""绘制 JSONL 地形等级。用法：python tools/plot_terrain_level.py logs/go1/RUN/metrics.jsonl --output levels.png。"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("terrain_levels.png"))
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(10, 6))
    for path in args.paths:
        source = path / "metrics.jsonl" if path.is_dir() else path
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        names = sorted({key for row in rows for key in row if key.startswith("terrain_level_")})
        for name in names:
            points = [(row["iteration"], row[name]) for row in rows if name in row]
            axis.plot(*zip(*points, strict=True), label=f"{source.parent.name}/{name.removeprefix('terrain_level_')}")
    axis.set(xlabel="Iteration", ylabel="Terrain level")
    axis.legend(fontsize=8)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
