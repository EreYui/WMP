#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""逐地形录制策略。用法：python tools/record_all_terrains.py --checkpoint model.pt --robot go1 --output videos。"""

import argparse
import os
import subprocess
from pathlib import Path

TERRAINS = (
    "wave",
    "rough_slope",
    "stairs_up",
    "stairs_down",
    "discrete",
    "gap",
    "continuous_gap",
    "bream",
    "stairs_cliff",
    "slope_cliff",
    "stepping_stones",
    "stepping_one_bridge",
    "climb",
    "tilt",
    "stool",
    "crawl",
    "rough_flat",
    "plum_piles",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--robot", choices=("go1", "go2"), required=True)
    parser.add_argument("--terrains", nargs="+", choices=TERRAINS, default=TERRAINS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--level", type=int, default=0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for terrain in args.terrains:
        destination = args.output / f"{args.robot}_{terrain}_level{args.level}.mp4"
        if destination.exists():
            raise FileExistsError(destination)
        subprocess.run(
            [
                str(Path(os.environ.get("ISAACLAB_ROOT", "/home/re/IsaacLab")) / "isaaclab.sh"),
                "-p",
                str(root / "scripts/play.py"),
                "--checkpoint",
                str(args.checkpoint),
                "--robot",
                args.robot,
                "--terrain",
                terrain,
                "--headless",
                "--steps",
                str(args.steps),
                "--level",
                str(args.level),
                "--video",
                str(destination),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
