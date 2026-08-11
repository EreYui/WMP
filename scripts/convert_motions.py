#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert WMP JSON or ``legged_mpc_amp`` CSV clips into sequence-safe NPZ files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "source" / "wmp_go2"))

from wmp_go2.learning.motions import convert_motions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="WMP .txt JSON or legged_mpc_amp .csv inputs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "source" / "wmp_go2" / "wmp_go2" / "data" / "motions",
    )
    parser.add_argument("--source-fps", type=float, default=50.0, help="FPS used by CSV inputs")
    args = parser.parse_args()
    inputs = args.inputs or sorted((REPOSITORY_ROOT / "WMP" / "datasets" / "mocap_motions").glob("*.txt"))
    if not inputs:
        parser.error("no motion inputs were found")
    manifest = convert_motions(inputs, args.output_dir, source_fps=args.source_fps)
    print(f"Converted {len(manifest['sequences'])} sequences to {args.output_dir}")
    for entry in manifest["sequences"]:
        print(f"  {entry['name']}: {entry['frames']} frames @ {entry['fps']:.1f} Hz")


if __name__ == "__main__":
    main()
