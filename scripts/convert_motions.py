#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Convert WMP JSON or ``legged_mpc_amp`` CSV clips into sequence-safe NPZ files."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "source" / "wmp_lab"))

from wmp_lab.learning.motions import TARGET_LEG_ORDER, WMP_SOURCE_LEG_ORDER, convert_motions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="WMP .txt JSON or legged_mpc_amp .csv inputs")
    parser.add_argument(
        "--dataset", choices=("go1", "go2", "a1", "upstream"), default="go1", help="Bundled upstream dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--source-fps", type=float, default=50.0, help="FPS used by CSV inputs")
    parser.add_argument("--source-leg-order", nargs=4, choices=TARGET_LEG_ORDER)
    parser.add_argument("--name-root", type=Path, help="Preserve relative directory names in output clip names")
    args = parser.parse_args()
    data_root = REPOSITORY_ROOT / "source" / "wmp_lab" / "wmp_lab" / "data"
    upstream = REPOSITORY_ROOT.parent / "wmp_go1" / "datasets"
    source_dir = (
        upstream
        / {"go1": "mocap_motions_collected", "go2": "go2_motion", "a1": "mocap_motions", "upstream": ""}[args.dataset]
    )
    if not args.inputs and args.dataset == "upstream":
        config = upstream.parent / "legged_gym/envs/a1/a1_amp_config.py"
        assignment = next(
            node
            for node in ast.parse(config.read_text()).body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "MOTION_FILES" for target in node.targets)
        )
        inputs = [upstream.parent / path for path in ast.literal_eval(assignment.value)]
    else:
        inputs = args.inputs or sorted(source_dir.rglob("*.txt"))
    if not inputs:
        parser.error("no motion inputs were found")
    output_dir = args.output_dir or data_root / (
        "motions" if args.inputs or args.dataset == "a1" else f"motions_{args.dataset}"
    )
    source_order = (
        tuple(args.source_leg_order)
        if args.source_leg_order
        else (TARGET_LEG_ORDER if not args.inputs else WMP_SOURCE_LEG_ORDER)
    )
    manifest = convert_motions(
        inputs,
        output_dir,
        source_fps=args.source_fps,
        source_leg_order=source_order,
        name_root=args.name_root or (source_dir if not args.inputs else None),
    )
    if not args.inputs:
        manifest["dataset"] = args.dataset
        manifest["upstream_repository"] = "https://github.com/haozhang04/wmp_go1"
        manifest["upstream_commit"] = "65cf1afd13e4792bce7b9def6f44372ecbb3cdf0"
        for entry in manifest["sequences"]:
            raw_path = Path(entry["source"])
            entry["source"] = (Path("..") / raw_path.relative_to(REPOSITORY_ROOT.parent)).as_posix()
        if args.dataset == "upstream":
            order = {
                str((Path("..") / path.relative_to(REPOSITORY_ROOT.parent)).as_posix()): i
                for i, path in enumerate(inputs)
            }
            manifest["sequences"].sort(key=lambda entry: order[entry["source"]])
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Converted {len(manifest['sequences'])} sequences to {output_dir}")
    for entry in manifest["sequences"]:
        print(f"  {entry['name']}: {entry['frames']} frames @ {entry['fps']:.1f} Hz")


if __name__ == "__main__":
    main()
