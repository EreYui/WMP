#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""审计 IsaacLab 内 AMP 对照数据；不启动训练、不裁剪原始参考动作。"""

import argparse
import hashlib
import json
import shlex
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import wmp_lab
from wmp_lab.learning.amp import AMPMotionDataset
from wmp_lab.learning.config import PAPER_CFG
from wmp_lab.robots import CONTROL_REVISION, robot_profile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "experiments/amp_isaaclab.json")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/validation/amp_alignment_20261015/audit.json")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan["control_revision"] != CONTROL_REVISION:
        raise ValueError("experiment plan control revision differs from installed code")
    results, shared = {}, {}
    for case in plan["cases"]:
        robot = case["robot"]
        path = Path(wmp_lab.__file__).parent / "data" / case["dataset"]
        expert = AMPMotionDataset(path, seed=7, sampling_mode="upstream_v1")
        assert expert.num_motions == case["clips"]
        profile = robot_profile(robot)
        urdf = (
            ROOT.parent
            / "wmp_go1/resources/robots"
            / robot
            / "urdf"
            / ("go1_new_2.urdf" if robot == "go1" else "go2.urdf")
        )
        joints = {node.attrib["name"]: node.find("limit") for node in ET.parse(urdf).getroot().findall("joint")}
        low = np.asarray([float(joints[name].attrib["lower"]) for name in profile.joint_order])
        high = np.asarray([float(joints[name].attrib["upper"]) for name in profile.joint_order])
        midpoint, half = (low + high) / 2, (high - low) * 0.45
        clips = []
        for name, sequence, weight in zip(expert.names, expert.sequences, expert._weights_np, strict=True):
            values = sequence.numpy()
            q, dq = values[:, :12], values[:, 18:]
            clips.append(
                dict(
                    name=name,
                    frames=len(q),
                    probability=float(weight),
                    hard_limit_joint_fraction=float(((q < low) | (q > high)).mean()),
                    soft_limit_joint_fraction=float(((q < midpoint - half) | (q > midpoint + half)).mean()),
                    velocity_limit_joint_fraction=float((np.abs(dq) > profile.velocity_limits).mean()),
                    mean_body_vx=float(values[:, 12].mean()),
                )
            )
        first, second, ids = expert.sample_transitions(32768)
        assert torch.isfinite(first).all() and torch.isfinite(second).all()
        assert ids.unique().numel() == expert.num_motions
        sample_hash = hashlib.sha256(
            first.numpy().tobytes() + second.numpy().tobytes() + ids.numpy().tobytes()
        ).hexdigest()
        command = [
            "./bash/leggedskill.sh",
            "-t",
            "--robot",
            robot,
            "--amp-set",
            case["amp_set"],
            *plan["common_args"],
            "--log-dir",
            f"logs/amp_comparison/{case['name']}_seed7",
        ]
        results[case["name"]] = dict(
            robot=robot,
            audit_contract=expert.contract,
            clips=clips,
            total_frames=sum(clip["frames"] for clip in clips),
            audit_pairs=32768,
            sample_sha256=sample_hash,
            command=shlex.join(command),
        )
        if case["amp_set"] == "upstream":
            shared[robot] = sample_hash
    assert shared["go1"] == shared["go2"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                plan=str(args.plan),
                plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                learning=PAPER_CFG.to_dict(),
                package_sha256={
                    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (ROOT / "source/wmp_lab/wmp_lab").rglob("*.py")
                },
                shared_pairs_identical=True,
                audit_preload=0,
                production_preload=2_000_000,
                training_started=False,
                cases=results,
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"[PASS] shared Go1/Go2 AMP pairs identical; saved {args.output}")


if __name__ == "__main__":
    main()
