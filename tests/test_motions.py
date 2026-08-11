# SPDX-License-Identifier: Apache-2.0

import csv
import json
from pathlib import Path

import numpy as np
import torch
from wmp_go2.constants import GO2_JOINT_ORDER
from wmp_go2.learning.amp import AMPMotionDataset
from wmp_go2.learning.motions import convert_motions, load_mpc_csv, load_wmp_json

ROOT = Path(__file__).resolve().parents[1]
WMP_MOTIONS = ROOT / "WMP" / "datasets" / "mocap_motions"


def test_all_wmp_motions_convert_with_explicit_mapping(tmp_path: Path):
    inputs = sorted(WMP_MOTIONS.glob("*.txt"))
    manifest = convert_motions(inputs, tmp_path)
    assert len(manifest["sequences"]) == 4
    assert manifest["joint_order"] == list(GO2_JOINT_ORDER)
    assert manifest["sequence_boundaries_preserved"] is True
    assert manifest["kinematic_retargeting"] is False
    for entry in manifest["sequences"]:
        assert entry["frames"] == 501
        assert entry["fps"] == 50.0
        with np.load(tmp_path / entry["file"]) as data:
            assert data["joint_pos"].shape == (501, 12)
            assert data["joint_vel"].shape == (501, 12)
            assert np.allclose(np.linalg.norm(data["root_quat_wxyz"], axis=-1), 1.0, atol=1.0e-5)

    source = json.loads(inputs[0].read_text())["Frames"][0]
    sequence = load_wmp_json(inputs[0])
    source_joint = np.asarray(source[7:19]).reshape(4, 3)  # FR, FL, RR, RL
    np.testing.assert_allclose(sequence.joint_pos[0].reshape(4, 3), source_joint[[1, 0, 3, 2]])
    np.testing.assert_allclose(sequence.root_quat_wxyz[0], np.asarray(source[3:7])[[3, 0, 1, 2]])


def test_amp_interpolation_never_crosses_sequence(tmp_path: Path):
    for name, marker in (("first", 0.0), ("second", 100.0)):
        array = np.full((5, 12), marker, np.float32)
        np.savez(
            tmp_path / f"{name}.npz",
            fps=np.float32(50.0),
            weight=np.float32(1.0),
            root_pos=np.zeros((5, 3), np.float32),
            root_quat_wxyz=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], np.float32), (5, 1)),
            root_lin_vel_b=np.full((5, 3), marker, np.float32),
            root_ang_vel_b=np.full((5, 3), marker, np.float32),
            joint_pos=array,
            joint_vel=array,
        )
    dataset = AMPMotionDataset(sorted(tmp_path.glob("*.npz")), seed=3)
    state, next_state, sequence_id = dataset.sample_transitions(256, dt=0.02)
    assert state.shape == next_state.shape == (256, 30)
    expected = sequence_id.float() * 100.0
    torch.testing.assert_close(state[:, 0], expected)
    torch.testing.assert_close(next_state[:, 0], expected)


def test_legged_mpc_csv_mapping(tmp_path: Path):
    csv_path = tmp_path / "trot.csv"
    fields = ["time", "base_px", "base_py", "base_pz", "base_yaw"]
    fields += [f"q{i}" for i in range(12)] + [f"dq{i}" for i in range(12)]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for frame in range(2):
            row = {name: 0.0 for name in fields}
            row["time"] = frame * 0.02
            row.update({f"q{i}": float(i) for i in range(12)})
            row.update({f"dq{i}": float(20 + i) for i in range(12)})
            writer.writerow(row)
    sequence = load_mpc_csv(csv_path)
    # [LF,LH,RF,RH] -> [FL,FR,RL,RR]
    np.testing.assert_array_equal(sequence.joint_pos[0], [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11])
