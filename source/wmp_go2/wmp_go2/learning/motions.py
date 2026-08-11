# SPDX-License-Identifier: Apache-2.0
"""Unified WMP JSON and ``legged_mpc_amp`` CSV motion conversion."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wmp_go2.constants import GO2_JOINT_ORDER

WMP_FRAME_DIM = 61
WMP_SOURCE_LEG_ORDER = ("FR", "FL", "RR", "RL")
MPC_SOURCE_LEG_ORDER = ("LF", "LH", "RF", "RH")
TARGET_LEG_ORDER = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class MotionSequence:
    name: str
    fps: float
    weight: float
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    root_lin_vel_b: np.ndarray
    root_ang_vel_b: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    def validate(self) -> None:
        if self.fps <= 0.0 or not np.isfinite(self.fps):
            raise ValueError(f"{self.name}: fps must be positive and finite")
        expected = {
            "root_pos": (self.num_frames, 3),
            "root_quat_wxyz": (self.num_frames, 4),
            "root_lin_vel_b": (self.num_frames, 3),
            "root_ang_vel_b": (self.num_frames, 3),
            "joint_pos": (self.num_frames, 12),
            "joint_vel": (self.num_frames, 12),
        }
        for field, shape in expected.items():
            value = getattr(self, field)
            if value.shape != shape:
                raise ValueError(f"{self.name}: {field} has shape {value.shape}, expected {shape}")
            if value.dtype != np.float32:
                raise ValueError(f"{self.name}: {field} must use float32")
            if not np.isfinite(value).all():
                raise ValueError(f"{self.name}: {field} contains non-finite values")
        norm = np.linalg.norm(self.root_quat_wxyz, axis=-1)
        if not np.allclose(norm, 1.0, atol=1.0e-4):
            raise ValueError(f"{self.name}: root quaternion is not normalized")

    def save(self, path: str | Path) -> None:
        self.validate()
        np.savez_compressed(
            Path(path),
            fps=np.float32(self.fps),
            weight=np.float32(self.weight),
            root_pos=self.root_pos,
            root_quat_wxyz=self.root_quat_wxyz,
            root_lin_vel_b=self.root_lin_vel_b,
            root_ang_vel_b=self.root_ang_vel_b,
            joint_pos=self.joint_pos,
            joint_vel=self.joint_vel,
        )


def _joint_remap(source_order: tuple[str, ...], aliases: dict[str, str] | None = None) -> np.ndarray:
    aliases = aliases or {}
    source_normalized = tuple(aliases.get(leg, leg) for leg in source_order)
    indices: list[int] = []
    for leg in TARGET_LEG_ORDER:
        start = source_normalized.index(leg) * 3
        indices.extend((start, start + 1, start + 2))
    return np.asarray(indices, dtype=np.int64)


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError("zero-norm quaternion in motion data")
    quat = quat / norm
    quat[quat[:, 0] < 0.0] *= -1.0
    return quat.astype(np.float32)


def quaternion_wxyz_from_rpy(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    quat = np.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        axis=-1,
    )
    return _normalize_quaternion(quat.astype(np.float32))


def load_wmp_json(path: str | Path) -> MotionSequence:
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    frames = np.asarray(payload["Frames"], dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != WMP_FRAME_DIM:
        raise ValueError(f"{path}: expected [frames, {WMP_FRAME_DIM}], got {frames.shape}")
    dt = float(payload["FrameDuration"])
    remap = _joint_remap(WMP_SOURCE_LEG_ORDER)
    quat_xyzw = frames[:, 3:7]
    quat_wxyz = _normalize_quaternion(quat_xyzw[:, (3, 0, 1, 2)])
    sequence = MotionSequence(
        name=path.stem,
        fps=1.0 / dt,
        weight=float(payload.get("MotionWeight", 1.0)),
        root_pos=frames[:, 0:3].copy(),
        root_quat_wxyz=quat_wxyz,
        root_lin_vel_b=frames[:, 31:34].copy(),
        root_ang_vel_b=frames[:, 34:37].copy(),
        joint_pos=frames[:, 7:19][:, remap].copy(),
        joint_vel=frames[:, 37:49][:, remap].copy(),
    )
    sequence.validate()
    return sequence


def _csv_column(rows: list[dict[str, str]], names: tuple[str, ...], default: float | None = None) -> np.ndarray:
    for name in names:
        if name in rows[0] and rows[0][name] not in (None, ""):
            return np.asarray([float(row[name]) for row in rows], dtype=np.float32)
    if default is not None:
        return np.full(len(rows), default, dtype=np.float32)
    raise ValueError(f"CSV is missing one of required columns: {names}")


def load_mpc_csv(path: str | Path, source_fps: float = 50.0) -> MotionSequence:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path}: CSV contains no samples")
    root_pos = np.stack([_csv_column(rows, (f"base_p{axis}", f"root_p{axis}"), 0.0) for axis in "xyz"], axis=-1)
    roll = _csv_column(rows, ("base_roll", "root_roll"), 0.0)
    pitch = _csv_column(rows, ("base_pitch", "root_pitch"), 0.0)
    yaw = _csv_column(rows, ("base_yaw", "root_yaw"), 0.0)
    lin = np.stack([_csv_column(rows, (f"root_lin_vel_b{axis}", f"base_v{axis}"), 0.0) for axis in "xyz"], -1)
    ang = np.stack([_csv_column(rows, (f"root_ang_vel_b{axis}", f"base_w{axis}"), 0.0) for axis in "xyz"], -1)
    joint_pos = np.stack([_csv_column(rows, (f"q{i}",)) for i in range(12)], -1)
    joint_vel = np.stack([_csv_column(rows, (f"dq{i}",)) for i in range(12)], -1)
    remap = _joint_remap(MPC_SOURCE_LEG_ORDER, {"LF": "FL", "RF": "FR", "LH": "RL", "RH": "RR"})
    sequence = MotionSequence(
        name=path.stem,
        fps=float(source_fps),
        weight=1.0,
        root_pos=root_pos.astype(np.float32),
        root_quat_wxyz=quaternion_wxyz_from_rpy(roll, pitch, yaw),
        root_lin_vel_b=lin.astype(np.float32),
        root_ang_vel_b=ang.astype(np.float32),
        joint_pos=joint_pos[:, remap].astype(np.float32),
        joint_vel=joint_vel[:, remap].astype(np.float32),
    )
    sequence.validate()
    return sequence


def convert_motions(inputs: list[str | Path], output_dir: str | Path, *, source_fps: float = 50.0) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for raw_path in sorted(map(Path, inputs)):
        sequence = load_mpc_csv(raw_path, source_fps) if raw_path.suffix.lower() == ".csv" else load_wmp_json(raw_path)
        output_path = output_dir / f"{sequence.name}.npz"
        sequence.save(output_path)
        entries.append(
            {
                "name": sequence.name,
                "file": output_path.name,
                "source": str(raw_path.resolve()),
                "frames": sequence.num_frames,
                "fps": sequence.fps,
                "weight": sequence.weight,
            }
        )
    manifest: dict[str, object] = {
        "format_version": 1,
        "joint_order": list(GO2_JOINT_ORDER),
        "quaternion_order": "wxyz",
        "sequence_boundaries_preserved": True,
        "kinematic_retargeting": False,
        "sequences": entries,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return manifest
