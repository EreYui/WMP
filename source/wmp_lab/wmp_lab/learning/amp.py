# SPDX-License-Identifier: Apache-2.0
# Portions Copyright 2020 The Google Research Authors and Bytedance Ltd. and/or its affiliates.
"""Sequence-safe AMP expert sampling and replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from wmp_lab.constants import AMP_OBSERVATION_DIM


class AMPMotionDataset:
    """Load one-file-per-sequence motions and sample interpolated 30-D AMP transitions."""

    def __init__(
        self,
        source: str | Path | list[str | Path],
        device: torch.device | str = "cpu",
        seed: int = 0,
        *,
        sampling_mode="legacy_uniform",
        preload=0,
        dt=0.02,
    ):
        self.device = torch.device(device)
        if sampling_mode not in ("upstream_v1", "legacy_uniform"):
            raise ValueError(f"unsupported AMP sampling mode: {sampling_mode}")
        if preload < 0 or dt <= 0:
            raise ValueError("AMP preload must be nonnegative and dt positive")
        self.sampling_mode, self.dt, self.preload = sampling_mode, dt, preload
        self._random = np.random.RandomState(seed)
        if isinstance(source, str | Path):
            source_path = Path(source)
            if source_path.is_dir():
                manifest_path = source_path / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                paths = [source_path / entry["file"] for entry in manifest["sequences"]]
            else:
                paths = [source_path]
        else:
            paths = [Path(path) for path in source]
        if not paths:
            raise ValueError("at least one motion sequence is required")
        self.sequences: list[torch.Tensor] = []
        fps: list[float] = []
        weights: list[float] = []
        self.names: list[str] = []
        for path in paths:
            with np.load(path) as data:
                amp = np.concatenate(
                    (data["joint_pos"], data["root_lin_vel_b"], data["root_ang_vel_b"], data["joint_vel"]), axis=-1
                ).astype(np.float32)
                fps.append(float(data["fps"]))
                weights.append(float(data["weight"]))
            if amp.ndim != 2 or amp.shape[1] != AMP_OBSERVATION_DIM or amp.shape[0] < 2:
                raise ValueError(f"{path}: invalid AMP array shape {amp.shape}")
            self.sequences.append(torch.as_tensor(amp, device=self.device))
            self.names.append(path.stem)
        self.fps = torch.tensor(fps, dtype=torch.float32, device=self.device)
        self.weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        self.weights /= self.weights.sum()
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self._weights_np = np.asarray(weights, dtype=np.float64)
        self._weights_np /= self._weights_np.sum()
        self._fps_np = np.asarray(fps, dtype=np.float64)
        self._lengths_np = np.asarray([len(sequence) for sequence in self.sequences], dtype=np.float64)
        self._durations_np = (self._lengths_np - 1) * (1 / self._fps_np)
        contract = dict(
            sampling_mode=sampling_mode,
            transition_dt=dt,
            preload=preload,
            seed=seed,
            files=[
                dict(
                    name=path.name,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    frames=len(sequence),
                    fps=rate,
                    weight=weight,
                )
                for path, sequence, rate, weight in zip(paths, self.sequences, fps, weights, strict=True)
            ],
        )
        self.contract = dict(
            contract,
            fingerprint=hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
            source=str(source_path.resolve()) if isinstance(source, str | Path) else [str(p.resolve()) for p in paths],
        )
        self.preloaded = None
        if preload:
            if sampling_mode != "upstream_v1":
                raise ValueError("preloaded AMP pool requires upstream_v1 sampling")
            ids, times = self._upstream_indices_times(preload, dt)
            self.preloaded = (
                torch.empty(preload, AMP_OBSERVATION_DIM, device=self.device),
                torch.empty(preload, AMP_OBSERVATION_DIM, device=self.device),
                torch.as_tensor(ids, device=self.device),
            )
            for start in range(0, preload, 65536):
                end = min(preload, start + 65536)
                self.preloaded[0][start:end] = self.frames_at_times(ids[start:end], times[start:end])
                self.preloaded[1][start:end] = self.frames_at_times(ids[start:end], times[start:end] + dt)

    def sampling_state(self):
        return dict(numpy=self._random.get_state(), torch=self.generator.get_state())

    def load_sampling_state(self, state):
        self._random.set_state(state["numpy"])
        self.generator.set_state(state["torch"].cpu())

    def _upstream_indices_times(self, batch_size, dt):
        ids = self._random.choice(len(self.sequences), size=batch_size, p=self._weights_np, replace=True)
        times = self._durations_np[ids] * self._random.uniform(size=batch_size) - (dt + 1 / self._fps_np[ids])
        return ids, np.maximum(np.zeros_like(times), times)

    def frames_at_times(self, ids, times):
        """上游时间映射与线性 AMP 插值；独立于根四元数的插值。"""
        ids, times = np.asarray(ids, dtype=np.int64), np.asarray(times, dtype=np.float64)
        position = times / self._durations_np[ids] * self._lengths_np[ids]
        low, high = np.floor(position).astype(np.int64), np.ceil(position).astype(np.int64)
        if np.any(low < 0) or np.any(high >= self._lengths_np[ids]):
            raise ValueError("motion is too short or time outside upstream interpolation range")
        result = torch.empty(len(ids), AMP_OBSERVATION_DIM, device=self.device)
        blend = torch.as_tensor(position - low, device=self.device, dtype=torch.float32)[:, None]
        for index in np.unique(ids):
            mask = ids == index
            weight = blend[mask]
            sequence = self.sequences[index]
            result[mask] = (1 - weight) * sequence[low[mask]] + weight * sequence[high[mask]]
        return result

    @property
    def observation_dim(self) -> int:
        return AMP_OBSERVATION_DIM

    @property
    def num_motions(self) -> int:
        return len(self.sequences)

    def sample_transitions(
        self, batch_size: int, dt: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(state, next_state, sequence_id)`` without crossing sequence boundaries."""
        dt = self.dt if dt is None else dt
        if dt != self.dt:
            raise ValueError("AMP transition dt differs from the fixed data contract")
        if self.sampling_mode == "upstream_v1":
            if self.preloaded is not None:
                indices = self._random.choice(self.preload, size=batch_size)
                return tuple(value[indices] for value in self.preloaded)
            ids, times = self._upstream_indices_times(batch_size, dt)
            return (
                self.frames_at_times(ids, times),
                self.frames_at_times(ids, times + dt),
                torch.as_tensor(ids, device=self.device),
            )
        seq_ids = torch.multinomial(self.weights, batch_size, replacement=True, generator=self.generator)
        states = torch.empty(batch_size, AMP_OBSERVATION_DIM, device=self.device)
        next_states = torch.empty_like(states)
        for seq_id in torch.unique(seq_ids).tolist():
            mask = seq_ids == seq_id
            count = int(mask.sum())
            sequence = self.sequences[seq_id]
            advance = dt * float(self.fps[seq_id])
            max_start = max(float(sequence.shape[0] - 1) - advance, 0.0)
            position = torch.rand(count, generator=self.generator, device=self.device) * max_start
            next_position = position + advance
            states[mask] = self._interpolate(sequence, position)
            next_states[mask] = self._interpolate(sequence, next_position)
        return states, next_states, seq_ids

    @staticmethod
    def _interpolate(sequence: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        low = position.floor().long().clamp_max(sequence.shape[0] - 1)
        high = (low + 1).clamp_max(sequence.shape[0] - 1)
        blend = (position - low).unsqueeze(-1)
        return torch.lerp(sequence[low], sequence[high], blend)

    def feed_forward_generator(self, num_mini_batches: int, mini_batch_size: int, dt: float | None = None):
        for _ in range(num_mini_batches):
            state, next_state, _ = self.sample_transitions(mini_batch_size, dt)
            yield state, next_state


class ReplayBuffer:
    def __init__(self, feature_dim: int, capacity: int, device: torch.device | str = "cpu"):
        self.states = torch.zeros(capacity, feature_dim, device=device)
        self.next_states = torch.zeros_like(self.states)
        self.capacity = capacity
        self.cursor = 0
        self.size = 0

    def insert(self, states: torch.Tensor, next_states: torch.Tensor) -> None:
        if states.shape != next_states.shape or states.shape[1:] != self.states.shape[1:]:
            raise ValueError("replay transition shape mismatch")
        if states.shape[0] > self.capacity:
            states, next_states = states[-self.capacity :], next_states[-self.capacity :]
        count = states.shape[0]
        indices = (torch.arange(count, device=states.device) + self.cursor) % self.capacity
        self.states[indices] = states.detach()
        self.next_states[indices] = next_states.detach()
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("cannot sample an empty replay buffer")
        indices = torch.randint(self.size, (batch_size,), device=self.states.device)
        return self.states[indices], self.next_states[indices]


class RunningMeanStd:
    def __init__(self, shape: int, epsilon: float = 1.0e-4, device: torch.device | str = "cpu"):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = torch.tensor(epsilon, device=device)

    def update(self, batch: torch.Tensor) -> None:
        batch_mean = batch.mean(0)
        batch_var = batch.var(0, unbiased=False)
        batch_count = torch.tensor(batch.shape[0], device=batch.device)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta.square() * self.count * batch_count / total
        self.mean, self.var, self.count = new_mean, m2 / total, total

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return ((value - self.mean) / torch.sqrt(self.var + 1.0e-4)).clamp(-10.0, 10.0)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.mean.copy_(state["mean"])
        self.var.copy_(state["var"])
        self.count.copy_(state["count"])
