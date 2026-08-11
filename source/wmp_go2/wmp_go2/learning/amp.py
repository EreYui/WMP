# SPDX-License-Identifier: Apache-2.0
"""Sequence-safe AMP expert sampling and replay."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wmp_go2.constants import AMP_OBSERVATION_DIM


class AMPMotionDataset:
    """Load one-file-per-sequence motions and sample interpolated 30-D AMP transitions."""

    def __init__(self, source: str | Path | list[str | Path], device: torch.device | str = "cpu", seed: int = 0):
        self.device = torch.device(device)
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

    @property
    def observation_dim(self) -> int:
        return AMP_OBSERVATION_DIM

    @property
    def num_motions(self) -> int:
        return len(self.sequences)

    def sample_transitions(self, batch_size: int, dt: float = 0.02) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(state, next_state, sequence_id)`` without crossing sequence boundaries."""
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

    def feed_forward_generator(self, num_mini_batches: int, mini_batch_size: int, dt: float = 0.02):
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
        return (value - self.mean) / torch.sqrt(self.var + 1.0e-8)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.mean.copy_(state["mean"])
        self.var.copy_(state["var"])
        self.count.copy_(state["count"])
