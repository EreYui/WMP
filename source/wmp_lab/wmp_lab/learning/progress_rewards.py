
# SPDX-License-Identifier: Apache-2.0
"""Episode-local forward-frontier gate for commanded obstacle locomotion.

A running frontier prevents a bounded forward/backward cycle from appearing as
new progress, regardless of its frequency. State is reset with the physical episode;
it is intentionally not restored into a newly spawned episode on checkpoint load.
"""
import math
from typing import NamedTuple

import torch

from .task_rewards import tracking_velocity

REFERENCE_REWARD_REVISION = "upstream_v3"
PROGRESS_REWARD_REVISION = "frontier_progress_v1"
REWARD_REVISIONS = (REFERENCE_REWARD_REVISION, PROGRESS_REWARD_REVISION)


def checkpoint_reward_revision(payload):
    extra = payload.get("extra", {})
    full = (extra.get("environment_contract") or {}).get("config", {})
    mini = (extra.get("environment") or {}).get("config", {})
    revision = full.get("reward_revision", mini.get("reward_revision", REFERENCE_REWARD_REVISION))
    if revision not in REWARD_REVISIONS:
        raise ValueError(f"unsupported saved reward revision: {revision}")
    if any(cfg.get("reward_revision", revision) != revision for cfg in (full, mini)):
        raise ValueError("checkpoint reward revision disagrees with environment contract")
    return revision


class ProgressStatus(NamedTuple):
    speed: torch.Tensor
    stalled: torch.Tensor


class ForwardProgressGate:
    def __init__(self, num_envs, dt, device="cpu", window_s=2.0, min_speed=0.1):
        if num_envs < 1 or not all(math.isfinite(v) and v > 0 for v in (dt, window_s, min_speed)):
            raise ValueError("positive finite progress dimensions and timing required")
        self.dt = dt
        self.steps = max(1, round(window_s / dt))
        self.window_s = self.steps * dt
        self.min_speed = min_speed
        self.frontier = torch.zeros(num_envs, device=device)
        self.direction = torch.ones_like(self.frontier)
        self.active = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.age = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.history = torch.zeros(self.steps, num_envs, device=device)
        self.cursor = 0
        self.speed = torch.zeros_like(self.frontier)
        self.stalled = torch.zeros_like(self.active)

    @torch.no_grad()
    def reset(self, ids, position, command, eligible):
        direction = torch.where(command < 0, -1.0, 1.0)
        frontier = direction * position
        self.direction[ids] = direction
        self.frontier[ids] = frontier
        self.history[:, ids] = frontier
        self.active[ids] = eligible & (command.abs() > self.min_speed)
        self.age[ids] = 0
        self.speed[ids] = 0
        self.stalled[ids] = False

    @torch.no_grad()
    def update(self, position, command, eligible):
        active = eligible & (command.abs() > self.min_speed)
        direction = torch.where(command < 0, -1.0, 1.0)
        changed = (active != self.active) | (active & (direction != self.direction))
        coordinate = direction * position
        # Avoid dynamic CUDA indexing or host synchronization on every control tick.
        self.history.copy_(torch.where(changed[None], coordinate[None], self.history))
        self.frontier.copy_(torch.where(changed, coordinate, self.frontier))
        self.age.masked_fill_(changed, 0)
        self.direction.copy_(direction)
        self.active.copy_(active)
        self.frontier.copy_(torch.maximum(self.frontier, coordinate))
        elapsed = (self.age + 1).clamp_max(self.steps) * self.dt
        self.speed.copy_((self.frontier - self.history[self.cursor]) / elapsed)
        self.history[self.cursor].copy_(self.frontier)
        self.cursor = (self.cursor + 1) % self.steps
        self.age.add_(1).clamp_max_(self.steps)
        self.stalled.copy_(active & (self.age >= self.steps) & (self.speed < self.min_speed))
        return ProgressStatus(self.speed, self.stalled)


def gate_positive_task_terms(terms, stalled):
    """Keep every negative term; suppress positive rewards for stalled movement."""
    return {name: torch.where(stalled & (value > 0), torch.zeros_like(value), value)
            for name, value in terms.items()}


def tracking_velocity_with_progress(velocity, command, obstacles, speed, direction, active):
    """Keep reference clipping and coefficients; use genuine directed ground advance."""
    adjusted = velocity.clone()
    adjusted[:, 0] = torch.where(active, direction * speed, adjusted[:, 0])
    return tracking_velocity(adjusted, command, obstacles)
