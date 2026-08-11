# SPDX-License-Identifier: Apache-2.0
"""Versioned WMP checkpoints without large replay buffers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    *,
    config: dict[str, Any],
    iteration: int,
    modules: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer],
    normalizer: Any | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "config": config,
        "iteration": int(iteration),
        "modules": {name: module.state_dict() for name, module in modules.items()},
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "normalizer": None if normalizer is None else normalizer.state_dict(),
        "replay_buffers_included": False,
    }
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    modules: dict[str, torch.nn.Module],
    optimizers: dict[str, torch.optim.Optimizer] | None = None,
    normalizer: Any | None = None,
    map_location: torch.device | str = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version {payload.get('version')}")
    missing = set(modules).difference(payload["modules"])
    if missing:
        raise KeyError(f"checkpoint is missing modules: {sorted(missing)}")
    for name, module in modules.items():
        module.load_state_dict(payload["modules"][name], strict=strict)
    if optimizers is not None:
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(payload["optimizers"][name])
    if normalizer is not None and payload.get("normalizer") is not None:
        normalizer.load_state_dict(payload["normalizer"])
    return payload
