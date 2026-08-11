# SPDX-License-Identifier: Apache-2.0
"""Helpers that make the legacy WMP observation slicing explicit and testable."""

import torch

from wmp_go2.constants import (
    HISTORY_DIM,
    HISTORY_FRAME_DIM,
    OBSERVATION_DIM,
    OBSERVATION_LAYOUT,
    PROPRIOCEPTION_DIM,
)


def split_observation(observation: torch.Tensor) -> dict[str, torch.Tensor]:
    if observation.shape[-1] != OBSERVATION_DIM:
        raise ValueError(f"expected {OBSERVATION_DIM} observations, got {observation.shape[-1]}")
    return {name: observation[..., span] for name, span in OBSERVATION_LAYOUT.items()}


def history_frame(observation: torch.Tensor) -> torch.Tensor:
    """Build the 42-D WMP history frame.

    This preserves WMP exactly: proprioception indices 0:6, skips the three
    velocity-command entries at 6:9, then appends prop[9:33] and all 12 actions.
    """
    parts = split_observation(observation)
    prop = parts["proprioception"]
    frame = torch.cat((prop[..., :6], prop[..., 9:], parts["previous_action"]), dim=-1)
    if frame.shape[-1] != HISTORY_FRAME_DIM:
        raise RuntimeError("history frame contract was violated")
    return frame


class HistoryBuffer:
    def __init__(self, num_envs: int, device: torch.device | str):
        self.data = torch.zeros(num_envs, HISTORY_DIM // HISTORY_FRAME_DIM, HISTORY_FRAME_DIM, device=device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.data.zero_()
        else:
            self.data[env_ids] = 0.0

    def append(self, observation: torch.Tensor) -> torch.Tensor:
        frame = history_frame(observation)
        self.data = torch.cat((self.data[:, 1:], frame.unsqueeze(1)), dim=1)
        return self.flatten()

    def flatten(self) -> torch.Tensor:
        return self.data.flatten(1)


def proprioception(observation: torch.Tensor) -> torch.Tensor:
    prop = split_observation(observation)["proprioception"]
    assert prop.shape[-1] == PROPRIOCEPTION_DIM
    return prop
