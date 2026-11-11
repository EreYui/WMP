# SPDX-License-Identifier: Apache-2.0
"""按地形分组的速度课程，提前终止仍按完整 episode 归一化。"""

import torch

TERRAIN_GROUPS = {
    "omni": ("wave", "rough_slope"),
    "rough_flat": ("rough_flat", "flat"),
    "stairs_discrete": ("stairs_up", "stairs_down", "discrete"),
    "forward": (
        "gap",
        "continuous_gap",
        "bream",
        "stairs_cliff",
        "stepping_stones",
        "stepping_one_bridge",
        "climb",
        "tilt",
        "stool",
        "crawl",
        "plum_piles",
    ),
    "slope_cliff": ("slope_cliff",),
}
GROUP_LIMITS = ((1.5, 1.0), (2.0, 1.0), (1.0, 0.8), (1.0, 0.0), (1.5, 0.0))


class CommandCurriculum:
    def __init__(self, terrain_names: list[str], device="cpu"):
        lookup = {name: i for i, names in enumerate(TERRAIN_GROUPS.values()) for name in names}
        self.groups = torch.tensor([lookup[name.removeprefix("wmp_")] for name in terrain_names], device=device)
        self.limits = torch.tensor(GROUP_LIMITS, device=device)
        self.upper = torch.tensor(((0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.0), (0.5, 0.0)), device=device)
        self.reward_sum = torch.zeros(5, device=device)
        self.episode_count = torch.zeros(5, device=device)

    def accumulate(self, ids, rewards, lengths, max_length: int, reward_scale: float = 1.0):
        for group in range(5):
            chosen = ids[(self.groups[ids] == group) & (lengths[ids] > 0)]
            self.reward_sum[group] += rewards[chosen].sum()
            self.episode_count[group] += chosen.numel()
            required = (self.groups == group).sum()
            if required > 0 and self.episode_count[group] >= required:
                score = self.reward_sum[group] / (self.episode_count[group] * max_length * reward_scale)
                if score > 0.8:
                    self.upper[group] = torch.minimum(self.upper[group] + 0.2, self.limits[group])
                self.reward_sum[group] = 0
                self.episode_count[group] = 0

    def sample(self, ids, zero_probability: float = 0.05):
        groups = self.groups[ids]
        upper = self.upper[groups]
        velocity = torch.rand(len(ids), 2, device=ids.device) * upper
        omni = groups < 3
        velocity[omni] = (torch.rand(int(omni.sum()), 2, device=ids.device) * 2 - 1) * upper[omni]
        stopped = (torch.rand(len(ids), device=ids.device) < zero_probability) & omni
        velocity[stopped] = 0
        heading = (torch.rand(len(ids), device=ids.device) * 2 - 1) * torch.pi
        heading[~omni] = 0
        return velocity, heading

    def state_dict(self):
        return {key: getattr(self, key).clone() for key in ("upper", "reward_sum", "episode_count")}

    def load_state_dict(self, state):
        for key, value in state.items():
            getattr(self, key).copy_(value)
