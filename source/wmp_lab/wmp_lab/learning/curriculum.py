# SPDX-License-Identifier: Apache-2.0
"""按地形分组的速度课程，提前终止仍按完整 episode 归一化。"""

import math

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


def reference_group_sizes(proportions, reference_envs):
    """按参考 Go1 的 ceil 分组边界计算评分所需的完成回合数。"""
    lookup = {name: i for i, names in enumerate(TERRAIN_GROUPS.values()) for name in names}
    total = sum(weight for _, weight in proportions)
    if total <= 0 or reference_envs < 1:
        raise ValueError("positive terrain weights and reference environment count are required")
    counts, cumulative, previous = [0] * len(TERRAIN_GROUPS), 0.0, 0
    for name, weight in proportions:
        cumulative += weight / total
        end = math.ceil(reference_envs * cumulative - 1e-9)
        counts[lookup[name.removeprefix("wmp_")]] += end - previous
        previous = end
    return counts


class CommandCurriculum:
    def __init__(self, terrain_names: list[str], device="cpu"):
        lookup = {name: i for i, names in enumerate(TERRAIN_GROUPS.values()) for name in names}
        self.groups = torch.tensor([lookup[name.removeprefix("wmp_")] for name in terrain_names], device=device)
        self._reference_configured = False
        self.required_episodes = torch.bincount(self.groups, minlength=5)
        self.limits = torch.tensor(GROUP_LIMITS, device=device)
        self.upper = torch.tensor(((0.5, 0.5), (0.5, 0.5), (0.5, 0.5), (0.5, 0.0), (0.5, 0.0)), device=device)
        self.reward_sum = torch.zeros(5, device=device)
        self.episode_count = torch.zeros(5, device=device)

    def configure_reference(self, proportions, reference_envs):
        self._reference_configured = True
        required = reference_group_sizes(proportions, reference_envs)
        self.required_episodes.copy_(torch.tensor(required, device=self.groups.device))

    def accumulate(self, ids, rewards, lengths, max_length: int, reward_scale: float = 1.0):
        for group in range(5):
            chosen = ids[(self.groups[ids] == group) & (lengths[ids] > 0)]
            self.reward_sum[group] += rewards[chosen].sum()
            self.episode_count[group] += chosen.numel()
            required = self.required_episodes[group]
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
        keys = ("upper", "reward_sum", "episode_count", "required_episodes")
        return {key: getattr(self, key).clone() for key in keys}

    def load_state_dict(self, state):
        for key, value in state.items():
            if (
                key == "required_episodes" and self._reference_configured
                and not torch.equal(value.to(self.groups.device), self.required_episodes)
            ):
                raise ValueError("command curriculum reference groups differ from checkpoint")
            getattr(self, key).copy_(value)


class TerrainCurriculumSlots:
    """每个物理环境轮换独立地形课程，匹配参考逻辑环境的回合证据频率。"""

    def __init__(self, levels, slots_per_env):
        if slots_per_env < 1:
            raise ValueError("terrain curriculum slots must be positive")
        self.levels = levels[:, None].repeat(1, slots_per_env)
        self.active = torch.zeros_like(levels)

    @property
    def active_levels(self):
        ids = torch.arange(len(self.active), device=self.active.device)
        return self.levels[ids, self.active]

    def advance(self, env_ids, completed_levels):
        self.levels[env_ids, self.active[env_ids]] = completed_levels
        self.active[env_ids] = (self.active[env_ids] + 1) % self.levels.shape[1]
        return self.levels[env_ids, self.active[env_ids]]

    def state_dict(self):
        return dict(levels=self.levels.clone(), active=self.active.clone())

    def load_state_dict(self, state):
        if state["levels"].shape != self.levels.shape or state["active"].shape != self.active.shape:
            raise ValueError("terrain curriculum slot layout differs from checkpoint")
        if (state["active"] < 0).any() or (state["active"] >= self.levels.shape[1]).any():
            raise ValueError("invalid active terrain curriculum slot")
        self.levels.copy_(state["levels"])
        self.active.copy_(state["active"])
