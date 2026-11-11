# SPDX-License-Identifier: Apache-2.0
"""旁路训练统计：不使用随机episode计数器，丢弃每环境首段不完整episode。"""

from collections import deque

import torch


class EpisodeMonitor:
    def __init__(self, num_envs, step_dt, device="cpu", window=4096):
        self.step_dt = step_dt
        self.age = torch.zeros(num_envs, device=device)
        self.task_return = torch.zeros_like(self.age)
        self.raw_return = torch.zeros_like(self.age)
        self.has_reset = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.episodes = deque(maxlen=window)
        self.completed = self.discarded = 0

    @torch.no_grad()
    def update(self, task_reward, terminated, truncated, raw_reward=None):
        self.age += 1
        self.task_return += task_reward.detach()
        self.raw_return += (task_reward if raw_reward is None else raw_reward).detach()
        done = terminated | truncated
        eligible = done & self.has_reset
        if eligible.any():
            values = (
                torch.stack(
                    (self.age * self.step_dt, self.task_return, self.raw_return, (truncated & ~terminated).float()), -1
                )[eligible]
                .cpu()
                .tolist()
            )
            self.episodes.extend(values)
            self.completed += len(values)
        self.discarded += int((done & ~self.has_reset).sum())
        self.has_reset |= done
        self.age[done] = 0
        self.task_return[done] = 0
        self.raw_return[done] = 0

    def metrics(self):
        result = dict(
            episodes_completed=self.completed,
            episodes_discarded_initial=self.discarded,
            episode_window_count=len(self.episodes),
        )
        if self.episodes:
            values = torch.tensor(list(self.episodes))
            result.update(
                episode_length_mean_s=float(values[:, 0].mean()),
                episode_length_median_s=float(values[:, 0].median()),
                episode_length_p90_s=float(torch.quantile(values[:, 0], 0.9)),
                episode_task_return_mean=float(values[:, 1].mean()),
                episode_unclipped_task_return_mean=float(values[:, 2].mean()),
                episode_timeout_fraction=float(values[:, 3].mean()),
            )
        return result
