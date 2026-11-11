# SPDX-License-Identifier: Apache-2.0
"""有界 CPU 时序环形缓存，深度采用 uint8 存储，采样保留 reset 标记。"""

import torch


class SequenceReplay:
    def __init__(self, num_envs: int, capacity: int):
        self.num_envs = num_envs
        self.capacity = max(2, capacity // num_envs)
        self.cursor = self.size = 0
        self.storage = {}

    def insert(self, frame):
        for key, value in frame.items():
            value = value.detach().cpu()
            if key in {"image", "real"}:
                value = ((value.clamp(-0.5, 0.5) + 0.5) * 255).round().to(torch.uint8)
            if key not in self.storage:
                self.storage[key] = torch.empty((self.capacity, *value.shape), dtype=value.dtype)
            self.storage[key][self.cursor].copy_(value)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, length: int, device="cpu", env_ids=None):
        if not self.size:
            raise RuntimeError("cannot sample empty sequence replay")
        length = min(length, self.size)
        start = torch.randint(self.size - length + 1, (batch_size,))
        env = (
            torch.randint(self.num_envs, (batch_size,))
            if env_ids is None
            else env_ids.cpu()[torch.randint(len(env_ids), (batch_size,))]
        )
        oldest = (self.cursor - self.size) % self.capacity
        time = (oldest + start[:, None] + torch.arange(length)) % self.capacity
        result = {key: value[time, env[:, None]].to(device) for key, value in self.storage.items()}
        for key in ("image", "real"):
            if key in result:
                result[key] = result[key].float() / 255 - 0.5
        result["is_first"][:, 0] = 1
        return result
