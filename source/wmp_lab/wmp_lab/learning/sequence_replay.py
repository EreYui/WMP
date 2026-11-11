# SPDX-License-Identifier: Apache-2.0
"""有界 CPU 时序环形缓存，保留浮点深度与 reset 标记。"""

import torch


class EpisodeReplay:
    """当前回合与完成回合槽分离；小环境保留多回合以匹配参考缓存规模。"""

    def __init__(self, num_envs, episode_capacity, camera_mask, episodes_per_env=1):
        if episodes_per_env < 1:
            raise ValueError("episodes_per_env must be positive")
        self.num_envs, self.capacity, self.episodes_per_env = num_envs, episode_capacity, episodes_per_env
        self.camera_ids = camera_mask.cpu().nonzero(as_tuple=False).flatten()
        self.camera_inverse = torch.full((num_envs,), -1, dtype=torch.long)
        self.camera_inverse[self.camera_ids] = torch.arange(len(self.camera_ids))
        self.current, self.storage = {}, {}
        self.indices = torch.zeros(num_envs, dtype=torch.long)
        self.lengths = torch.zeros(num_envs * episodes_per_env, dtype=torch.long)
        self.next_slot = torch.zeros(num_envs, dtype=torch.long)

    @property
    def size(self):
        return int(self.lengths.max())

    @property
    def total_samples(self):
        return int(self.lengths.sum())

    def has_completed(self, env_ids):
        lengths = self.lengths.reshape(self.num_envs, self.episodes_per_env)
        return (lengths[env_ids.detach().cpu()].sum(-1) > 0).to(env_ids.device)

    def finish_episodes(self, env_ids):
        env_ids = env_ids.detach().cpu().long()
        slot = self.next_slot[env_ids]
        destinations = env_ids * self.episodes_per_env + slot
        for key, value in self.current.items():
            if key == "real":
                rows = self.camera_inverse[env_ids]
                valid = rows >= 0
                targets = rows[valid] * self.episodes_per_env + slot[valid]
                self.storage[key][targets] = value[rows[valid]]
            else:
                self.storage[key][destinations] = value[env_ids]
        self.lengths[destinations] = self.indices[env_ids]
        self.indices[env_ids] = 0
        self.next_slot[env_ids] = (slot + 1) % self.episodes_per_env

    def insert(self, frame):
        # reset 后第一帧不入库，与上游 wm_is_first 过滤一致。
        valid = ~frame["is_first"].detach().cpu().bool()
        if (self.indices[valid] >= self.capacity).any():
            raise RuntimeError("episode replay capacity exceeded; check episode timeout")
        for key, value in frame.items():
            if key == "is_first":
                continue
            value = value.detach().cpu()
            ids = torch.arange(self.num_envs)
            if key == "real":
                ids = self.camera_ids
                value = value[ids]
            if key not in self.current:
                shape = (len(ids), self.capacity, *value.shape[1:])
                self.current[key] = torch.empty(shape, dtype=value.dtype)
                stored_shape = (len(ids) * self.episodes_per_env, *shape[1:])
                self.storage[key] = torch.empty(stored_shape, dtype=value.dtype)
            selected = valid[ids]
            rows = torch.arange(len(ids))[selected]
            self.current[key][rows, self.indices[ids[selected]]] = value[selected]
        self.indices[valid] += 1

    def sample(self, batch_size, length, device="cpu", env_ids=None):
        weights = self.lengths.float()
        if env_ids is not None:
            # 深度监督按有效相机回合槽均匀采样，不按 episode 长度加权。
            mask = torch.zeros(self.num_envs, dtype=torch.bool)
            mask[env_ids.cpu()] = True
            weights = ((weights > 0) & mask.repeat_interleave(self.episodes_per_env)).float()
        if not weights.any():
            raise RuntimeError("cannot sample without a completed episode")
        slots = torch.multinomial(weights, batch_size, replacement=True)
        length = min(length, int(self.lengths[slots].min()))
        start = (torch.rand(batch_size) * (self.lengths[slots] - length + 1)).long()
        time = start[:, None] + torch.arange(length)
        env = slots // self.episodes_per_env
        result = {}
        for key, value in self.storage.items():
            if key == "real":
                result[key] = torch.full((batch_size, length, *value.shape[2:]), 0.5, device=device)
                camera = self.camera_inverse[env]
                real = camera >= 0
                camera_slots = camera[real] * self.episodes_per_env + slots[real] % self.episodes_per_env
                if real.any():
                    result[key][real.to(device)] = value[camera_slots[:, None], time[real]].to(device)
            else:
                result[key] = value[slots[:, None], time].to(device)
        result["is_first"] = torch.zeros(batch_size, length, device=device)
        result["is_first"][:, 0] = 1
        return result


class SequenceReplay:
    def __init__(self, num_envs: int, capacity: int, camera_mask=None):
        self.num_envs = num_envs
        self.capacity = max(2, capacity // num_envs)
        self.cursor = self.size = 0
        self.storage = {}
        self.camera_ids = None if camera_mask is None else camera_mask.cpu().nonzero(as_tuple=False).flatten()
        self.camera_inverse = torch.full((num_envs,), -1, dtype=torch.long)
        if self.camera_ids is not None:
            self.camera_inverse[self.camera_ids] = torch.arange(len(self.camera_ids))

    @property
    def total_samples(self):
        return self.size * self.num_envs

    def insert(self, frame):
        for key, value in frame.items():
            value = value.detach().cpu()
            if key == "real" and self.camera_ids is not None:
                value = value[self.camera_ids]
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
        result = {}
        for key, value in self.storage.items():
            if key == "real" and self.camera_ids is not None and not len(self.camera_ids):
                result[key] = torch.full((batch_size, length, *value.shape[2:]), 0.5, device=device)
                continue
            ids = self.camera_inverse[env].clamp_min(0) if key == "real" and self.camera_ids is not None else env
            result[key] = value[time, ids[:, None]].to(device)
        result["is_first"][:, 0] = 1
        return result
