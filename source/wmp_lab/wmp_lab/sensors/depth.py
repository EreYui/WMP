# SPDX-License-Identifier: Apache-2.0
"""共享环境索引的稀疏深度相机，仅向分配相机的环境发射射线。"""

import torch
from isaaclab.sensors import RayCasterCamera


class SparseRayCasterCamera(RayCasterCamera):
    def __init__(self, cfg, selected):
        self.selected = selected
        super().__init__(cfg)

    def _update_buffers_impl(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        ids = ids[self.selected[ids]]
        if len(ids):
            super()._update_buffers_impl(ids)
