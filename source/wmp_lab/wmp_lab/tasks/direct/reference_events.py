# SPDX-License-Identifier: Apache-2.0
"""参考任务每个机器人共用一个摩擦桶，静摩擦与动摩擦一致。"""

import torch


def randomize_reference_material(env, env_ids=None):
    robot = env.scene["robot"]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs)
    env_ids = env_ids.cpu()
    properties = robot.root_physx_view.get_material_properties()
    buckets = torch.empty(64).uniform_(0.5, 2.0)
    friction = buckets[torch.randint(64, (len(env_ids),))]
    properties[env_ids, :, 0] = friction[:, None]
    properties[env_ids, :, 1] = friction[:, None]
    properties[env_ids, :, 2] = 0
    robot.root_physx_view.set_material_properties(properties, env_ids)
