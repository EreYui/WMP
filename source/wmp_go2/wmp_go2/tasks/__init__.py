# SPDX-License-Identifier: Apache-2.0
"""Register WMP Go2 direct-workflow Gym tasks."""

import gymnasium as gym

from .direct.go2_env import WMPGo2Env
from .direct.go2_env_cfg import WMPGo2RoughEnvCfg, WMPGo2RoughSmokeEnvCfg

_TASKS = {
    "Isaac-WMP-Go2-Rough-Direct-v0": WMPGo2RoughEnvCfg,
    "Isaac-WMP-Go2-Rough-Direct-Smoke-v0": WMPGo2RoughSmokeEnvCfg,
}

for task_id, cfg_class in _TASKS.items():
    if task_id not in gym.registry:
        gym.register(
            id=task_id,
            entry_point="wmp_go2.tasks.direct.go2_env:WMPGo2Env",
            disable_env_checker=True,
            order_enforce=False,
            kwargs={"env_cfg_entry_point": cfg_class},
        )

__all__ = ["WMPGo2Env", "WMPGo2RoughEnvCfg", "WMPGo2RoughSmokeEnvCfg"]
