# SPDX-License-Identifier: Apache-2.0
"""注册共享实现的 Go1/Go2 IsaacLab 任务。"""

import gymnasium as gym

from .direct.wmp_env import WMPEnv
from .direct.wmp_env_cfg import (
    WMPGo1RoughEnvCfg,
    WMPGo1RoughSmokeEnvCfg,
    WMPGo2RoughEnvCfg,
    WMPGo2RoughSmokeEnvCfg,
)

for robot, full, smoke in (
    ("Go1", WMPGo1RoughEnvCfg, WMPGo1RoughSmokeEnvCfg),
    ("Go2", WMPGo2RoughEnvCfg, WMPGo2RoughSmokeEnvCfg),
):
    for suffix, cfg in (("v0", full), ("Smoke-v0", smoke)):
        task = f"Isaac-WMP-{robot}-Rough-Direct-{suffix}"
        if task not in gym.registry:
            gym.register(
                id=task,
                entry_point="wmp_lab.tasks.direct.wmp_env:WMPEnv",
                disable_env_checker=True,
                order_enforce=False,
                kwargs={"env_cfg_entry_point": cfg},
            )

__all__ = ["WMPEnv"]
