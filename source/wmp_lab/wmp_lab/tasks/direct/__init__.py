# SPDX-License-Identifier: Apache-2.0
"""Go1/Go2 共用环境与机器人配置。"""

from .wmp_env import WMPEnv
from .wmp_env_cfg import (
    WMPGo1RoughEnvCfg,
    WMPGo1RoughSmokeEnvCfg,
    WMPGo2RoughEnvCfg,
    WMPGo2RoughSmokeEnvCfg,
    make_env_cfg,
)

WMPGo1Env = WMPGo2Env = WMPEnv
__all__ = [
    "WMPEnv",
    "WMPGo1Env",
    "WMPGo2Env",
    "WMPGo1RoughEnvCfg",
    "WMPGo1RoughSmokeEnvCfg",
    "WMPGo2RoughEnvCfg",
    "WMPGo2RoughSmokeEnvCfg",
    "make_env_cfg",
]
