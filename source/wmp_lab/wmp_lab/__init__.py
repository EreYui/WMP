# SPDX-License-Identifier: Apache-2.0
"""Go1/Go2 共用的 WMP IsaacLab 扩展。

Learning modules intentionally do not import Isaac Sim. Importing :mod:`wmp_lab`
registers the Gym tasks when IsaacLab and Gymnasium are available.
"""

from .constants import AMP_OBSERVATION_DIM, GO1_JOINT_ORDER, GO2_JOINT_ORDER, JOINT_ORDER, OBSERVATION_LAYOUT

try:
    from . import tasks as tasks
except (ImportError, ModuleNotFoundError):  # permits pure PyTorch tooling outside Isaac Sim
    tasks = None

__all__ = ["AMP_OBSERVATION_DIM", "JOINT_ORDER", "GO1_JOINT_ORDER", "GO2_JOINT_ORDER", "OBSERVATION_LAYOUT"]
