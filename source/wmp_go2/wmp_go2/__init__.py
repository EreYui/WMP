# SPDX-License-Identifier: Apache-2.0
"""WMP Go2 IsaacLab extension.

Learning modules intentionally do not import Isaac Sim. Importing :mod:`wmp_go2`
registers the Gym tasks when IsaacLab and Gymnasium are available.
"""

from .constants import AMP_OBSERVATION_DIM, GO2_JOINT_ORDER, OBSERVATION_LAYOUT

try:
    from . import tasks as tasks
except (ImportError, ModuleNotFoundError):  # permits pure PyTorch tooling outside Isaac Sim
    tasks = None

__all__ = ["AMP_OBSERVATION_DIM", "GO2_JOINT_ORDER", "OBSERVATION_LAYOUT"]
