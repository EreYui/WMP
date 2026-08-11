# SPDX-License-Identifier: Apache-2.0
"""Stable WMP tensor and robot ordering contracts."""

from types import MappingProxyType

LEG_ORDER = ("FL", "FR", "RL", "RR")
JOINT_TYPE_ORDER = ("hip", "thigh", "calf")
GO2_JOINT_ORDER = tuple(f"{leg}_{joint}_joint" for leg in LEG_ORDER for joint in JOINT_TYPE_ORDER)

PRIVILEGED_DIM = 53
PROPRIOCEPTION_DIM = 33
ACTION_DIM = 12
HEIGHT_DIM = 187
FORWARD_HEIGHT_DIM = 525
OBSERVATION_DIM = 285
HISTORY_STEPS = 5
HISTORY_FRAME_DIM = 42
HISTORY_DIM = HISTORY_STEPS * HISTORY_FRAME_DIM
AMP_OBSERVATION_DIM = 30
DEPTH_SHAPE = (64, 64, 1)
WORLD_MODEL_ACTION_HISTORY_STEPS = 5
WORLD_MODEL_ACTION_DIM = ACTION_DIM * WORLD_MODEL_ACTION_HISTORY_STEPS

OBSERVATION_LAYOUT = MappingProxyType(
    {
        "privileged": slice(0, 53),
        "proprioception": slice(53, 86),
        "previous_action": slice(86, 98),
        "height": slice(98, 285),
    }
)


def validate_contracts() -> None:
    """Fail early if a tensor layout is edited inconsistently."""
    assert len(GO2_JOINT_ORDER) == ACTION_DIM
    assert sum(s.stop - s.start for s in OBSERVATION_LAYOUT.values()) == OBSERVATION_DIM
    assert HISTORY_FRAME_DIM == PROPRIOCEPTION_DIM + 3 + 6


validate_contracts()
