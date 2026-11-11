# SPDX-License-Identifier: Apache-2.0
"""上游显式 PD：先裁剪力矩，再施加物理子步电机强度。"""

import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass

from wmp_lab.control import unitree_go2_clip_effort


class UpstreamPDActuator(IdealPDActuator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.effort_scale = torch.ones_like(self.effort_limit)

    def compute(self, control_action, joint_pos, joint_vel):
        result = super().compute(control_action, joint_pos, joint_vel)
        self.applied_effort.mul_(self.effort_scale)
        return result


@configclass
class UpstreamPDActuatorCfg(IdealPDActuatorCfg):
    class_type: type = UpstreamPDActuator


class UnitreeGo2Actuator(IdealPDActuator):
    """Official Go2HV curve with WMP's existing substep delay handled by the env.

    Unitree's DelayedPD default adds no delay. In this integration the environment
    retains its reference 0/1/2-substep delay, so another delay buffer is unnecessary.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._joint_vel = torch.zeros_like(self.computed_effort)
        self.effort_scale = torch.ones_like(self.effort_limit)
        self._gear_ratio = self._parse_joint_parameter(getattr(self.cfg, "gear_ratio", 1.0), 1.0)
        if not torch.isfinite(self._gear_ratio).all() or not (self._gear_ratio > 0).all():
            raise ValueError("Go2 joint gear ratios must be finite and positive")
        # Resolve the immutable profile once, avoiding a GPU sync in every physics substep.
        self._unit_gear = bool((self._gear_ratio == 1.0).all())

    def compute(self, control_action, joint_pos, joint_vel):
        self._joint_vel.copy_(joint_vel)
        result = super().compute(control_action, joint_pos, joint_vel)
        self.applied_effort.mul_(self.effort_scale)
        result.joint_efforts = self.applied_effort
        return result

    def _clip_effort(self, effort):
        if self._unit_gear:
            return unitree_go2_clip_effort(effort, self._joint_vel)
        return unitree_go2_clip_effort(effort, self._joint_vel, gear_ratio=self._gear_ratio)


@configclass
class UnitreeGo2ActuatorCfg(IdealPDActuatorCfg):
    class_type: type = UnitreeGo2Actuator
    gear_ratio: float | dict[str, float] = 1.0
