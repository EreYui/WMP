# SPDX-License-Identifier: Apache-2.0
"""Go2HV torque-speed law shared by simulation and NumPy deployment.

Adapted from Unitree RL Lab, Apache-2.0:
https://github.com/unitreerobotics/unitree_rl_lab/blob/
4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/
source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree_actuators.py
WMP keeps its existing physics-substep delay and motor-strength randomization.
"""

import numpy as np
import torch

UNITREE_GO2_TORQUE_SPEED = {"X1": 13.5, "X2": 30.0, "Y1": 20.2, "Y2": 23.4}


def _arrays(*values):
    """Keep Torch's dtype/device when present; otherwise use NumPy arrays."""
    reference = next((value for value in values if isinstance(value, torch.Tensor)), None)
    if reference is not None:
        dtype = reference.dtype if reference.is_floating_point() else torch.get_default_dtype()
        return tuple(torch.as_tensor(value, dtype=dtype, device=reference.device) for value in values)
    return tuple(np.asarray(value) for value in values)


def unitree_go2_torque_limit(effort, velocity, *, gear_ratio=1.0):
    """Return the nonnegative Go2HV envelope for each requested torque and speed.

    The official curve uses 20.2 Nm in the direction of motion and 23.4 Nm
    otherwise, then falls linearly from 13.5 to 30 rad/s at the motor.
    A positive finite gear ratio maps joint torque/speed to motor torque/speed.
    Ratios must be validated at actuator or deployment construction, rather
    than synchronizing the GPU to validate a constant on every physics step.
    The default ratio retains the original v1 arithmetic path.
    """
    if not isinstance(gear_ratio, (int, float)) or gear_ratio != 1.0:
        joint_effort, joint_velocity, ratio = _arrays(effort, velocity, gear_ratio)
        motor_limit = unitree_go2_torque_limit(joint_effort / ratio, joint_velocity * ratio)
        return motor_limit * ratio
    effort, velocity = _arrays(effort, velocity)
    x1, x2 = UNITREE_GO2_TORQUE_SPEED["X1"], UNITREE_GO2_TORQUE_SPEED["X2"]
    y1, y2 = UNITREE_GO2_TORQUE_SPEED["Y1"], UNITREE_GO2_TORQUE_SPEED["Y2"]
    if isinstance(effort, torch.Tensor):
        peak = torch.where(velocity * effort > 0, torch.full_like(effort, y1), torch.full_like(effort, y2))
        speed = velocity.abs()
        declining = (-peak / (x2 - x1) * (speed - x1) + peak).clamp_min(0.0)
        return torch.where(speed < x1, peak, declining)
    peak = np.where(velocity * effort > 0, y1, y2)
    speed = np.abs(velocity)
    declining = np.maximum(-peak / (x2 - x1) * (speed - x1) + peak, 0.0)
    return np.where(speed < x1, peak, declining)


def unitree_go2_clip_effort(effort, velocity, *, gear_ratio=1.0):
    """Clip a raw PD torque with the official Go2HV envelope."""
    effort, velocity = _arrays(effort, velocity)
    limit = unitree_go2_torque_limit(effort, velocity, gear_ratio=gear_ratio)
    if isinstance(effort, torch.Tensor):
        return torch.clip(effort, -limit, limit)
    return np.clip(effort, -limit, limit)


def unitree_go2_torque(
    target, position, velocity, *, stiffness=25.0, damping=0.5, effort_scale=1.0, gear_ratio=1.0
):
    """Compute PD, apply the Go2HV envelope, then WMP's motor-strength scale.

    This function applies no additional delay or target-position clamp.
    The official Go2HV Fs/Fd values are zero; engine joint friction is
    configured separately and is not subtracted a second time here.
    A non-unit transmission changes both available joint torque and speed.
    """
    target, position, velocity, stiffness, damping, effort_scale = _arrays(
        target, position, velocity, stiffness, damping, effort_scale
    )
    effort = stiffness * (target - position) - damping * velocity
    return unitree_go2_clip_effort(effort, velocity, gear_ratio=gear_ratio) * effort_scale
