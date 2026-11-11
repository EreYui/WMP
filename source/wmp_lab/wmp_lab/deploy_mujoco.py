# SPDX-License-Identifier: Apache-2.0
"""独立 MuJoCo 策略部署、动作回放与参考动作采集。"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .robots import (
    CONTROL_REVISION,
    ENVIRONMENT_REVISION,
    GO2_CONTROL_REVISION,
    GO2_JOINT_CONTROL_REVISION,
    LEGACY_CONTROL_REVISION,
    default_control_revision,
    robot_profile,
)


def deployment_torque(profile, control_revision, target, position, velocity):
    """按保存的控制版本计算力矩；新 Go2 使用官方扭速包络。"""
    if control_revision == GO2_CONTROL_REVISION:
        from .control import unitree_go2_torque

        return unitree_go2_torque(
            target, position, velocity, stiffness=profile.stiffness, damping=profile.damping,
        )
    if control_revision == GO2_JOINT_CONTROL_REVISION:
        from .control import unitree_go2_torque

        return unitree_go2_torque(
            target, position, velocity, stiffness=profile.stiffness, damping=profile.damping,
            gear_ratio=profile.joint_gear_ratios,
        )
    if control_revision not in (CONTROL_REVISION, LEGACY_CONTROL_REVISION):
        raise ValueError(f"unsupported control revision: {control_revision}")
    torque = np.asarray(profile.stiffness) * (np.asarray(target) - position) - np.asarray(profile.damping) * velocity
    limits = np.asarray(profile.torque_limits)
    return np.clip(torque, -limits, limits)


class MujocoRobot:
    def __init__(self, robot: str, control_revision=None, environment_revision=ENVIRONMENT_REVISION):
        import mujoco

        self.mj = mujoco
        control_revision = default_control_revision(robot) if control_revision is None else control_revision
        self.profile = robot_profile(
            robot, control_revision=control_revision, environment_revision=environment_revision,
        )
        self.control_revision = control_revision
        root = Path(__file__).parent / "assets/robots" / robot
        source = root / ("xml/go1.xml" if robot == "go1" else "go2.xml")
        tree = ET.parse(source)
        model = tree.getroot()
        model.find("compiler").set("meshdir", str(root / ("meshes" if robot == "go1" else "mujoco_meshes")))
        model.find("option").set("timestep", "0.005")
        world = model.find("worldbody")
        ET.SubElement(world, "geom", name="ground", type="plane", size="40 40 .1", friction="1 0.005 0.0001")
        ET.SubElement(world, "light", pos="0 0 5", dir="0 0 -1")
        body = world.find("body")
        pitch = np.deg2rad(self.profile.camera_pitch_deg)
        ET.SubElement(
            body,
            "camera",
            name="wmp_depth",
            pos=" ".join(map(str, self.profile.camera_position)),
            xyaxes=f"0 -1 0 {np.sin(pitch)} 0 {np.cos(pitch)}",
            fovy="58",
        )
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(model, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.joint_ids = np.asarray(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.profile.joint_order]
        )
        if np.any(self.joint_ids < 0):
            raise RuntimeError("MuJoCo joint mapping failed")
        self.q_ids = self.model.jnt_qposadr[self.joint_ids]
        self.v_ids = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = np.asarray(
            [np.flatnonzero(self.model.actuator_trnid[:, 0] == joint)[0] for joint in self.joint_ids]
        )
        if control_revision in (CONTROL_REVISION, GO2_CONTROL_REVISION, GO2_JOINT_CONTROL_REVISION):
            limits = np.asarray(self.profile.torque_limits)
            self.model.actuator_ctrlrange[self.actuator_ids] = np.stack((-limits, limits), -1)
            self.model.actuator_forcerange[self.actuator_ids] = np.stack((-limits, limits), -1)
        self.base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body.get("name"))
        self.default_q = np.asarray(self.profile.default_joint_pos)
        self.renderer = None
        self.reset()

    def reset(self):
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = (0, 0, self.profile.spawn_height)
        self.data.qpos[3:7] = (1, 0, 0, 0)
        self.data.qpos[self.q_ids] = self.default_q
        self.mj.mj_forward(self.model, self.data)

    def body_velocity(self):
        velocity = np.zeros(6)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_BODY, self.base_id, velocity, 1)
        return velocity[3:], velocity[:3]

    def observation(self, command, previous_action):
        _, angular = self.body_velocity()
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        gravity = rotation.T @ np.array((0, 0, -1))
        return np.concatenate(
            (
                angular * 0.25,
                gravity,
                np.asarray(command) * np.array((1, 1, 0.25)),
                self.data.qpos[self.q_ids] - self.default_q,
                self.data.qvel[self.v_ids] * 0.05,
                previous_action,
            )
        ).astype(np.float32)

    def depth(self):
        if self.renderer is None:
            self.renderer = self.mj.Renderer(self.model, height=64, width=64)
            self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera="wmp_depth")
        return np.nan_to_num(self.renderer.render(), nan=2, posinf=2).clip(0, 2) / 2 - 0.5

    def step(self, action):
        target = self.default_q + self.profile.action_scale * np.clip(action, -6, 6)
        for _ in range(4):
            self.data.ctrl[self.actuator_ids] = deployment_torque(
                self.profile, self.control_revision, target,
                self.data.qpos[self.q_ids], self.data.qvel[self.v_ids],
            )
            self.mj.mj_step(self.model, self.data)

    def motion_frame(self):
        linear, angular = self.body_velocity()
        return np.concatenate(
            (
                self.data.qpos[:3],
                self.data.qpos[3:7][[1, 2, 3, 0]],
                self.data.qpos[self.q_ids],
                np.zeros(12),
                linear,
                angular,
                self.data.qvel[self.v_ids],
                np.zeros(12),
            )
        ).tolist()

    def close(self):
        if self.renderer:
            self.renderer.close()
