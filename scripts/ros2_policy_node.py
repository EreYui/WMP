#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ROS2/Gazebo/实机传感器到关节目标桥。用法：python scripts/ros2_policy_node.py --policy go1_policy.pt。

输入为 JointState、Imu、Twist 和深度 Image；输出 JointState 目标，电机/控制器驱动由外部部署包提供。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/wmp_lab"))

from wmp_lab.deployment_depth import ReferenceDepthBuffer  # noqa: E402
from wmp_lab.robots import (  # noqa: E402
    CONTROL_REVISION,
    ENVIRONMENT_REVISION,
    GO2_CONTROL_REVISION,
    GO2_JOINT_CONTROL_REVISION,
    LEGACY_CONTROL_REVISION,
)


def validate_deployment_metadata(metadata):
    """仅按导出参数解释策略；不从当前机器人的默认配置覆盖旧策略。"""
    profile = metadata["robot"]
    revision = metadata.get("control_revision", LEGACY_CONTROL_REVISION)
    if metadata["batch_size"] != 1:
        raise ValueError("ROS deployment requires a single-robot policy")
    if revision not in (CONTROL_REVISION, GO2_CONTROL_REVISION, GO2_JOINT_CONTROL_REVISION, LEGACY_CONTROL_REVISION):
        raise ValueError(f"unsupported control revision: {revision}")
    go2_only = revision in (GO2_CONTROL_REVISION, GO2_JOINT_CONTROL_REVISION)
    if profile["name"] not in ("go1", "go2") or (go2_only and profile["name"] != "go2"):
        raise ValueError("policy robot and control revision disagree")
    names = profile["joint_order"]
    pose = np.asarray(profile["default_joint_pos"], dtype=np.float64)
    scale = np.asarray(profile["action_scale"], dtype=np.float64)
    if len(names) != 12 or len(set(names)) != 12 or pose.shape != (12,):
        raise ValueError("policy metadata must specify 12 unique joints and their default positions")
    if (
        scale.shape not in ((), (12,)) or not np.isfinite(pose).all()
        or not np.isfinite(scale).all() or (scale <= 0).any()
    ):
        raise ValueError("invalid policy default positions or action scale")
    if revision == GO2_JOINT_CONTROL_REVISION:
        gear = np.asarray(profile.get("joint_gear_ratios", []), dtype=np.float64)
        if gear.shape != (12,) or not np.isfinite(gear).all() or (gear <= 0).any():
            raise ValueError("v2 deployment requires 12 positive finite joint gear ratios")
    return profile


def policy_joint_target(profile, action):
    """动作仅保持训练的 ±6 裁剪，不增加关节目标硬限幅。"""
    return np.asarray(profile["default_joint_pos"]) + np.asarray(profile["action_scale"]) * np.clip(action, -6, 6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument("--imu-topic", default="/imu/data")
    parser.add_argument("--depth-topic", default="/camera/depth/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/depth/camera_info")
    parser.add_argument(
        "--normalized-depth", action="store_true", help="Input is already 64x64 normalized network depth"
    )
    parser.add_argument("--command-topic", default="/cmd_vel")
    parser.add_argument("--target-topic", default="/wmp/joint_targets")
    args, ros_args = parser.parse_known_args()
    import cv2
    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, Imu, JointState

    metadata = json.loads(args.policy.with_suffix(".json").read_text())
    profile = validate_deployment_metadata(metadata)
    policy = torch.jit.load(str(args.policy))
    policy.reset()

    class PolicyNode(Node):
        def __init__(self):
            super().__init__("wmp_policy")
            self.get_logger().info(
                "Position-target control contract: "
                f"revision={metadata.get('control_revision', LEGACY_CONTROL_REVISION)}, "
                f"actuator={profile.get('actuator_model', 'unspecified')}, "
                f"Kp={profile['stiffness']}, Kd={profile['damping']}, "
                f"torque_speed_curve={profile.get('torque_speed_curve')}, "
                f"joint_gear_ratios={profile.get('joint_gear_ratios', [1.0] * 12)}; "
                "external motor control must match; this bridge publishes joint position targets only."
            )
            if metadata.get("control_revision") == GO2_JOINT_CONTROL_REVISION:
                self.get_logger().info(
                    "v2 experimental transmission: joint PD torque / g -> motor curve; "
                    "motor velocity = joint velocity * g; joint output torque = motor output * g. "
                    "This node does not configure hardware motor controllers."
                )
            self.joint = self.imu = self.depth = None
            self.received = {}
            self.camera_info = None
            self.command = np.zeros(3, dtype=np.float32)
            self.previous = np.zeros(12, dtype=np.float32)
            self.delayed_depth = np.full((1, 64, 64, 1), 0.5, dtype=np.float32)
            self.reference_depth = (
                ReferenceDepthBuffer() if metadata.get("environment_revision") == ENVIRONMENT_REVISION else None
            )
            self.step_count = 0
            self.was_stale = True
            self.publisher = self.create_publisher(JointState, args.target_topic, 10)
            self.create_subscription(JointState, args.joint_topic, self.on_joint, qos_profile_sensor_data)
            self.create_subscription(Imu, args.imu_topic, self.on_imu, qos_profile_sensor_data)
            self.create_subscription(Image, args.depth_topic, self.on_depth, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, args.camera_info_topic, self.on_camera_info, qos_profile_sensor_data)
            self.create_subscription(Twist, args.command_topic, self.on_command, 10)
            self.create_timer(0.02, self.tick)

        def on_joint(self, message):
            if not all(name in message.name for name in profile["joint_order"]):
                return
            ids = [message.name.index(name) for name in profile["joint_order"]]
            if len(message.position) < len(message.name) or len(message.velocity) < len(message.name):
                return
            position, velocity = np.asarray(message.position)[ids], np.asarray(message.velocity)[ids]
            if not np.isfinite(position).all() or not np.isfinite(velocity).all():
                return
            self.joint = (position, velocity)
            self.received["joint"] = time.monotonic()

        def on_imu(self, message):
            quat = np.array(
                (message.orientation.w, message.orientation.x, message.orientation.y, message.orientation.z)
            )
            if not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-6:
                return
            w, x, y, z = quat / np.linalg.norm(quat)
            gravity = np.array((2 * (w * y - x * z), -2 * (y * z + w * x), -1 + 2 * (x * x + y * y)))
            angular = np.array((message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z))
            if not np.isfinite(angular).all():
                return
            self.imu = (angular, gravity)
            self.received["imu"] = time.monotonic()

        def on_camera_info(self, message):
            if message.k[0] > 0 and message.k[4] > 0:
                self.camera_info = tuple(message.k)

        def on_depth(self, message):
            if message.encoding not in ("32FC1", "16UC1"):
                return
            dtype = (
                np.dtype(">f4" if message.is_bigendian else "<f4")
                if message.encoding == "32FC1"
                else np.dtype(">u2" if message.is_bigendian else "<u2")
            )
            stride = message.step // dtype.itemsize
            values = np.frombuffer(bytes(message.data), dtype=dtype).reshape(message.height, stride)[:, : message.width]
            values = values.astype(np.float32)
            if args.normalized_depth:
                if values.shape != (64, 64) or not np.isfinite(values).all():
                    return
                self.depth = values.clip(-0.5, 0.5).reshape(1, 64, 64, 1)
            else:
                if self.camera_info is None:
                    return
                values *= 0.001 if message.encoding == "16UC1" else 1.0
                values = np.where((values > 0) & np.isfinite(values), values, 2.0)
                k = self.camera_info
                focal = 64 / (2 * np.tan(np.deg2rad(58) / 2))
                u, v = np.meshgrid(np.arange(64, dtype=np.float32) + 0.5, np.arange(64, dtype=np.float32) + 0.5)
                map_x = (u - 32) / focal * k[0] + k[2]
                map_y = (v - 32) / focal * k[4] + k[5]
                values = cv2.remap(
                    values, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=2.0
                )
                self.depth = (values.clip(0, 2) / 2 - 0.5).reshape(1, 64, 64, 1)
            self.received["depth"] = time.monotonic()

        def on_command(self, message):
            self.command = np.array((message.linear.x, message.linear.y, message.angular.z), dtype=np.float32)

        def tick(self):
            if len(self.received) < 3 or any(
                time.monotonic() - timestamp > 0.3 for timestamp in self.received.values()
            ):
                if not self.was_stale:
                    self.get_logger().warning("Sensor stream stale; waiting for fresh inputs.")
                self.was_stale = True
                return
            if self.was_stale:
                policy.reset()
                self.previous[:] = 0
                self.step_count = 0
                if self.reference_depth is not None:
                    self.reference_depth.reset()
                self.was_stale = False
            position, velocity = self.joint
            angular, gravity = self.imu
            obs = np.concatenate(
                (
                    angular * 0.25,
                    gravity,
                    self.command * (1, 1, 0.25),
                    position - np.array(profile["default_joint_pos"]),
                    velocity * 0.05,
                    self.previous,
                )
            )
            with torch.inference_mode():
                action = policy(
                    torch.tensor(obs[None], dtype=torch.float32),
                    torch.from_numpy(
                        self.reference_depth.delayed if self.reference_depth is not None else self.delayed_depth
                    ),
                ).numpy()[0]
            if self.reference_depth is not None:
                if self.reference_depth.capture_due(self.step_count):
                    self.reference_depth.capture(self.depth)
            elif self.step_count % 5 == 0:
                self.delayed_depth = self.depth.copy()
            self.step_count += 1
            if not np.isfinite(action).all():
                self.get_logger().error("Non-finite policy action; output suppressed.")
                return
            self.previous = np.clip(action, -6, 6)
            target = JointState()
            target.header.stamp = self.get_clock().now().to_msg()
            target.name = profile["joint_order"]
            target.position = policy_joint_target(profile, self.previous).tolist()
            self.publisher.publish(target)

    rclpy.init(args=ros_args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
