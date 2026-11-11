# SPDX-License-Identifier: Apache-2.0
"""按上游真实关节名称验证限制与历史控制版本选择。"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch
from wmp_lab.learning.config import SMOKE_CFG
from wmp_lab.learning.runner import WMPRunner
from wmp_lab.robots import CONTROL_REVISION, LEGACY_CONTROL_REVISION, checkpoint_control_revision, robot_profile


@pytest.mark.parametrize("robot", ["go1", "go2"])
def test_joint_limits_match_original_urdf_by_name(robot):
    upstream = Path(__file__).resolve().parents[2] / "wmp_go1/resources/robots"
    path = upstream / robot / "urdf" / ("go1_new_2.urdf" if robot == "go1" else "go2.urdf")
    joints = {joint.attrib["name"]: joint.find("limit") for joint in ET.parse(path).getroot().findall("joint")}
    profile = robot_profile(robot, control_revision=CONTROL_REVISION)
    assert len(profile.torque_limits) == len(profile.velocity_limits) == 12
    for name, torque, velocity in zip(profile.joint_order, profile.torque_limits, profile.velocity_limits, strict=True):
        assert torque == float(joints[name].attrib["effort"])
        assert velocity == float(joints[name].attrib["velocity"])


def test_historical_checkpoints_retain_legacy_control_and_new_ones_are_explicit():
    assert checkpoint_control_revision({"version": 2}) == LEGACY_CONTROL_REVISION
    assert checkpoint_control_revision({"version": 3, "extra": {"environment": None}}) == LEGACY_CONTROL_REVISION
    assert checkpoint_control_revision({"extra": {"control_revision": CONTROL_REVISION}}) == CONTROL_REVISION
    for name, old_limit in (("go1", 23.7), ("go2", 23.5)):
        assert robot_profile(name, legacy=True).torque_limits == (old_limit,) * 12
        assert robot_profile(name, legacy=True).velocity_limits == (30.0,) * 12


def test_runner_control_revision_survives_save_and_old_v3_detection(tmp_path):
    runner = WMPRunner(SMOKE_CFG)
    path = runner.save(tmp_path / "new.pt")
    assert WMPRunner.from_checkpoint(path).control_revision == CONTROL_REVISION
    payload = torch.load(path, weights_only=False)
    del payload["extra"]["control_revision"]
    torch.save(payload, tmp_path / "historical_v3.pt")
    assert WMPRunner.from_checkpoint(tmp_path / "historical_v3.pt").control_revision == LEGACY_CONTROL_REVISION
