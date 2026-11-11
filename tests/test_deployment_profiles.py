from wmp_lab.learning.progress_rewards import checkpoint_reward_revision
# SPDX-License-Identifier: Apache-2.0
"""验证新 Go2 与历史控制版本的部署参数路由，不启动模拟器或 ROS。"""

import ast
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from wmp_lab.deploy_mujoco import deployment_torque
from wmp_lab.learning.config import SMOKE_CFG
from wmp_lab.learning.deployment import (
    deployment_actuation_metadata,
    deployment_environment_metadata,
    export_policy,
)
from wmp_lab.learning.runner import WMPRunner
from wmp_lab.robots import (
    CONTROL_REVISION,
    ENVIRONMENT_REVISION,
    GO2_CONTROL_REVISION,
    GO2_JOINT_CONTROL_REVISION,
    LEGACY_CONTROL_REVISION,
    LEGACY_ENVIRONMENT_REVISION,
    checkpoint_control_revision,
    default_control_revision,
    robot_metadata,
    robot_profile,
)

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name.replace("/", "_"), ROOT / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _function(script, name, namespace):
    tree = ast.parse((ROOT / script).read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / script), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("robot", ["go1", "go2"])
@pytest.mark.parametrize("revision", [CONTROL_REVISION, LEGACY_CONTROL_REVISION])
def test_historical_deployment_keeps_fixed_pd_and_limits(robot, revision):
    profile = robot_profile(robot, control_revision=revision)
    position = np.linspace(-0.8, 0.8, 12)
    velocity = np.linspace(-31.0, 31.0, 12)
    target = position + np.linspace(-2.0, 2.0, 12)
    expected = np.clip(
        40.0 * (target - position) - velocity,
        -np.asarray(profile.torque_limits), np.asarray(profile.torque_limits),
    )
    np.testing.assert_array_equal(deployment_torque(profile, revision, target, position, velocity), expected)
    if robot == "go2" and revision == CONTROL_REVISION:
        assert profile.torque_limits == (23.7, 23.7, 35.55) * 4
        assert profile.velocity_limits == (30.1, 30.1, 20.07) * 4


def test_unitree_deployment_uses_directional_torque_speed_curve():
    profile = robot_profile("go2", control_revision=GO2_CONTROL_REVISION)
    velocity = np.array([0.0, 10.0, 20.0, 20.0, -20.0, -20.0, 30.0, 35.0])
    desired = np.array([40.0, 40.0, 40.0, -40.0, -40.0, 40.0, 40.0, 40.0])
    target = (desired + 0.5 * velocity) / 25.0
    actual = deployment_torque(profile, GO2_CONTROL_REVISION, target, np.zeros(8), velocity)
    expected = np.array([23.4, 20.2, 20.2 * 10.0 / 16.5, -23.4 * 10.0 / 16.5,
                         -20.2 * 10.0 / 16.5, 23.4 * 10.0 / 16.5, 0.0, 0.0])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_unknown_torque_revision_is_rejected():
    profile = robot_profile("go1", control_revision=CONTROL_REVISION)
    with pytest.raises(ValueError, match="unsupported control revision"):
        deployment_torque(profile, "unrecognized", np.zeros(12), np.zeros(12), np.zeros(12))


def test_saved_environment_metadata_preserves_solver_and_source_snapshot():
    contract = dict(
        fingerprint="saved-fingerprint", source={"robots.py": "source-hash"}, assets={"go2.usd": "asset-hash"},
        config=dict(decimation=4, sim=dict(dt=0.005, physx=dict(min_position_iteration_count=8,
                                                            max_velocity_iteration_count=4))),
    )
    result = deployment_environment_metadata(SimpleNamespace(environment_contract=contract))
    assert result["environment_contract_fingerprint"] == "saved-fingerprint"
    assert result["simulation"]["physics_dt"] == 0.005
    assert result["simulation"]["physx"]["min_position_iteration_count"] == 8
    assert result["source_sha256"] == {"robots.py": "source-hash"}
    assert result["asset_sha256"] == {"go2.usd": "asset-hash"}
    result["source_sha256"]["robots.py"] = "changed"
    assert contract["source"]["robots.py"] == "source-hash"
    missing = deployment_environment_metadata(SimpleNamespace())
    assert missing["simulation"]["physics_dt"] is None
    assert missing["simulation"]["physx"] == {}


def test_train_restores_saved_revision_and_rejects_conflicting_contract():
    resolve = _function("scripts/train.py", "resolve_training_control_revision", dict(
        checkpoint_control_revision=checkpoint_control_revision,
        default_control_revision=default_control_revision,
        LEGACY_CONTROL_REVISION=LEGACY_CONTROL_REVISION, robot_profile=robot_profile,
    ))
    assert resolve("go1", None) == CONTROL_REVISION
    assert resolve("go2", None) == GO2_JOINT_CONTROL_REVISION
    for revision in (CONTROL_REVISION, GO2_CONTROL_REVISION, GO2_JOINT_CONTROL_REVISION):
        saved = dict(extra=dict(
            control_revision=revision, environment_contract=dict(config=dict(control_revision=revision)),
        ))
        assert resolve("go2", saved) == revision
    saved["extra"]["environment_contract"]["config"]["control_revision"] = CONTROL_REVISION
    with pytest.raises(ValueError, match="disagrees"):
        resolve("go2", saved)
    with pytest.raises(ValueError, match="playback only"):
        resolve("go2", dict(version=2))


def test_playback_never_replaces_old_revision_with_new_robot_default():
    options = _function("scripts/play.py", "checkpoint_environment_options", dict(
        LEGACY_CONTROL_REVISION=LEGACY_CONTROL_REVISION, LEGACY_ENVIRONMENT_REVISION=LEGACY_ENVIRONMENT_REVISION,
        checkpoint_reward_revision=checkpoint_reward_revision,
    ))
    for revision in (LEGACY_CONTROL_REVISION, CONTROL_REVISION, GO2_CONTROL_REVISION, GO2_JOINT_CONTROL_REVISION):
        runner = SimpleNamespace(
            control_revision=revision, environment_revision=ENVIRONMENT_REVISION,
            environment_contract=dict(config=dict(control_revision=revision, camera_implementation="wmp_go1")),
        )
        assert options(runner)["control_revision"] == revision
        assert options(runner)["camera_implementation"] == "wmp_go1"
        assert options(runner)["reward_revision"] == "upstream_v3"
    assert options(SimpleNamespace())["control_revision"] == LEGACY_CONTROL_REVISION


def test_mujoco_tool_restores_policy_metadata_before_constructing_robot():
    module = _script("tools/mujoco_deploy.py")
    assert module.deployment_options("go2")["control_revision"] == GO2_JOINT_CONTROL_REVISION
    old = dict(robot=dict(name="go2"), batch_size=1, control_revision=CONTROL_REVISION,
               environment_revision=ENVIRONMENT_REVISION)
    assert module.deployment_options("go2", old)["control_revision"] == CONTROL_REVISION
    with pytest.raises(ValueError, match="does not match"):
        module.deployment_options("go1", old)


@pytest.mark.parametrize(
    "revision", [LEGACY_CONTROL_REVISION, CONTROL_REVISION, GO2_CONTROL_REVISION, GO2_JOINT_CONTROL_REVISION],
)
def test_ros_targets_use_exported_pose_and_scale_without_new_hard_clamp(revision):
    module = _script("scripts/ros2_policy_node.py")
    metadata = dict(batch_size=1, control_revision=revision,
                    robot=json.loads(json.dumps(robot_metadata("go2", revision, ENVIRONMENT_REVISION))))
    profile = module.validate_deployment_metadata(metadata)
    action = np.linspace(-8.0, 8.0, 12)
    np.testing.assert_allclose(
        module.policy_joint_target(profile, action),
        np.asarray(profile["default_joint_pos"]) + np.asarray(profile["action_scale"]) * np.clip(action, -6, 6),
    )
    # 导出参数是权威值，不能悄悄换成当前默认站姿。
    profile["default_joint_pos"][0] += 0.03
    assert module.policy_joint_target(profile, np.zeros(12))[0] == profile["default_joint_pos"][0]


def test_ros_rejects_invalid_control_robot_and_scale():
    module = _script("scripts/ros2_policy_node.py")
    metadata = dict(batch_size=1, control_revision=GO2_CONTROL_REVISION,
                    robot=robot_metadata("go1", CONTROL_REVISION, ENVIRONMENT_REVISION))
    with pytest.raises(ValueError, match="disagree"):
        module.validate_deployment_metadata(metadata)
    metadata["control_revision"] = CONTROL_REVISION
    metadata["robot"]["action_scale"] = float("nan")
    with pytest.raises(ValueError, match="invalid"):
        module.validate_deployment_metadata(metadata)


def test_old_go2_checkpoint_export_retains_saved_profile_after_default_switch(tmp_path):
    runner = WMPRunner(replace(SMOKE_CFG, robot="go2"))
    runner.control_revision = CONTROL_REVISION
    checkpoint = runner.save(tmp_path / "old_go2.pt")
    restored = WMPRunner.from_checkpoint(checkpoint)
    assert restored.control_revision == CONTROL_REVISION
    path, error = export_policy(restored, tmp_path / "old_go2_policy.pt")
    assert error < 1e-4
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["format_version"] == 4
    assert metadata["control_revision"] == CONTROL_REVISION
    assert metadata["robot"]["stiffness"] == 40.0
    assert metadata["robot"]["damping"] == 1.0
    assert metadata["robot"]["torque_limits"] == [23.7, 23.7, 35.55] * 4
    assert metadata["robot"]["velocity_limits"] == [30.1, 30.1, 20.07] * 4


def test_deployment_gear_mapping_is_v2_only(monkeypatch):
    import wmp_lab.control as control

    calls = []

    def capture(target, position, velocity, **kwargs):
        calls.append(kwargs)
        return np.zeros_like(target)

    monkeypatch.setattr(control, "unitree_go2_torque", capture)
    profile = SimpleNamespace(stiffness=25.0, damping=0.5, joint_gear_ratios=(1.0, 1.0, 45.43 / 23.7) * 4)
    zeros = np.zeros(12)
    deployment_torque(profile, GO2_CONTROL_REVISION, zeros, zeros, zeros)
    deployment_torque(profile, GO2_JOINT_CONTROL_REVISION, zeros, zeros, zeros)
    assert "gear_ratio" not in calls[0]
    assert calls[1]["gear_ratio"] == profile.joint_gear_ratios


def test_v2_joint_curve_has_more_low_speed_torque_and_lower_no_load_speed():
    profile = robot_profile("go2", control_revision=GO2_JOINT_CONTROL_REVISION)
    gear = np.asarray(profile.joint_gear_ratios)
    expected_gear = np.asarray((1.0, 1.0, 45.43 / 23.7) * 4)
    np.testing.assert_allclose(gear, expected_gear, rtol=0, atol=1e-12)
    zeros = np.zeros(12)
    low = deployment_torque(profile, GO2_JOINT_CONTROL_REVISION, np.full(12, 4.0), zeros, zeros)
    np.testing.assert_allclose(low, 23.4 * gear, rtol=0, atol=1e-12)
    velocity = np.full(12, 10.0)
    target = (100.0 + 0.5 * velocity) / 25.0
    actual = deployment_torque(profile, GO2_JOINT_CONTROL_REVISION, target, zeros, velocity)
    motor_limit = np.where(10.0 * gear < 13.5, 20.2, 20.2 * np.maximum(30.0 - 10.0 * gear, 0) / 16.5)
    np.testing.assert_allclose(actual, motor_limit * gear, rtol=0, atol=1e-12)
    velocity[:] = 16.0
    target = (100.0 + 0.5 * velocity) / 25.0
    actual = deployment_torque(profile, GO2_JOINT_CONTROL_REVISION, target, zeros, velocity)
    np.testing.assert_array_equal(actual[[2, 5, 8, 11]], np.zeros(4))
    assert (actual[[0, 1, 3, 4, 6, 7, 9, 10]] > 0).all()


def test_v2_export_records_motor_and_joint_spaces_without_changing_old_exports(tmp_path):
    runner = WMPRunner(replace(SMOKE_CFG, robot="go2"))
    assert runner.control_revision == GO2_JOINT_CONTROL_REVISION
    path, error = export_policy(runner, tmp_path / "v2_policy.pt")
    assert error < 1e-4
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["format_version"] == 4
    assert metadata["control_revision"] == GO2_JOINT_CONTROL_REVISION
    semantics = metadata["actuation_semantics"]
    assert semantics["pd_torque_space"] == "joint"
    assert semantics["torque_speed_curve_space"] == "motor"
    assert semantics["joint_gear_ratios"] == metadata["robot"]["joint_gear_ratios"]
    assert semantics["motor_requested_torque"] == "joint_pd_torque / gear_ratio"
    for revision in (CONTROL_REVISION, GO2_CONTROL_REVISION, LEGACY_CONTROL_REVISION):
        assert deployment_actuation_metadata({}, revision) == {}


def test_old_v1_checkpoint_export_keeps_identity_curve_after_default_switch(tmp_path):
    runner = WMPRunner(replace(SMOKE_CFG, robot="go2"))
    runner.control_revision = GO2_CONTROL_REVISION
    checkpoint = runner.save(tmp_path / "v1_go2.pt")
    restored = WMPRunner.from_checkpoint(checkpoint)
    path, error = export_policy(restored, tmp_path / "v1_go2_policy.pt")
    assert error < 1e-4
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["control_revision"] == GO2_CONTROL_REVISION
    assert metadata["robot"]["actuator_model"] == "unitree_go2hv"
    assert metadata["robot"]["stiffness"] == 25.0
    assert metadata["robot"]["damping"] == 0.5
    assert metadata["robot"]["torque_limits"] == [23.4] * 12
    assert "actuation_semantics" not in metadata


@pytest.mark.parametrize("bad_gears", [[], [1.0] * 11, [float("nan")] * 12, [0.0] * 12])
def test_ros_v2_rejects_missing_or_invalid_transmission(bad_gears):
    module = _script("scripts/ros2_policy_node.py")
    profile = robot_metadata("go2", GO2_JOINT_CONTROL_REVISION, ENVIRONMENT_REVISION)
    metadata = dict(
        batch_size=1, control_revision=GO2_JOINT_CONTROL_REVISION,
        robot=json.loads(json.dumps(profile)),
    )
    metadata["robot"]["joint_gear_ratios"] = bad_gears
    with pytest.raises(ValueError, match="gear ratios"):
        module.validate_deployment_metadata(metadata)
