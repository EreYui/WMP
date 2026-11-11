# SPDX-License-Identifier: Apache-2.0
"""Go2HV math checked against the pinned official source and installed PD method."""

import ast
import importlib.util
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "source/wmp_lab/wmp_lab"
REFERENCE = Path(
    os.environ.get(
        "WMP_UNITREE_REFERENCE_ROOT",
        ROOT / ".work/go2_unitree_lab_20261031/unitree_rl_lab",
    )
)
OFFICIAL = REFERENCE / "source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree_actuators.py"


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONTROL = load_file("wmp_control_test", PACKAGE / "control.py")


def definition(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
    code = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    return namespace[name]


def installed_pd_method():
    paths = [
        Path(sys.prefix)
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/isaaclab"
        / "source/isaaclab/isaaclab/actuators/actuator_pd.py",
        Path(os.environ.get("ISAACLAB_ROOT", "/unconfigured")) / "source/isaaclab/isaaclab/actuators/actuator_pd.py",
    ]
    path = next((path for path in paths if path.exists()), None)
    if path is None:
        pytest.skip("installed IsaacLab IdealPD source is needed for the actuator-chain comparison")
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "IdealPDActuator")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "compute")
    scope = {}
    code = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            method,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), scope)
    return scope["compute"]


def installed_parameter_method():
    path = (
        Path(sys.prefix)
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/isaaclab"
        / "source/isaaclab/isaaclab/actuators/actuator_base.py"
    )
    if not path.exists():
        pytest.skip("installed IsaacLab parameter parser is needed for joint-name mapping")
    resolver_path = path.parents[1] / "utils/string.py"
    resolver = definition(resolver_path, "resolve_matching_names_values", dict(re=re))
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ActuatorBase")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_parse_joint_parameter"
    )
    scope = dict(torch=torch, string_utils=SimpleNamespace(resolve_matching_names_values=resolver))
    code = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), scope)
    return scope["_parse_joint_parameter"]


def actuator_classes():
    if not OFFICIAL.exists():
        pytest.skip("pinned Unitree RL Lab checkout is needed; set WMP_UNITREE_REFERENCE_ROOT")

    class PDBase:
        def __init__(self, cfg=None):
            self.cfg = cfg
            self._num_envs, self.num_joints, self._device = 2, 12, "cpu"
            canonical = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR")
                         for joint in ("hip", "thigh", "calf")]
            self.joint_names = list(getattr(cfg, "joint_names", canonical))
            shape = (self._num_envs, self.num_joints)
            self.stiffness = torch.full(shape, 25.0, dtype=torch.float64)
            self.damping = torch.full(shape, 0.5, dtype=torch.float64)
            self.effort_limit = torch.full(shape, 10.0, dtype=torch.float64)
            self.computed_effort = torch.zeros(shape, dtype=torch.float64)
            self.applied_effort = self.computed_effort.clone()

        def _parse_joint_parameter(self, value, fallback):
            if isinstance(value, dict):
                return installed_parameter_method()(self, value, fallback)
            return torch.full_like(self.computed_effort, fallback if value is None else value)

        def _clip_effort(self, effort):
            return effort.clamp(-self.effort_limit, self.effort_limit)

    # Exercise the actual framework PD implementation instead of rewriting it.
    PDBase.compute = installed_pd_method()
    ours = definition(
        PACKAGE / "tasks/direct/actuators.py",
        "UnitreeGo2Actuator",
        dict(torch=torch, IdealPDActuator=PDBase, unitree_go2_clip_effort=CONTROL.unitree_go2_clip_effort),
    )
    official = definition(OFFICIAL, "UnitreeActuator", dict(torch=torch, DelayedPDActuator=PDBase))
    previous = definition(
        PACKAGE / "tasks/direct/actuators.py",
        "UpstreamPDActuator",
        dict(torch=torch, IdealPDActuator=PDBase),
    )
    return ours, official, previous


def official_actuator():
    _, cls, _ = actuator_classes()
    cfg = SimpleNamespace(X1=13.5, X2=30.0, Y1=20.2, Y2=23.4, Fs=0.0, Fd=0.0, Va=0.01)
    return cls(cfg)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_numpy_and_torch_match_official_clip_including_boundaries(dtype):
    official = official_actuator()
    speed = torch.tensor(
        [-60, -30.0001, -30, -29.9999, -20, -13.5001, -13.5, -13.4999, -1, -0.0,
         0, 1, 13.4999, 13.5, 13.5001, 20, 29.9999, 30, 30.0001, 60],
        dtype=dtype,
    )
    effort = torch.tensor([-100.0, -23.4, -20.2, -1.0, 0.0, 1.0, 20.2, 23.4, 100.0], dtype=dtype)
    velocity, requested = torch.meshgrid(speed, effort, indexing="ij")
    official._joint_vel = velocity
    for name, value in [("_effort_y1", 20.2), ("_effort_y2", 23.4),
                        ("_velocity_x1", 13.5), ("_velocity_x2", 30.0)]:
        setattr(official, name, torch.full_like(requested, value))
    expected = official._clip_effort(requested)
    torch.testing.assert_close(CONTROL.unitree_go2_clip_effort(requested, velocity), expected, rtol=0, atol=0)
    numpy_result = CONTROL.unitree_go2_clip_effort(requested.numpy(), velocity.numpy())
    np.testing.assert_allclose(numpy_result, expected.numpy(), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("scale", (0.8, 1.0, 1.2))
def test_full_pd_actuator_and_numpy_deployment_match_official_then_scale(scale):
    cls, _, _ = actuator_classes()
    actual, reference = cls(), official_actuator()
    position = torch.linspace(-1, 1, 24, dtype=torch.float64).reshape(2, 12)
    velocity = torch.tensor(
        [-35, -30, -20, -13.5, -1, 0, 0, 1, 13.5, 20, 30, 35] * 2,
        dtype=torch.float64,
    ).reshape(2, 12)
    requested = torch.linspace(-80, 80, 24, dtype=torch.float64).reshape(2, 12)
    target = position + (requested + 0.5 * velocity) / 25
    actual.effort_scale.fill_(scale)

    def action():
        return SimpleNamespace(
            joint_positions=target.clone(),
            joint_velocities=torch.zeros_like(velocity),
            joint_efforts=torch.zeros_like(velocity),
        )

    result = actual.compute(action(), position, velocity)
    expected = reference.compute(action(), position, velocity).joint_efforts * scale
    torch.testing.assert_close(result.joint_efforts, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual.applied_effort, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        CONTROL.unitree_go2_torque(target, position, velocity, effort_scale=scale),
        expected, rtol=0, atol=0,
    )
    np.testing.assert_allclose(
        CONTROL.unitree_go2_torque(target.numpy(), position.numpy(), velocity.numpy(), effort_scale=scale),
        expected.numpy(), rtol=1e-12, atol=1e-12,
    )
    assert result.joint_positions is None and result.joint_velocities is None


def test_moving_against_torque_has_distinct_envelope_and_no_extra_calf_ratio():
    velocity = np.ones((2, 12))
    requested = np.stack((np.full(12, 100.0), np.full(12, -100.0)))
    result = CONTROL.unitree_go2_clip_effort(requested, velocity)
    np.testing.assert_allclose(result[0], 20.2)
    np.testing.assert_allclose(result[1], -23.4)
    np.testing.assert_allclose(CONTROL.unitree_go2_clip_effort(requested, velocity * 30), 0.0)
    np.testing.assert_allclose(CONTROL.unitree_go2_clip_effort(requested, velocity * 60), 0.0)


def test_previous_fixed_limit_actuator_keeps_its_motor_strength_order():
    _, _, previous = actuator_classes()
    actuator = previous()
    actuator.effort_scale.fill_(1.2)
    position = torch.zeros(2, 12, dtype=torch.float64)
    velocity = torch.zeros_like(position)
    action = SimpleNamespace(
        joint_positions=torch.full_like(position, 20.0),
        joint_velocities=velocity,
        joint_efforts=torch.zeros_like(position),
    )
    result = actuator.compute(action, position, velocity)
    torch.testing.assert_close(result.joint_efforts, torch.full_like(position, 12.0), rtol=0, atol=0)


def test_torch_helper_preserves_device_dtype_and_broadcasts_joint_parameters():
    target = torch.full((2, 12), 2.0, dtype=torch.float32)
    velocity = torch.zeros_like(target)
    result = CONTROL.unitree_go2_torque(
        target, torch.zeros_like(target), velocity,
        stiffness=torch.full((12,), 25.0), damping=0.5, effort_scale=torch.tensor([[0.8], [1.2]]),
    )
    assert isinstance(result, torch.Tensor) and result.dtype == target.dtype and result.device == target.device
    torch.testing.assert_close(result[0], torch.full((12,), 23.4 * 0.8))
    torch.testing.assert_close(result[1], torch.full((12,), 23.4 * 1.2))


GEAR = 45.43 / 23.7


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_geared_curve_matches_official_in_motor_coordinates(dtype):
    official = official_actuator()
    speed = torch.tensor(
        [-40.0, -30 / GEAR, -13.5 / GEAR, -10, -5, 0, 5, 10, 13.5 / GEAR, 30 / GEAR, 40],
        dtype=dtype,
    )
    requested = torch.tensor([-100.0, -1, 0, 1, 100.0], dtype=dtype)
    velocity, effort = torch.meshgrid(speed, requested, indexing="ij")
    ratio = torch.full_like(effort, GEAR)
    motor_effort, motor_speed = effort / ratio, velocity * ratio
    official._joint_vel = motor_speed
    for name, value in [("_effort_y1", 20.2), ("_effort_y2", 23.4),
                        ("_velocity_x1", 13.5), ("_velocity_x2", 30.0)]:
        setattr(official, name, torch.full_like(effort, value))
    expected = official._clip_effort(motor_effort) * ratio
    actual = CONTROL.unitree_go2_clip_effort(effort, velocity, gear_ratio=ratio)
    # Clipping in joint coordinates keeps an unsaturated torque unchanged;
    # dividing it by g and multiplying back adds a float64 rounding step.
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    numpy = CONTROL.unitree_go2_clip_effort(effort.numpy(), velocity.numpy(), gear_ratio=ratio.numpy())
    np.testing.assert_allclose(numpy, expected.numpy(), rtol=3e-6, atol=3e-6)
    # An extra gear increases low-speed torque but moves the no-load cutoff down.
    np.testing.assert_allclose(
        CONTROL.unitree_go2_clip_effort(100.0, 30 / GEAR, gear_ratio=GEAR), 0.0, atol=1e-12
    )


def test_transmission_conserves_mechanical_power():
    motor = official_actuator()
    velocity = torch.tensor([-15, -10, -5, 0, 5, 10, 15], dtype=torch.float64)
    effort = torch.full_like(velocity, 100.0)
    motor._joint_vel = velocity * GEAR
    for name, value in [("_effort_y1", 20.2), ("_effort_y2", 23.4),
                        ("_velocity_x1", 13.5), ("_velocity_x2", 30.0)]:
        setattr(motor, name, torch.full_like(velocity, value))
    motor_torque = motor._clip_effort(effort / GEAR)
    joint_torque = CONTROL.unitree_go2_clip_effort(effort, velocity, gear_ratio=GEAR)
    torch.testing.assert_close(joint_torque * velocity, motor_torque * motor._joint_vel, rtol=1e-14, atol=1e-14)
    # The peak-power point itself also transforms in both torque and speed.
    motor_peak_power = 20.2 * 13.5
    joint_peak_power = (20.2 * GEAR) * (13.5 / GEAR)
    assert joint_peak_power == pytest.approx(motor_peak_power, abs=1e-12)


def test_unit_ratio_matches_v1_bitwise_for_all_helpers():
    rng = np.random.default_rng(17)
    target, position = rng.normal(size=(2, 12)), rng.normal(size=(2, 12))
    velocity = rng.uniform(-40, 40, size=(2, 12))
    effort = 25 * (target - position) - 0.5 * velocity
    for args, function in [
        ((effort, velocity), CONTROL.unitree_go2_torque_limit),
        ((effort, velocity), CONTROL.unitree_go2_clip_effort),
        ((target, position, velocity), CONTROL.unitree_go2_torque),
    ]:
        reference = function(*args)
        np.testing.assert_array_equal(function(*args, gear_ratio=1.0), reference)
        np.testing.assert_array_equal(function(*args, gear_ratio=np.ones(12)), reference)
        tensor_args = tuple(torch.from_numpy(arg) for arg in args)
        tensor_reference = function(*tensor_args)
        assert torch.equal(function(*tensor_args, gear_ratio=1.0), tensor_reference)
        assert torch.equal(function(*tensor_args, gear_ratio=torch.ones(12, dtype=torch.float64)), tensor_reference)


def test_actual_joint_name_parser_maps_gears_without_relying_on_joint_order():
    cls, _, _ = actuator_classes()
    canonical = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR")
                 for joint in ("hip", "thigh", "calf")]
    names = canonical[::2] + canonical[1::2]
    cfg = SimpleNamespace(
        joint_names=names,
        gear_ratio={".*_calf_joint": GEAR, ".*_hip_joint": 1.0, ".*_thigh_joint": 1.0},
    )
    actuator = cls(cfg)
    expected = torch.tensor([GEAR if "calf" in name else 1.0 for name in names], dtype=torch.float32)
    torch.testing.assert_close(actuator._gear_ratio, expected.expand(2, -1), rtol=0, atol=0)
    assert not actuator._unit_gear
    assert cls()._unit_gear


@pytest.mark.parametrize("ratio", (0.0, -1.0, float("nan"), float("inf")))
def test_invalid_gear_rejected_at_actuator_initialization(ratio):
    cls, _, _ = actuator_classes()
    with pytest.raises(ValueError, match="finite and positive"):
        cls(SimpleNamespace(gear_ratio=ratio))


def test_incomplete_joint_gear_map_rejected_at_actuator_initialization():
    cls, _, _ = actuator_classes()
    with pytest.raises(ValueError, match="finite and positive"):
        cls(SimpleNamespace(gear_ratio={".*_calf_joint": GEAR}))


@pytest.mark.parametrize("scale", (0.8, 1.0, 1.2))
def test_geared_pd_chain_and_deployment_match_motor_curve_before_strength(scale):
    cls, _, _ = actuator_classes()
    cfg = SimpleNamespace(gear_ratio={".*_calf_joint": GEAR, ".*_hip_joint": 1.0, ".*_thigh_joint": 1.0})
    actuator = cls(cfg)
    actuator.effort_scale.fill_(scale)
    position = torch.linspace(-1, 1, 24, dtype=torch.float64).reshape(2, 12)
    velocity = torch.tensor([-15, -10, -5, 0, 5, 10, 15, 10, 5, 0, -5, -10],
                            dtype=torch.float64).expand(2, -1)
    requested = torch.linspace(-100, 100, 24, dtype=torch.float64).reshape(2, 12)
    target = position + (requested + 0.5 * velocity) / 25
    action = SimpleNamespace(
        joint_positions=target.clone(), joint_velocities=torch.zeros_like(position),
        joint_efforts=torch.zeros_like(position),
    )
    actual = actuator.compute(action, position, velocity).joint_efforts
    ratio = actuator._gear_ratio.to(dtype=torch.float64)
    motor = official_actuator()
    motor._joint_vel = velocity * ratio
    expected = motor._clip_effort(requested / ratio) * ratio * scale
    # PD reconstruction introduces only float64 rounding before motor clipping.
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(
        CONTROL.unitree_go2_torque(target, position, velocity, gear_ratio=ratio, effort_scale=scale),
        actual, rtol=0, atol=0,
    )
    numpy = CONTROL.unitree_go2_torque(
        target.numpy(), position.numpy(), velocity.numpy(), gear_ratio=ratio.numpy(), effort_scale=scale
    )
    np.testing.assert_allclose(numpy, actual.numpy(), rtol=1e-12, atol=1e-12)
