# SPDX-License-Identifier: Apache-2.0
"""以实际参考源码核对配置、地形、观测、资产和推理时序。"""

import ast
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from wmp_lab.learning.config import PAPER_CFG, SMOKE_CFG
from wmp_lab.learning.deployment import DeploymentPolicy
from wmp_lab.learning.runner import WMPRunner
from wmp_lab.reference import meshes, primitives
from wmp_lab.reference.assets import prepare_reference_urdf
from wmp_lab.reference.heights import feet_at_edge, height_points, rotate_height_points, sample_heights
from wmp_lab.reference.terrain import Terrain, convert_heightfield_to_trimesh

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "wmp_go1"


def definitions(path, namespace, names=None):
    nodes = [
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef | ast.FunctionDef) and (names is None or n.name in names)
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_effective_reference_training_configuration_and_soft_std():
    scope = definitions(
        UPSTREAM / "legged_gym/envs/base/legged_robot_config.py", {"BaseConfig": type("BaseConfig", (), {})}
    )
    scope["MOTION_FILES"] = []
    definitions(UPSTREAM / "legged_gym/envs/a1/a1_amp_config.py", scope)
    reference = scope["A1AMPCfgPPO"]
    assert PAPER_CFG.gamma == reference.algorithm.gamma == 0.99
    assert scope["A1AMPCfg"].env.num_envs == 4100
    assert PAPER_CFG.num_envs == 4096  # 以 gpufree 已成功的运行覆盖源码默认环境数。
    assert PAPER_CFG.robot == "go1"
    soft = torch.tensor([[[-0.9, 0.9]]] * 12).reshape(1, 12, 2)

    class Env:
        num_envs = 1
        camera_mask = torch.ones(1, dtype=torch.bool)
        _joint_ids = list(range(12))
        _robot = SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=soft, joint_pos_limits=soft / 0.9))

        def reset(self):
            return {"policy": torch.zeros(1, 285)}, {}

    runner = WMPRunner(SMOKE_CFG)
    runner._initialize_rollout(Env())
    torch.testing.assert_close(runner.ppo.min_std, 1.8 * torch.tensor((0.05, 0.02, 0.05) * 4))


def test_reference_height_order_three_cell_query_and_small_edges():
    points = height_points()
    torch.testing.assert_close(points[:11, 0], torch.full((11,), -0.8))
    torch.testing.assert_close(points[:11, 1], torch.tensor([i / 10 for i in range(-5, 6)]))
    forward = height_points(True)
    assert forward.shape == (525, 3)
    torch.testing.assert_close(forward[:25, 0], torch.zeros(25))
    raw = np.zeros((17, 11), dtype=np.int16)
    raw[9:] = 30
    _, _, edge = convert_heightfield_to_trimesh(raw, 0.1, 0.005, 0.75)
    assert edge.any()
    hits = sample_heights(torch.tensor(raw), torch.tensor([[[0.85, 0.05, 0.0], [0.95, 0.05, 0.0]]]), 0, 0.1, 0.005)
    torch.testing.assert_close(hits, torch.tensor([[0.0, 0.15]]))
    at_edge = feet_at_edge(
        torch.tensor(edge), torch.tensor([[[0.8, 0.1, 0.0]]]), torch.ones(1, 1, dtype=torch.bool), 0, 0.1
    )
    assert at_edge.item()


def _reference_primitives():
    path = Path("/home/re/workspace/isaacgym/python/isaacgym/terrain_utils.py")
    if not path.exists():
        pytest.skip("original Isaac Gym terrain source is not installed")

    def interp2d(y, x, values, kind):
        spline = RectBivariateSpline(x, y, values, kx=1, ky=1)
        return lambda yy, xx: spline(xx, yy)

    scope = dict(np=np, choice=np.random.choice, interpolate=SimpleNamespace(interp2d=interp2d))
    definitions(path, scope)
    return SimpleNamespace(**{key: value for key, value in scope.items() if not key.startswith("__")})


@pytest.mark.parametrize(
    "kind",
    [
        "random_uniform_terrain",
        "pyramid_sloped_terrain",
        "pyramid_stairs_terrain",
        "discrete_obstacles_terrain",
        "wave_terrain",
        "stepping_stones_terrain",
    ],
)
def test_base_terrains_match_installed_original_functions(kind):
    reference = _reference_primitives()
    arguments = {
        "random_uniform_terrain": dict(min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2),
        "pyramid_sloped_terrain": dict(slope=0.35, platform_size=3),
        "pyramid_stairs_terrain": dict(step_width=0.32, step_height=-0.17, platform_size=3),
        "discrete_obstacles_terrain": dict(max_height=0.21, min_size=1, max_size=2, num_rects=20, platform_size=3),
        "wave_terrain": dict(num_waves=5, amplitude=0.22),
        "stepping_stones_terrain": dict(stone_size=0.3, stone_distance=0.1, max_height=0.12, platform_size=2),
    }[kind]
    first = primitives.SubTerrain(width=80, length=80, horizontal_scale=0.1, vertical_scale=0.005)
    second = primitives.SubTerrain(width=80, length=80, horizontal_scale=0.1, vertical_scale=0.005)
    np.random.seed(7)
    getattr(primitives, kind)(first, **arguments)
    np.random.seed(7)
    getattr(reference, kind)(second, **arguments)
    np.testing.assert_array_equal(first.height_field_raw, second.height_field_raw)


def test_complete_reference_map_matches_source_geometry_and_spawn():
    scope = definitions(
        UPSTREAM / "legged_gym/utils/terrain.py",
        dict(
            np=np,
            random=random,
            terrain_utils=_reference_primitives(),
            trimesh=meshes,
            binary_dilation=binary_dilation,
            LeggedRobotCfg=SimpleNamespace(terrain=object),
        ),
    )
    # 一个非零列覆盖所有原17类，比较相同随机状态下的整张高度图、附加网格与出生点。
    cfg = SimpleNamespace(
        mesh_type="trimesh",
        terrain_length=8,
        terrain_width=8,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_size=1,
        num_rows=2,
        num_cols=17,
        terrain_proportions=[1 / 17] * 17,
        slope_threshold=0.75,
        curriculum=True,
        selected=False,
    )
    random.seed(7)
    np.random.seed(7)
    first = Terrain(SimpleNamespace(**cfg.__dict__), 17)
    random.seed(7)
    np.random.seed(7)
    second = scope["Terrain"](SimpleNamespace(**cfg.__dict__), 17)
    for name in ("height_field_raw", "env_origins", "vertices", "triangles", "x_edge_mask"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_asset_preparation_removes_empty_bodies_and_preserves_mass(tmp_path):
    source = ROOT / "source/wmp_lab/wmp_lab/assets/robots/go1/urdf/go1_isaaclab.urdf"
    prepared = prepare_reference_urdf(source, tmp_path / "robot.urdf")
    original, actual = ET.parse(source).getroot(), ET.parse(prepared).getroot()

    def mass(root):
        return sum(float(node.get("value")) for node in root.findall("link/inertial/mass"))

    assert np.isclose(mass(original), mass(actual))
    assert not any("shoulder" in link.get("name") for link in actual.findall("link"))
    assert len([j for j in actual.findall("joint") if j.get("type") == "revolute"]) == 12
    assert {j.get("name") for j in actual.findall("joint") if j.get("type") == "fixed"} == {
        f"{leg}_foot_fixed" for leg in ("FL", "FR", "RL", "RR")
    }
    base = actual.find("link[@name='base']")
    assert float(base.find("inertial/mass").get("value")) == pytest.approx(5.301)


def test_reference_export_update_phase_and_sampling_reset():
    runner = WMPRunner(SMOKE_CFG)
    policy = DeploymentPolicy(runner, sample_rssm=False)
    observation, depth = torch.randn(1, 45), torch.rand(1, 64, 64, 1) - 0.5
    for _ in range(4):
        policy(observation, depth)
        assert not policy.world_feature.any()
    policy(observation, depth)
    assert policy.world_feature.any()
    feature = policy.world_feature.clone()
    policy.reset_envs(torch.tensor([True]))
    torch.testing.assert_close(policy.world_feature, feature)
    policy.reset()
    assert not policy.world_feature.any()


def _method(path, name, **scope):
    function = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def test_environment_observation_fields_match_original_compute_observations():
    n = 2
    generator = torch.Generator().manual_seed(23)

    def rand(width):
        return torch.rand(n, width, generator=generator)

    forces = rand(39).reshape(n, 13, 3)
    base = SimpleNamespace(
        root_lin_vel_b=rand(3),
        root_ang_vel_b=rand(3),
        projected_gravity_b=rand(3),
        joint_pos=rand(12),
        default_joint_pos=rand(12),
        joint_vel=rand(12),
    )
    ours = SimpleNamespace(
        num_envs=n,
        device="cpu",
        reference_mode=True,
        cfg=SimpleNamespace(enable_observation_noise=False),
        _robot=SimpleNamespace(data=base),
        _joint_ids=list(range(12)),
        _feet_ids=list(range(1, 5)),
        _undesired_ids=list(range(5, 13)),
        _friction=rand(1),
        _restitution=rand(1),
        _added_mass=rand(1),
        _com_offset=rand(3),
        _p_gain_scale=rand(12),
        _d_gain_scale=rand(12),
        _commands=rand(3),
        _actions=rand(12),
        _contact_sensor=SimpleNamespace(
            data=SimpleNamespace(net_forces_w=forces, net_forces_w_history=forces[:, None].expand(-1, 3, -1, -1))
        ),
        _body_height_scanner=None,
        _height=lambda scanner, dim: torch.zeros(n, dim),
    )
    method = _method(
        ROOT / "source/wmp_lab/wmp_lab/tasks/direct/wmp_env.py",
        "_get_observations",
        torch=torch,
        HEIGHT_DIM=187,
        OBSERVATION_DIM=285,
    )
    actual = method(ours)["policy"]
    scale = SimpleNamespace(
        lin_vel=1,
        ang_vel=0.25,
        dof_pos=1,
        dof_vel=0.05,
        height_measurements=5,
        contact_force=0.005,
        com_pos=20,
        pd_gains=5,
    )
    reference = SimpleNamespace(
        base_lin_vel=base.root_lin_vel_b,
        base_ang_vel=base.root_ang_vel_b,
        projected_gravity=base.projected_gravity_b,
        commands=ours._commands,
        commands_scale=torch.tensor((1, 1, 0.25)),
        dof_pos=base.joint_pos,
        default_dof_pos=base.default_joint_pos,
        dof_vel=base.joint_vel,
        actions=ours._actions,
        obs_scales=scale,
        root_states=torch.zeros(n, 13),
        measured_heights=torch.zeros(n, 187),
        randomized_frictions=ours._friction,
        randomized_restitutions=ours._restitution,
        randomized_added_masses=ours._added_mass,
        randomized_com_pos=ours._com_offset,
        randomized_p_gains=(ours._p_gain_scale + 1) * 40,
        p_gains=torch.ones(12) * 40,
        randomized_d_gains=ours._d_gain_scale + 1,
        d_gains=torch.ones(12),
        sensor_forces=forces[:, ours._feet_ids],
        contact_forces=forces,
        penalised_contact_indices=ours._undesired_ids,
        cfg=SimpleNamespace(
            env=SimpleNamespace(privileged_obs=True),
            terrain=SimpleNamespace(measure_heights=True),
            normalization=SimpleNamespace(base_height=0.3),
            domain_rand=SimpleNamespace(
                randomize_friction=True,
                randomize_restitution=True,
                randomize_base_mass=True,
                randomize_com_pos=True,
                randomize_gains=True,
            ),
        ),
        add_noise=False,
        num_obs=285,
        num_privileged_obs=285,
    )
    reference.root_states[:, 2] = 0.3
    _method(UPSTREAM / "legged_gym/envs/base/legged_robot.py", "compute_observations", torch=torch)(reference)
    torch.testing.assert_close(actual, reference.obs_buf, rtol=1e-6, atol=1e-6)


def test_reference_timeout_bootstrap_uses_saved_transition_value():
    from dataclasses import replace

    class Env:
        num_envs = 2
        camera_mask = torch.ones(2, dtype=torch.bool)

        def reset(self):
            return {"policy": torch.full((2, 285), 2.0)}, {}

        def step(self, action):
            return (
                {"policy": torch.zeros(2, 285)},
                torch.tensor([0.02, 0.02]),
                torch.tensor([False, False]),
                torch.tensor([True, True]),
                {"terminal_observation": torch.full((2, 285), 9.0)},
            )

        def get_amp_observations(self):
            return torch.zeros(2, 30)

        def get_forward_height_map(self):
            return torch.zeros(2, 525)

        def get_depth(self):
            return torch.zeros(2, 64, 64, 1)

    runner = WMPRunner(replace(SMOKE_CFG, world_train_start=10**9))
    runner.policy.evaluate = lambda obs, feature: obs[:, :1]
    runner.discriminator.predict_amp_reward = lambda state, next_state, reward: (reward, reward)
    captured = []

    def capture(batch, expert, **kwargs):
        captured.append(batch["return"])
        return {"test_metric": 0}

    runner.ppo.update = capture
    runner.collect_and_update(Env(), object(), rollout_steps=1)
    torch.testing.assert_close(captured[0], torch.tensor([0.02 + 0.99 * 2] * 2))


def test_height_rotation_matches_original_projected_quaternion():
    quaternion = torch.tensor([[0.7, 0.2, 0.3, 0.5]])
    points = height_points()
    quat = quaternion[:, (1, 2, 3, 0)].repeat(len(points), 1)

    def apply(q, v):
        xyz, w = q[:, :3], q[:, 3:]
        uv = torch.cross(xyz, v, dim=-1)
        return v + 2 * (w * uv + torch.cross(xyz, uv, dim=-1))

    original = _method(
        UPSTREAM / "legged_gym/utils/math.py",
        "quat_apply_yaw",
        normalize=lambda q: q / q.norm(dim=-1, keepdim=True).clamp_min(1e-9),
        quat_apply=apply,
    )
    torch.testing.assert_close(rotate_height_points(quaternion, points)[0], original(quat, points))


def test_world_updates_follow_environment_clock_and_keep_feature_until_reset_tick():
    from dataclasses import replace

    class Env:
        num_envs = 1
        camera_mask = torch.ones(1, dtype=torch.bool)

        def reset(self):
            self._global_step = 1
            return {"policy": torch.zeros(1, 285)}, {}

        def step(self, action):
            self._global_step += 1
            return (
                {"policy": torch.zeros(1, 285)},
                torch.tensor([0.02]),
                torch.tensor([self._global_step == 8]),
                torch.tensor([False]),
                {},
            )

        def get_amp_observations(self):
            return torch.zeros(1, 30)

        def get_forward_height_map(self):
            return torch.zeros(1, 525)

        def get_depth(self):
            return torch.zeros(1, 64, 64, 1)

    env = Env()
    runner = WMPRunner(replace(SMOKE_CFG, world_train_start=10**9))
    calls, features = [], {}
    observe, act = runner.world_model.observe_step, runner.policy.act

    def observed(*args, **kwargs):
        calls.append((env._global_step, args[3].clone()))
        return observe(*args, **kwargs)

    def acting(obs, history, feature):
        features[env._global_step] = feature.clone()
        return act(obs, history, feature)

    runner.world_model.observe_step = observed
    runner.policy.act = acting
    runner.ppo.update = lambda *args, **kwargs: {"test_metric": 0}
    runner.collect_and_update(env, object(), rollout_steps=11)
    assert [step for step, _ in calls] == [5, 10]
    assert all(first.item() == 1 for _, first in calls)
    assert features[7].any()
    torch.testing.assert_close(features[8], features[7])


def test_command_resampling_and_timeout_match_original_boundaries():
    from wmp_lab.robots import CONTROL_REVISION

    calls = []
    robot = SimpleNamespace(
        default_joint_pos=torch.zeros(1, 12),
        root_lin_vel_b=torch.zeros(1, 3),
        root_lin_vel_w=torch.zeros(1, 3),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
    )
    env = SimpleNamespace(
        reference_mode=True,
        num_envs=1,
        device="cpu",
        step_dt=0.02,
        max_episode_length=1000,
        episode_length_buf=torch.zeros(1, dtype=torch.long),
        _commands=torch.zeros(1, 3),
        _actions=torch.zeros(1, 12),
        _previous_actions=torch.zeros(1, 12),
        _last_last_actions=torch.zeros(1, 12),
        _robot=SimpleNamespace(data=robot),
        _joint_ids=list(range(12)),
        _global_step=0,
        _terrain=SimpleNamespace(terrain_levels=torch.zeros(1, dtype=torch.long)),
        _contact_sensor=SimpleNamespace(
            data=SimpleNamespace(net_forces_w=torch.zeros(1, 1, 3), net_forces_w_history=torch.zeros(1, 3, 1, 3))
        ),
        _base_contact_ids=[0],
        _base_id=[0],
        cfg=SimpleNamespace(
            control_revision=CONTROL_REVISION,
            enable_domain_randomization=False,
            action_scale=0.25,
            command_resampling_s=10,
            reward_semantics="upstream_v3",
        ),
        _resample_commands=lambda ids: calls.append(ids.clone()),
        _update_heading_commands=lambda: None,
    )
    path = ROOT / "source/wmp_lab/wmp_lab/tasks/direct/wmp_env.py"
    _method(path, "_pre_physics_step", torch=torch, EXPLICIT_PD_REVISIONS=(CONTROL_REVISION,))(env, torch.zeros(1, 12))
    assert not calls
    dones = _method(path, "_get_dones", torch=torch)
    env.episode_length_buf[:] = 1
    dones(env)
    assert not calls
    env.episode_length_buf[:] = 500
    dones(env)
    assert len(calls) == 1
    env.episode_length_buf[:] = 1000
    assert not dones(env)[1].item()
    env.episode_length_buf[:] = 1001
    assert dones(env)[1].item()


def test_deployment_depth_buffer_matches_delayed_environment_frames():
    from wmp_lab.deployment_depth import ReferenceDepthBuffer

    buffer = ReferenceDepthBuffer()
    inputs = {}
    for step in range(15):
        inputs[step] = float(buffer.delayed.mean())
        if buffer.capture_due(step):
            buffer.capture(np.full((64, 64), step, dtype=np.float32))
    assert inputs[4] == 0
    assert inputs[9] == 3
    assert inputs[14] == 8
    buffer.reset()
    assert not buffer.delayed.any()


def test_environment_contract_canonicalizes_callable_instances():
    from wmp_lab.learning.contracts import canonical

    class Event:
        def __call__(self):
            return None

    assert canonical(Event()) == canonical(Event())
    assert "0x" not in canonical(Event())
