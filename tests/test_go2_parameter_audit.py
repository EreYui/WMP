# SPDX-License-Identifier: Apache-2.0
"""对照有效上游配置和真实奖励函数，覆盖 Go1/Go2 共用的训练定义。"""

import ast
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from wmp_lab.learning.config import PAPER_CFG
from wmp_lab.learning.task_rewards import stuck_penalty, tracking_velocity
from wmp_lab.reference.heights import feet_at_edge
from wmp_lab.robots import CONTROL_REVISION, robot_profile

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "wmp_go1"


def reference_config():
    scope = {"BaseConfig": type("BaseConfig", (), {}), "MOTION_FILES": []}
    for relative in ("base/legged_robot_config.py", "a1/a1_amp_config.py"):
        path = UPSTREAM / "legged_gym/envs" / relative
        nodes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["A1AMPCfg"], scope["A1AMPCfgPPO"]


def method(path, class_name, name, **scope):
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(torch=torch, **scope)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_effective_ppo_amp_world_and_pd_parameters_for_both_robots():
    env, train = reference_config()
    world = yaml.safe_load((UPSTREAM / "dreamer/configs.yaml").read_text())["defaults"]
    pairs = {
        "num_envs": 4096,  # gpufree 成功运行的显式覆盖值；参考源码默认仍为 4100。
        "rollout_steps": train.runner.num_steps_per_env,
        "learning_epochs": train.algorithm.num_learning_epochs,
        "mini_batches": train.algorithm.num_mini_batches,
        "actor_hidden": tuple(train.policy.actor_hidden_dims),
        "critic_hidden": tuple(train.policy.critic_hidden_dims),
        "history_hidden": tuple(train.policy.encoder_hidden_dims),
        "amp_hidden": tuple(train.runner.amp_discr_hidden_dims),
        "replay_size": train.algorithm.amp_replay_buffer_size,
        "lr": train.algorithm.learning_rate,
        "gamma": train.algorithm.gamma,
        "gae_lambda": train.algorithm.lam,
        "clip_param": train.algorithm.clip_param,
        "entropy_coef": train.algorithm.entropy_coef,
        "amp_reward_coef": train.runner.amp_reward_coef,
        "amp_task_reward_lerp": train.runner.amp_task_reward_lerp,
        "world_hidden": world["dyn_hidden"],
        "world_feature_dim": world["dyn_deter"],
        "world_stochastic_dim": world["dyn_stoch"],
        "world_categories": world["dyn_discrete"],
        "world_update_interval": env.depth.update_interval,
        "world_train_steps": world["train_steps_per_iter"],
        "world_train_start": world["train_start_steps"],
        "world_batch_size": world["batch_size"],
        "world_sequence_length": world["batch_length"],
        "world_lr": float(world["model_lr"]),
        "reward_loss_scale": world["reward_head"]["loss_scale"],
        "depth_lr": train.depth_predictor.lr,
        "depth_train_steps": train.depth_predictor.training_iters,
        "depth_batch_size": train.depth_predictor.batch_size,
    }
    for robot in ("go1", "go2"):
        cfg = replace(PAPER_CFG, robot=robot)
        for name, expected in pairs.items():
            assert getattr(cfg, name) == expected, (robot, name)
        profile = robot_profile(robot, control_revision=CONTROL_REVISION)
        assert profile.stiffness == env.control.stiffness["joint"]
        assert profile.damping == env.control.damping["joint"]
        assert profile.action_scale == env.control.action_scale
        assert list(profile.default_joint_pos) == [env.init_state.default_joint_angles[j] for j in profile.joint_order]
    go1, go2 = asdict(replace(PAPER_CFG, robot="go1")), asdict(replace(PAPER_CFG, robot="go2"))
    assert {key for key in go1 if go1[key] != go2[key]} == {"robot"}
    tree = ast.parse((ROOT / "scripts/train.py").read_text())
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant):
                for keyword in node.keywords:
                    if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                        defaults[node.args[0].value] = keyword.value.value
    assert defaults["--iterations"] == train.runner.max_iterations
    assert defaults["--save-interval"] == train.runner.save_interval


@pytest.mark.parametrize("robot", ("go1", "go2"))
def test_every_nonzero_task_reward_matches_original_functions(robot):
    torch.manual_seed(7)
    n, joints, bodies = 18, 12, 13
    reference_cfg, _ = reference_config()
    profile = robot_profile(robot)
    dof_pos = torch.randn(n, joints) * 0.8
    dof_vel = torch.randn(n, joints) * 20
    old_vel = dof_vel + torch.randn(n, joints) * 0.05
    dof_limits = torch.tensor([-0.9, 0.9]).expand(joints, 2).clone()
    velocity_limits = torch.tensor(profile.velocity_limits)
    force = torch.randn(n, bodies, 3) * 2
    feet, undesired = [1, 2, 3, 4], list(range(5, 13))
    lin_vel, ang_vel = torch.randn(n, 3), torch.randn(n, 3)
    commands = torch.randn(n, 3)
    commands[::3] = 0
    actions, previous, older = (torch.randn(n, joints) * 0.1 for _ in range(3))
    torque = torch.randn(n, joints)
    heading = torch.linspace(-1.5, 1.5, n)
    root = torch.zeros(n, 13)
    root[:, 2] = torch.linspace(0.2, 0.6, n)
    feet_positions = torch.rand(n, bodies, 3) * 0.8
    edge_mask = torch.rand(24, 24) > 0.5
    previous_contacts = torch.rand(n, 4) > 0.5
    filtered = (force[:, feet, 2] > 1) | previous_contacts
    air = torch.rand(n, 4)
    origins = torch.zeros(n, 3)
    names = (
        "wave",
        "rough_slope",
        "stairs_up",
        "stairs_down",
        "discrete",
        "gap",
        "continuous_gap",
        "bream",
        "stairs_cliff",
        "slope_cliff",
        "stepping_stones",
        "stepping_one_bridge",
        "climb",
        "tilt",
        "stool",
        "crawl",
        "rough_flat",
        "plum_piles",
    )
    masks = {"wmp_" + name: torch.arange(n) == index for index, name in enumerate(names)}
    original = SimpleNamespace(
        num_envs=n,
        device="cpu",
        cfg=reference_cfg,
        dt=0.02,
        torques=torque,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        last_dof_vel=old_vel,
        dof_pos_limits=dof_limits,
        dof_vel_limits=velocity_limits,
        actions=actions,
        last_actions=previous,
        last_last_actions=older,
        base_lin_vel=lin_vel,
        base_ang_vel=ang_vel,
        commands=commands,
        contact_forces=force,
        feet_indices=feet,
        penalised_contact_indices=undesired,
        feet_air_time=air.clone(),
        contact_filt=filtered,
        root_states=root,
        env_origins=origins,
        x_edge_mask=edge_mask,
        terrain=SimpleNamespace(cfg=SimpleNamespace(border_size=0, horizontal_scale=0.1)),
        rigid_body_states=torch.cat((feet_positions, torch.zeros(n, bodies, 10)), -1),
        base_quat=torch.stack((torch.zeros(n), torch.zeros(n), (heading / 2).sin(), (heading / 2).cos()), -1),
        forward_vec=torch.tensor([1.0, 0, 0]).expand(n, 3),
    )
    for name in ("gap", "continuous_gap", "bream", "climb", "stool", "crawl"):
        index = names.index(name)
        setattr(original, name + "_start_idx", index)
        setattr(original, name + "_end_idx", index + 1)
    data = SimpleNamespace(
        root_lin_vel_b=lin_vel,
        root_ang_vel_b=ang_vel,
        applied_torque=torque,
        joint_acc=torch.zeros(n, joints),
        joint_vel=dof_vel,
        joint_pos=dof_pos,
        soft_joint_pos_limits=dof_limits.expand(n, -1, -1),
        joint_vel_limits=velocity_limits.expand(n, -1),
        root_lin_vel_w=lin_vel,
        root_pos_w=root[:, :3],
        heading_w=heading,
        body_pos_w=feet_positions,
    )
    sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=force, last_air_time=torch.zeros(n, bodies)),
        compute_first_contact=lambda dt: torch.zeros(n, bodies, dtype=torch.bool),
    )
    ours = SimpleNamespace(
        num_envs=n,
        device="cpu",
        cfg=SimpleNamespace(
            reward_semantics="upstream_v3",
            stuck_penalty_scale=1,
            terrain=SimpleNamespace(terrain_generator=SimpleNamespace(horizontal_scale=0.1)),
        ),
        step_dt=0.02,
        _commands=commands,
        _robot=SimpleNamespace(data=data),
        _joint_ids=list(range(joints)),
        _previous_joint_velocity=old_vel.clone(),
        _actions=actions,
        _previous_actions=previous,
        _last_last_actions=older,
        _contact_sensor=sensor,
        _undesired_ids=undesired,
        _feet_ids=feet,
        _feet_body_ids=feet,
        _feet_air_time=air.clone(),
        _last_foot_contacts=previous_contacts,
        _terrain_type_masks=masks,
        _directional_terrain_mask=(torch.arange(n) >= 5) & (torch.arange(n) <= 15),
        _tracking_episode_sum=torch.zeros(n),
        _reward_curriculum_scale=0.46,
        reference_mode=True,
        _terrain=SimpleNamespace(env_origins=origins, x_edge_mask=edge_mask, heightfield_border=0),
        extras={},
        get_amp_observations=lambda: torch.zeros(n, 30),
        _get_observations=lambda: {"policy": torch.zeros(n, 285)},
    )
    reward_fn = method(
        ROOT / "source/wmp_lab/wmp_lab/tasks/direct/wmp_env.py",
        "WMPEnv",
        "_get_rewards",
        tracking_velocity=tracking_velocity,
        stuck_penalty=stuck_penalty,
        feet_at_edge=feet_at_edge,
    )
    result = reward_fn(ours)
    terms = dict(
        tracking="tracking_lin_vel",
        yaw_tracking="tracking_ang_vel",
        torque="torques",
        acceleration="dof_acc",
        action_rate="action_rate",
        smoothness="smoothness",
        collision="collision",
        joint_pos_limits="dof_pos_limits",
        joint_vel_limits="dof_vel_limits",
        air_time="feet_air_time",
        vertical="lin_vel_z",
        stumble="feet_stumble",
        edge="feet_edge",
        stuck="stuck",
        cheat="cheat",
        stool_height="stool_base_height",
    )

    def quat_apply_xyzw(q, value):
        xyz = q[:, :3]
        t = xyz.cross(value, dim=-1) * 2
        return value + q[:, 3:] * t + xyz.cross(t, dim=-1)

    expected = {}
    for name, source in terms.items():
        function = method(
            UPSTREAM / "legged_gym/envs/base/legged_robot.py",
            "LeggedRobot",
            "_reward_" + source,
            quat_apply=quat_apply_xyzw,
        )
        value = function(original) * getattr(reference_cfg.rewards.scales, source)
        expected[name] = value * (ours._reward_curriculum_scale if name == "edge" else 1)
        torch.testing.assert_close(ours._reward_terms[name], expected[name], rtol=2e-6, atol=2e-7, msg=name)
    torch.testing.assert_close(result, (sum(expected.values()) * 0.02).clamp_min(0), rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(ours._feet_air_time, original.feet_air_time)


def test_vendored_camera_core_retains_fixed_reference_source():
    relative = "legged_gym/sensors/warp/"
    packaged = ROOT / "source/wmp_lab/wmp_lab/sensors/upstream_wmp"
    assert (packaged / "warp_camera_kernels.py").read_bytes() == (
        UPSTREAM / relative / "warp_camera_kernels.py"
    ).read_bytes()
    original = (UPSTREAM / relative / "warp_cam.py").read_text()
    adapted = (packaged / "warp_cam.py").read_text()
    assert adapted == original.replace(
        "from legged_gym.sensors.warp.warp_camera_kernels import  DepthCameraWarpKernels",
        "from .warp_camera_kernels import DepthCameraWarpKernels",
    )
