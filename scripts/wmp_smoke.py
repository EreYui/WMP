#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""真实 CUDA smoke；WMP_ROBOT=go1/go2 选择机器人，由共享 IsaacLab wrapper 启动。"""

import argparse
import ast
import json
import os
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import wmp_lab  # noqa: E402,F401
from wmp_lab.learning import AMPMotionDataset, WMPRunner  # noqa: E402
from wmp_lab.learning.config import PAPER_CFG, SMOKE_CFG  # noqa: E402
from wmp_lab.learning.deployment import export_policy  # noqa: E402
from wmp_lab.robots import CONTROL_REVISION, robot_profile  # noqa: E402
from wmp_lab.tasks.direct.config_audit import audit_robot_configs  # noqa: E402
from wmp_lab.tasks.direct.wmp_env_cfg import make_env_cfg, terrain_config  # noqa: E402


def verify_control(env, robot):
    from isaaclab.utils.types import ArticulationActions

    profile = robot_profile(robot)
    actuator = env._robot.actuators["legs"]
    expected_limits = dict(zip(profile.joint_order, profile.torque_limits, strict=True))
    expected_velocity = dict(zip(profile.joint_order, profile.velocity_limits, strict=True))
    limits = torch.tensor([expected_limits[name] for name in actuator.joint_names], device=env.device)
    velocities = torch.tensor([expected_velocity[name] for name in actuator.joint_names], device=env.device)
    torch.testing.assert_close(actuator.effort_limit, limits.expand_as(actuator.effort_limit))
    torch.testing.assert_close(actuator.velocity_limit_sim, velocities.expand_as(actuator.velocity_limit_sim))
    torch.testing.assert_close(
        env._robot.data.joint_vel_limits[:, env._joint_ids],
        torch.tensor(profile.velocity_limits, device=env.device).expand(env.num_envs, -1),
    )
    # 从固定原源码提取实际控制函数；速度覆盖 DC 曲线会错误降低力矩的区间。
    source = Path(__file__).resolve().parents[2] / "wmp_go1/legged_gym/envs/base/legged_robot.py"
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "LeggedRobot"
    )
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_compute_torques")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    q = torch.randn_like(actuator.effort_limit)
    dq = torch.linspace(-60, 60, q.numel(), device=env.device).reshape_as(q)
    actions = torch.rand_like(q) * 12 - 6
    fake = SimpleNamespace(
        cfg=SimpleNamespace(
            control=SimpleNamespace(control_type="P", action_scale=0.25),
            domain_rand=SimpleNamespace(randomize_gains=False),
        ),
        p_gains=actuator.stiffness,
        d_gains=actuator.damping,
        default_dof_pos=torch.zeros_like(q),
        dof_pos=q,
        dof_vel=dq,
        torque_limits=limits,
    )
    reference = scope["_compute_torques"](fake, actions)
    for scale in (0.8, 1.0, 1.2):
        actuator.effort_scale.fill_(scale)
        result = actuator.compute(
            ArticulationActions(
                joint_positions=actions * 0.25, joint_velocities=torch.zeros_like(q), joint_efforts=torch.zeros_like(q)
            ),
            q,
            dq,
        )
        torch.testing.assert_close(result.joint_efforts, reference * scale, rtol=0, atol=0)
    original_dr = env.cfg.enable_domain_randomization
    env.cfg.enable_domain_randomization = False
    for latency in (0, 1, 2):
        env._latency_substeps, env._physics_substep = latency, 0
        env._actions.fill_(1)
        env._previous_actions.fill_(-1)
        for substep in range(env.cfg.decimation):
            env._apply_action()
            expected = env._robot.data.default_joint_pos[:, env._joint_ids] + (0.25 if substep >= latency else -0.25)
            torch.testing.assert_close(env._processed_actions, expected, rtol=0, atol=0)
    env.cfg.enable_domain_randomization = original_dr
    env.reset()
    print(f"[PASS] {robot}: named limits, original PD torque and post-clip strength, latency 0/1/2", flush=True)


def main():
    robot = os.environ.get("WMP_ROBOT", "go2")
    config_audit = audit_robot_configs()
    print("[PASS] Go1/Go2: only explicit robot adaptations differ in effective configs", flush=True)
    print(f"[SMOKE] {robot}: validating all terrain generators", flush=True)
    terrain = terrain_config(robot, "all")
    for name, cfg in terrain.sub_terrains.items():
        cfg.size = terrain.size
        if hasattr(cfg, "horizontal_scale"):
            cfg.horizontal_scale, cfg.vertical_scale = terrain.horizontal_scale, terrain.vertical_scale
        for difficulty in (0.0, 1.0):
            meshes, origin = cfg.function(difficulty, cfg)
            if not meshes or not all(torch.isfinite(torch.as_tensor(mesh.vertices)).all() for mesh in meshes):
                raise RuntimeError(f"invalid geometry {name} at {difficulty}")
            if len(origin) != 3:
                raise RuntimeError(f"invalid origin {name}")
    full = os.environ.get("WMP_SMOKE_FULL") == "1"
    cfg = make_env_cfg(robot, smoke=not full, terrain="all" if full else None, num_envs=18 if full else 4)
    if full:
        cfg.terrain.terrain_generator.num_rows = 2
        cfg.terrain.terrain_generator.num_cols = 18
        cfg.logical_camera_envs = 8
        cfg.episode_length_s = 2.0
    env = gym.make(f"Isaac-WMP-{robot.capitalize()}-Rough-Direct-Smoke-v0", cfg=cfg).unwrapped
    try:
        if cfg.control_revision == CONTROL_REVISION:
            verify_control(env, robot)
        assert [env._robot.body_names[i] for i in env._feet_body_ids] == [
            f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")
        ]
        assert cfg.robot.spawn.articulation_props.enabled_self_collisions
        if robot == "go1":
            assert not any("shoulder" in name for name in env._robot.body_names)
            nominal_mass = float(env._robot.data.default_mass[0].sum())
            assert abs(nominal_mass - 11.809932) < 1e-4, nominal_mass
        print(f"[SMOKE] {robot}: reset, sensors and terminal transitions", flush=True)
        obs, _ = env.reset()
        assert obs["policy"].shape == (env.num_envs, 285)
        assert env.get_amp_observations().shape == (env.num_envs, 30)
        assert env.get_forward_height_map().shape == (env.num_envs, 525)
        assert env.camera_mask.sum() == cfg.logical_camera_envs
        assert env.depth_supervision_mask.any()
        if full:
            assert len(set(env.terrain_names)) == 18
            assert env._friction.std() > 0
            assert env._motor_strength.std() > 0
            assert env._added_mass.abs().max() > 0
        assert env._depth_camera.buffer_report()["cameras"] == int(env.camera_mask.sum())
        assert torch.isfinite(env.get_depth()).all()
        assert env.get_depth().min() >= -0.5 and env.get_depth().max() <= 0.5
        env.episode_length_buf[0] = env.max_episode_length
        observation, _, _, truncated, info = env.step(torch.zeros(env.num_envs, 12, device=env.device))
        assert truncated[0]
        assert info["terminal_amp_states"].shape == (env.num_envs, 30)
        assert info["terminal_observation"].shape == (env.num_envs, 285)
        assert torch.isfinite(observation["policy"]).all()
        dataset_name = f"motions_{robot}" if os.environ.get("WMP_MOTION_SET") == "robot" else "motions_upstream"
        motion_dir = Path(wmp_lab.__file__).parent / "data" / dataset_name
        expert = AMPMotionDataset(motion_dir, env.device, seed=7, sampling_mode="upstream_v1", preload=64)
        print(f"[AMP] {robot}: clips={expert.num_motions}, fingerprint={expert.contract['fingerprint']}", flush=True)
        learning = SMOKE_CFG
        if os.environ.get("WMP_SMOKE_PAPER") == "1":
            learning = replace(
                PAPER_CFG,
                reference_num_envs=None,
                replay_mode="ring",
                world_train_steps=1,
                depth_train_steps=1,
                depth_batch_size=4,
                world_train_start=1,
                world_replay_capacity=256,
                world_batch_size=4,
                world_sequence_length=4,
                replay_size=1024,
            )
        runner = WMPRunner(replace(learning, robot=robot, num_envs=env.num_envs), env.device)
        print(f"[SMOKE] {robot}: continuous joint learning updates", flush=True)
        metrics = runner.collect_and_update(env, expert, rollout_steps=max(args.steps, 10))
        assert "world_loss" in metrics
        expected_std = torch.tensor((0.05, 0.02, 0.05) * 4, device=env.device) * (
            env._robot.data.soft_joint_pos_limits[0, env._joint_ids, 1]
            - env._robot.data.soft_joint_pos_limits[0, env._joint_ids, 0]
        )
        torch.testing.assert_close(runner.ppo.min_std, expected_std)
        assert runner.cfg.gamma == 0.99
        step_before = runner.rollout_step
        runner.collect_and_update(env, expert, rollout_steps=5)
        assert runner.rollout_step == step_before + 5
        with tempfile.TemporaryDirectory(prefix=f"wmp_{robot}_") as directory:
            checkpoint = runner.save(Path(directory) / "checkpoint.pt")
            clone = WMPRunner(runner.cfg, env.device)
            clone.load(checkpoint)
            assert clone.iteration == 2
            expected_amp = expert.sample_transitions(8)
            resumed_expert = AMPMotionDataset(motion_dir, env.device, seed=7, sampling_mode="upstream_v1", preload=64)
            clone.bind_expert(resumed_expert)
            for actual, expected in zip(resumed_expert.sample_transitions(8), expected_amp, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            clone.collect_and_update(env, resumed_expert, rollout_steps=10)
            assert clone.iteration == 3
            path, error = export_policy(clone, Path(directory) / "policy.pt", batch_size=2)
            assert path.is_file() and error <= 1e-4
        print(
            f"[PASS] {robot}: all terrains, WMP Warp depth, reset, PPO/AMP/RSSM, checkpoint and TorchScript", flush=True
        )
        if os.environ.get("WMP_VALIDATION_OUTPUT"):
            from pxr import PhysxSchema, UsdPhysics

            offsets, self_collision_attributes = [], []
            for prim in env.sim.stage.Traverse():
                if str(prim.GetPath()).startswith("/World/envs/env_0/Robot/") and prim.HasAPI(UsdPhysics.CollisionAPI):
                    offsets.append(PhysxSchema.PhysxCollisionAPI(prim).GetContactOffsetAttr().Get())
                if str(prim.GetPath()).startswith("/World/envs/env_0/Robot") and prim.HasAPI(
                    UsdPhysics.ArticulationRootAPI
                ):
                    self_collision_attributes.append(
                        PhysxSchema.PhysxArticulationAPI(prim).GetEnabledSelfCollisionsAttr().Get()
                    )
            assert offsets and all(abs(value - 0.01) < 1e-6 for value in offsets)
            assert self_collision_attributes and all(value is True for value in self_collision_attributes)
            if robot == "go2":
                assert len(env._robot.body_names) == 19, env._robot.body_names
                assert len(offsets) == 27, len(offsets)
                assert abs(float(env._robot.data.default_mass[0].sum()) - 16.086998) < 1e-4
            target = Path(os.environ["WMP_VALIDATION_OUTPUT"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(
                    dict(
                        passed=True,
                        robot=robot,
                        environment_revision=env.cfg.environment_revision,
                        nominal_mass_kg=float(env._robot.data.default_mass[0].sum()),
                        body_names=env._robot.body_names,
                        foot_sensor_names=[env._contact_sensor.body_names[i] for i in env._feet_ids],
                        foot_body_names=[env._robot.body_names[i] for i in env._feet_body_ids],
                        self_collision=env.cfg.robot.spawn.articulation_props.enabled_self_collisions,
                        actual_self_collision_attributes=self_collision_attributes,
                        collision_shapes=len(offsets),
                        camera=env._depth_camera.buffer_report(),
                        config_audit=config_audit,
                        contact_offsets=offsets,
                        gamma=runner.cfg.gamma,
                        amp_clips=expert.num_motions,
                        terrain_types=sorted(set(env.terrain_names)),
                        metrics=metrics,
                        checkpoint_resumed_iteration=clone.iteration,
                        export_max_abs_error=error,
                        environment_fingerprint=runner.environment_contract["fingerprint"],
                    ),
                    indent=2,
                )
            )
        print("[METRICS] " + " ".join(f"{k}={v:.5g}" for k, v in metrics.items()), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    # 此工作站的 Kit 关闭可能挂起；仅固定步 smoke 使用立即退出。
    os._exit(0)
