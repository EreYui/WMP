# SPDX-License-Identifier: Apache-2.0
"""DirectRLEnv implementation of the WMP Go2 observation and control contract."""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster, RayCasterCamera

from wmp_go2.constants import FORWARD_HEIGHT_DIM, GO2_JOINT_ORDER, HEIGHT_DIM, OBSERVATION_DIM

from .go2_env_cfg import WMPGo2RoughEnvCfg


class WMPGo2Env(DirectRLEnv):
    cfg: WMPGo2RoughEnvCfg

    def __init__(self, cfg: WMPGo2RoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_ids, names = self._robot.find_joints(list(GO2_JOINT_ORDER), preserve_order=True)
        if tuple(names) != GO2_JOINT_ORDER:
            raise RuntimeError(f"Go2 joint mapping mismatch: {names}")
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_foot", preserve_order=True)
        self._undesired_ids, _ = self._contact_sensor.find_bodies([".*_thigh", ".*_calf"], preserve_order=True)
        self._actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._global_step = 0
        self._friction = torch.ones(self.num_envs, 1, device=self.device)
        self._restitution = torch.zeros(self.num_envs, 1, device=self.device)
        self._added_mass = torch.zeros(self.num_envs, 1, device=self.device)
        self._com_offset = torch.zeros(self.num_envs, 3, device=self.device)
        self._p_gain_scale = torch.zeros(self.num_envs, 12, device=self.device)
        self._d_gain_scale = torch.zeros(self.num_envs, 12, device=self.device)
        self._motor_strength = torch.ones(self.num_envs, 12, device=self.device)
        self._latency_actions = torch.zeros_like(self._actions)
        self.camera_mask = torch.arange(self.num_envs, device=self.device) < min(
            self.cfg.logical_camera_envs, self.num_envs
        )
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self._body_height_scanner = RayCaster(self.cfg.body_height_scanner)
        self._forward_height_scanner = RayCaster(self.cfg.forward_height_scanner)
        self._depth_camera = RayCasterCamera(self.cfg.depth_camera)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.scene.sensors["body_height_scanner"] = self._body_height_scanner
        self.scene.sensors["forward_height_scanner"] = self._forward_height_scanner
        self.scene.sensors["depth_camera"] = self._depth_camera
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._previous_actions.copy_(self._actions)
        self._actions.copy_(actions.clamp(-6.0, 6.0))
        if self.cfg.enable_domain_randomization:
            use_delayed = torch.rand(self.num_envs, 1, device=self.device) < 0.25
            applied_actions = torch.where(use_delayed, self._latency_actions, self._actions)
            self._latency_actions.copy_(self._actions)
        else:
            applied_actions = self._actions
        self._processed_actions = (
            self._robot.data.default_joint_pos[:, self._joint_ids]
            + self.cfg.action_scale * applied_actions * self._motor_strength
        )
        self._global_step += 1
        interval = max(1, int(self.cfg.command_resampling_s / self.step_dt))
        env_ids = ((self.episode_length_buf % interval) == 0).nonzero(as_tuple=False).flatten()
        if env_ids.numel():
            self._resample_commands(env_ids)

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        self._commands[env_ids, 0].uniform_(*self.cfg.command_x_range)
        self._commands[env_ids, 1].uniform_(*self.cfg.command_y_range)
        self._commands[env_ids, 2].uniform_(*self.cfg.command_yaw_range)

    def _height(self, scanner: RayCaster, expected_dim: int) -> torch.Tensor:
        height = (self._robot.data.root_pos_w[:, 2:3] - scanner.data.ray_hits_w[..., 2] - 0.30).clamp(-1.0, 1.0)
        height = torch.nan_to_num(height, nan=0.0, posinf=1.0, neginf=-1.0) * 5.0
        if height.shape[-1] != expected_dim:
            raise RuntimeError(f"ray pattern produced {height.shape[-1]} samples, expected {expected_dim}")
        return height

    def get_forward_height_map(self) -> torch.Tensor:
        return self._height(self._forward_height_scanner, FORWARD_HEIGHT_DIM)

    def get_depth(self) -> torch.Tensor:
        depth = self._depth_camera.data.output["distance_to_camera"]
        return torch.nan_to_num(depth, nan=2.0, posinf=2.0, neginf=0.0).clamp(0.0, 2.0) / 2.0

    def get_amp_observations(self) -> torch.Tensor:
        return torch.cat(
            (
                self._robot.data.joint_pos[:, self._joint_ids],
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.joint_vel[:, self._joint_ids],
            ),
            dim=-1,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        contact_force = self._contact_sensor.data.net_forces_w[:, self._feet_ids].flatten(1) * 0.005
        history_force = self._contact_sensor.data.net_forces_w_history
        undesired = (
            torch.max(torch.norm(history_force[:, :, self._undesired_ids], dim=-1), dim=1).values > 0.1
        ).float()
        privileged = torch.cat(
            (
                self._friction,
                self._restitution,
                self._added_mass,
                self._com_offset * 20.0,
                self._p_gain_scale * 5.0,
                self._d_gain_scale * 5.0,
                contact_force,
                undesired,
                self._robot.data.root_lin_vel_b,
            ),
            dim=-1,
        )
        proprioception = torch.cat(
            (
                self._robot.data.root_ang_vel_b * 0.25,
                self._robot.data.projected_gravity_b,
                self._commands,
                self._robot.data.joint_pos[:, self._joint_ids] - self._robot.data.default_joint_pos[:, self._joint_ids],
                self._robot.data.joint_vel[:, self._joint_ids] * 0.05,
            ),
            dim=-1,
        )
        observation = torch.cat(
            (privileged, proprioception, self._actions, self._height(self._body_height_scanner, HEIGHT_DIM)), dim=-1
        ).clamp(-100.0, 100.0)
        if observation.shape[-1] != OBSERVATION_DIM:
            raise RuntimeError(f"observation layout produced {observation.shape[-1]} values")
        return {"policy": observation}

    def _get_rewards(self) -> torch.Tensor:
        lin_error = (self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]).square().sum(-1)
        yaw_error = (self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2]).square()
        torque = self._robot.data.applied_torque[:, self._joint_ids].square().sum(-1)
        acceleration = self._robot.data.joint_acc[:, self._joint_ids].square().sum(-1)
        action_rate = (self._actions - self._previous_actions).square().sum(-1)
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        air_time = ((self._contact_sensor.data.last_air_time[:, self._feet_ids] - 0.5) * first_contact).sum(-1)
        moving = torch.norm(self._commands[:, :2], dim=-1) > 0.1
        reward = (
            1.5 * torch.exp(-lin_error / 0.15)
            + 0.5 * torch.exp(-yaw_error / 0.15)
            - 0.0001 * torque
            - 2.5e-7 * acceleration
            - 0.03 * action_rate
            + 0.5 * air_time * moving
            - self._robot.data.root_lin_vel_b[:, 2].square()
        ) * self.step_dt
        return reward.clamp_min(0.0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        history_force = self._contact_sensor.data.net_forces_w_history
        base_force = torch.max(torch.norm(history_force[:, :, self._base_id], dim=-1), dim=1).values
        terminated = torch.any(base_force > 1.0, dim=1)
        timed_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, timed_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._latency_actions[env_ids] = 0.0
        if self.cfg.enable_domain_randomization:
            self._friction[env_ids].uniform_(0.5, 2.0)
            self._added_mass[env_ids].uniform_(0.0, 3.0)
            self._com_offset[env_ids].uniform_(-0.05, 0.05)
            self._p_gain_scale[env_ids].uniform_(-0.2, 0.2)
            self._d_gain_scale[env_ids].uniform_(-0.2, 0.2)
            self._motor_strength[env_ids].uniform_(0.8, 1.2)
        self._resample_commands(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
