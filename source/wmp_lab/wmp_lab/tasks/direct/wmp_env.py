# SPDX-License-Identifier: Apache-2.0
"""DirectRLEnv implementation of the WMP Go2 observation and control contract."""

from __future__ import annotations

import random

import isaaclab.sim as sim_utils
import torch
from wmp_lab.learning.progress_rewards import (
    ForwardProgressGate, PROGRESS_REWARD_REVISION, gate_positive_task_terms, tracking_velocity_with_progress,
)
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

from wmp_lab.constants import FORWARD_HEIGHT_DIM, GO2_JOINT_ORDER, HEIGHT_DIM, OBSERVATION_DIM
from wmp_lab.learning.curriculum import CommandCurriculum, TerrainCurriculumSlots
from wmp_lab.learning.task_rewards import stuck_penalty, tracking_velocity
from wmp_lab.reference.heights import feet_at_edge, height_points, rotate_height_points, sample_heights
from wmp_lab.robots import ENVIRONMENT_REVISION, EXPLICIT_PD_REVISIONS, robot_profile
from wmp_lab.sensors.depth import SparseRayCasterCamera
from wmp_lab.sensors.wmp_warp_camera import WMPWarpDepthCamera

from .terrains import next_terrain_levels
from .wmp_env_cfg import WMPEnvCfg


class WMPEnv(DirectRLEnv):
    cfg: WMPEnvCfg

    def __init__(self, cfg: WMPEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.depth_backend == "isaac":
            cfg.sim.use_fabric = False
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_ids, names = self._robot.find_joints(list(GO2_JOINT_ORDER), preserve_order=True)
        if tuple(names) != GO2_JOINT_ORDER:
            raise RuntimeError(f"Robot joint mapping mismatch: {names}")
        self._base_id, _ = self._robot.find_bodies("base")
        self._base_contact_ids, _ = self._contact_sensor.find_bodies("base")
        foot_names = [f"{leg}_foot" for leg in ("FL", "FR", "RL", "RR")]
        self._feet_ids, sensor_feet = self._contact_sensor.find_bodies(foot_names, preserve_order=True)
        self._feet_body_ids, body_feet = self._robot.find_bodies(foot_names, preserve_order=True)
        if sensor_feet != body_feet or sensor_feet != foot_names:
            raise RuntimeError("Contact and articulation foot name mappings differ")
        self._undesired_ids, _ = self._contact_sensor.find_bodies([".*_thigh", ".*_calf"], preserve_order=True)
        self._actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._global_step = 0
        self._physics_substep = self._latency_substeps = 0
        self._actuator_joint_ids = [self._robot.actuators["legs"].joint_names.index(name) for name in GO2_JOINT_ORDER]
        self._friction = torch.ones(self.num_envs, 1, device=self.device)
        self._restitution = torch.zeros(self.num_envs, 1, device=self.device)
        self._added_mass = torch.zeros(self.num_envs, 1, device=self.device)
        self._com_offset = torch.zeros(self.num_envs, 3, device=self.device)
        self._p_gain_scale = torch.zeros(self.num_envs, 12, device=self.device)
        self._d_gain_scale = torch.zeros(self.num_envs, 12, device=self.device)
        self._motor_strength = torch.ones(self.num_envs, 12, device=self.device)
        self._latency_actions = torch.zeros_like(self._actions)
        self._episode_start_pos = self._robot.data.root_pos_w[:, :2].clone()
        self._tracking_episode_sum = torch.zeros(self.num_envs, device=self.device)
        self._heading_targets = torch.zeros(self.num_envs, device=self.device)
        self._last_last_actions = torch.zeros_like(self._actions)
        self._feet_last_z = torch.zeros(self.num_envs, 4, device=self.device)
        self._feet_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self._last_foot_contacts = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self._previous_joint_velocity = torch.zeros(self.num_envs, 12, device=self.device)
        self._progress_tracker = (
            ForwardProgressGate(self.num_envs, self.step_dt, self.device,
                                cfg.progress_window_s, cfg.progress_min_speed)
            if getattr(cfg, "reward_revision", None) == PROGRESS_REWARD_REVISION else None
        )
        self._reward_curriculum_scale = 0.1
        self.command_curriculum = CommandCurriculum(self.terrain_names, self.device)
        self._terrain_curriculum_slots = None
        self._depth_frames = None
        self._depth_frame_step = -1
        self._camera_randomized = False
        self._terminal_observation = None
        self._terminal_amp = None
        from pxr import UsdPhysics

        prim = self.sim.stage.GetPrimAtPath(self._robot.root_physx_view.link_paths[0][self._base_id[0]])
        nominal_com = UsdPhysics.MassAPI(prim).GetCenterOfMassAttr().Get()
        self._nominal_base_com = torch.tensor(tuple(nominal_com) if nominal_com else (0, 0, 0), device=self.device)
        profile = robot_profile(
            cfg.robot_name, control_revision=cfg.control_revision, environment_revision=cfg.environment_revision
        )
        self._nominal_stiffness, self._nominal_damping = profile.stiffness, profile.damping
        self._sync_physical_parameters()
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

    @property
    def reference_mode(self):
        return self.cfg.environment_revision == ENVIRONMENT_REVISION

    def configure_training_schedule(self, reference_envs):
        """速度课程按参考分组回合数评分，不因物理环境数降低而提前升速。"""
        generator = self.cfg.terrain.terrain_generator
        proportions = [(name, cfg.proportion) for name, cfg in generator.sub_terrains.items()]
        self.command_curriculum.configure_reference(proportions, reference_envs)
        if reference_envs % self.num_envs:
            raise ValueError("terrain curriculum reference must be divisible by physical environment count")
        slots = reference_envs // self.num_envs
        if self._terrain_curriculum_slots is None:
            self._terrain_curriculum_slots = TerrainCurriculumSlots(self._terrain.terrain_levels, slots)
        elif self._terrain_curriculum_slots.levels.shape[1] != slots:
            raise ValueError("cannot change terrain curriculum slot count during a run")

    @property
    def curriculum_levels(self):
        slots = self._terrain_curriculum_slots
        return self._terrain.terrain_levels if slots is None else slots.levels.flatten()

    def reset(self, *args, **kwargs):
        observation, info = super().reset(*args, **kwargs)
        if self.reference_mode:
            observation, _, _, _, info = self.step(torch.zeros(self.num_envs, 12, device=self.device))
        return observation, info

    def step(self, actions):
        result = super().step(actions)
        if self.reference_mode:
            self.get_depth()
        return result

    def _configure_terrain_types(self) -> None:
        generator_cfg = self.cfg.terrain.terrain_generator
        if generator_cfg is None or not hasattr(self._terrain, "terrain_types"):
            self._terrain_type_masks = {}
            self.terrain_names = ["flat"] * self.num_envs
        else:
            names = tuple(generator_cfg.sub_terrains)
            proportions = torch.tensor(
                [generator_cfg.sub_terrains[name].proportion for name in names], device=self.device
            )
            proportions /= proportions.sum()
            choices = torch.arange(generator_cfg.num_cols, device=self.device) / generator_cfg.num_cols + 0.001
            columns = torch.searchsorted(proportions.cumsum(0), choices)
            env_type = columns[self._terrain.terrain_types]
            self._terrain_type_masks = {name: env_type == i for i, name in enumerate(names)}
            self.terrain_names = [names[i].removeprefix("wmp_") for i in env_type.tolist()]
        self._directional_terrain_mask = torch.tensor(
            [
                name not in {"wave", "rough_slope", "stairs_up", "stairs_down", "discrete", "rough_flat", "flat"}
                for name in self.terrain_names
            ],
            device=self.device,
        )
        self._jump_terrain_mask = torch.tensor(
            [name in {"gap", "continuous_gap", "plum_piles"} for name in self.terrain_names], device=self.device
        )
        self._tilt_terrain_mask = self._terrain_type_masks.get(
            "wmp_tilt", torch.zeros_like(self._directional_terrain_mask)
        )

    def _make_stratified_camera_mask(self) -> torch.Tensor:
        """Prioritize 3-D obstacles, then spread remaining real cameras across all terrain columns."""
        budget = min(self.cfg.logical_camera_envs, self.num_envs)
        selected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        priority = torch.zeros_like(selected)
        upstream = self.cfg.reward_semantics == "upstream_v3"
        priority_names = (
            (
                "wmp_gap",
                "wmp_continuous_gap",
                "wmp_bream",
                "wmp_stairs_cliff",
                "wmp_slope_cliff",
                "wmp_stepping_stones",
                "wmp_stepping_one_bridge",
                "wmp_climb",
            )
            if upstream
            else self.cfg.camera_priority_terrains
        )
        for name in priority_names:
            priority |= self._terrain_type_masks.get(name, torch.zeros_like(selected))

        def evenly_spaced(ids: torch.Tensor, count: int) -> torch.Tensor:
            if count <= 0 or ids.numel() == 0:
                return ids[:0]
            if count >= ids.numel():
                return ids
            if self.reference_mode:
                return ids[torch.randperm(ids.numel(), device=self.device)[:count]]
            positions = torch.div(torch.arange(count, device=self.device) * ids.numel(), count, rounding_mode="floor")
            return ids[positions]

        priority_ids = priority.nonzero(as_tuple=False).flatten()
        mandatory = torch.zeros_like(selected)
        for name in self.cfg.camera_mandatory_terrains:
            mandatory |= self._terrain_type_masks.get(name, torch.zeros_like(selected))
        if int(mandatory.sum()) > budget:
            raise ValueError(f"camera budget {budget} cannot cover {int(mandatory.sum())} mandatory 3-D environments")
        selected[mandatory] = True
        priority_ids = (priority & ~mandatory).nonzero(as_tuple=False).flatten()
        priority_count = budget - int(mandatory.sum())
        if upstream:
            priority_count = int(round(0.8 * priority_count)) if self.reference_mode else int(0.8 * priority_count)
        chosen_priority = evenly_spaced(priority_ids, priority_count)
        selected[chosen_priority] = True
        remaining = budget - int(selected.sum())
        candidate_ids = (~selected & ~priority if upstream else ~selected).nonzero(as_tuple=False).flatten()
        selected[evenly_spaced(candidate_ids, remaining)] = True
        remaining = budget - int(selected.sum())
        selected[evenly_spaced((~selected).nonzero(as_tuple=False).flatten(), remaining)] = True
        return selected

    def _setup_scene(self) -> None:
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self._configure_terrain_types()
        self.camera_mask = self._make_stratified_camera_mask()
        self._camera_env_ids = self.camera_mask.nonzero(as_tuple=False).flatten()
        self.depth_supervision_mask = self.camera_mask.clone()
        for name in self.cfg.camera_mandatory_terrains:
            self.depth_supervision_mask &= ~self._terrain_type_masks.get(name, torch.zeros_like(self.camera_mask))
        # 在克隆环境前改模板，使全部 env 继承同一碰撞修订。
        from wmp_lab.reference.collisions import apply_collision_revision

        self._disabled_collision_prims = apply_collision_revision(
            self.sim.stage,
            self._robot.cfg.prim_path.replace("env_.*", "env_0"),
            self.cfg.robot_name,
            getattr(self.cfg, "collision_revision", "native_v1"),
        )
        self.scene.clone_environments(copy_from_source=False)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self._body_height_scanner = self._forward_height_scanner = None
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if self.reference_mode:
            self._height_points = height_points(device=self.device)
            self._forward_height_points = height_points(forward=True, device=self.device)
        else:
            self._body_height_scanner = RayCaster(self.cfg.body_height_scanner)
            self._forward_height_scanner = RayCaster(self.cfg.forward_height_scanner)
            self.scene.sensors["body_height_scanner"] = self._body_height_scanner
            self.scene.sensors["forward_height_scanner"] = self._forward_height_scanner
        self._depth_camera = None
        if len(self._camera_env_ids):
            sensor_cfg = self.cfg.depth_camera.copy()
            if self.cfg.depth_backend == "warp" and self.cfg.camera_implementation == "wmp_go1":
                if not self.reference_mode:
                    raise ValueError("wmp_go1 camera requires the reference terrain mesh")
                self._depth_camera = WMPWarpDepthCamera(
                    self._terrain.render_vertices,
                    self._terrain.render_triangles,
                    self._camera_env_ids,
                    device=self.device,
                    local_positions=sensor_cfg.offset.pos,
                    local_quaternions_ros=sensor_cfg.offset.rot,
                    width=sensor_cfg.pattern_cfg.width,
                    height=sensor_cfg.pattern_cfg.height,
                )
                self._depth_output_key = "distance_to_image_plane"
            elif self.cfg.depth_backend == "warp":
                self._depth_camera = SparseRayCasterCamera(sensor_cfg, self.camera_mask)
                self._depth_output_key = "distance_to_image_plane"
            else:
                self._depth_camera = Camera(
                    CameraCfg(
                        prim_path="/World/envs/env_.*/DepthCamera",
                        update_period=0.1,
                        height=64,
                        width=64,
                        data_types=["distance_to_image_plane"],
                        update_latest_camera_pose=True,
                        spawn=sim_utils.PinholeCameraCfg(
                            focal_length=17.0,
                            horizontal_aperture=sensor_cfg.pattern_cfg.horizontal_aperture,
                            clipping_range=(0.01, 2.0),
                        ),
                        offset=CameraCfg.OffsetCfg(
                            pos=sensor_cfg.offset.pos, rot=sensor_cfg.offset.rot, convention="ros"
                        ),
                    )
                )
                self._depth_output_key = "distance_to_image_plane"
            if not isinstance(self._depth_camera, WMPWarpDepthCamera):
                self.scene.sensors["depth_camera"] = self._depth_camera
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._last_last_actions.copy_(self._previous_actions)
        self._previous_actions.copy_(self._actions)
        self._actions.copy_(actions.clamp(-6.0, 6.0))
        if self.cfg.control_revision in EXPLICIT_PD_REVISIONS:
            self._physics_substep = 0
            self._latency_substeps = random.randint(0, 2) if self.cfg.enable_domain_randomization else 0
            applied_actions = self._actions
        elif self.cfg.enable_domain_randomization:
            use_delayed = torch.rand(self.num_envs, 1, device=self.device) < 0.25
            applied_actions = torch.where(use_delayed, self._latency_actions, self._actions)
            self._latency_actions.copy_(self._actions)
        else:
            applied_actions = self._actions
        self._processed_actions = (
            self._robot.data.default_joint_pos[:, self._joint_ids] + self.cfg.action_scale * applied_actions
        )
        self._global_step += 1
        interval = max(1, int(self.cfg.command_resampling_s / self.step_dt))
        env_ids = ((self.episode_length_buf % interval) == 0).nonzero(as_tuple=False).flatten()
        if not self.reference_mode:
            if env_ids.numel():
                self._resample_commands(env_ids)
            self._update_heading_commands()

    def _apply_action(self) -> None:
        if self.cfg.control_revision in EXPLICIT_PD_REVISIONS:
            delayed = self._physics_substep < self._latency_substeps
            actions = self._previous_actions if delayed else self._actions
            self._processed_actions = (
                self._robot.data.default_joint_pos[:, self._joint_ids] + self.cfg.action_scale * actions
            )
            self._physics_substep += 1
            if self.cfg.enable_domain_randomization:
                self._motor_strength.uniform_(0.8, 1.2)
            else:
                self._motor_strength.fill_(1)
            self._robot.actuators["legs"].effort_scale[:, self._actuator_joint_ids] = self._motor_strength
            self._robot.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
            self._robot.set_joint_effort_target(torch.zeros_like(self._actions), joint_ids=self._joint_ids)
            return
        self._robot.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
        actuator = self._robot.actuators["legs"]
        nominal_torque = (
            actuator.stiffness[:, self._joint_ids]
            * (self._processed_actions - self._robot.data.joint_pos[:, self._joint_ids])
            - actuator.damping[:, self._joint_ids] * self._robot.data.joint_vel[:, self._joint_ids]
        )
        self._robot.set_joint_effort_target((self._motor_strength - 1) * nominal_torque, joint_ids=self._joint_ids)

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        self._commands[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.command_x_range)
        self._commands[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.command_y_range)
        self._commands[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.command_yaw_range)
        jump_ids = env_ids[self._jump_terrain_mask[env_ids]]
        if jump_ids.numel():
            self._commands[jump_ids, 0] = torch.empty(len(jump_ids), device=self.device).uniform_(
                *self.cfg.jump_command_x_range
            )
        tilt_ids = env_ids[self._tilt_terrain_mask[env_ids]]
        if tilt_ids.numel():
            self._commands[tilt_ids, 0] = torch.empty(len(tilt_ids), device=self.device).uniform_(
                *self.cfg.tilt_command_x_range
            )
        directional_ids = env_ids[self._directional_terrain_mask[env_ids]]
        self._commands[directional_ids, 2] = 0.0
        if self.cfg.enable_command_curriculum:
            velocity, heading = self.command_curriculum.sample(env_ids, self.cfg.zero_command_probability)
            self._commands[env_ids, :2] = velocity
            self._heading_targets[env_ids] = heading
            if self.cfg.robot_name == "go2" and self.cfg.reward_semantics != "upstream_v3":
                for selected, minimum in (
                    (jump_ids, self.cfg.jump_command_x_range[0]),
                    (tilt_ids, self.cfg.tilt_command_x_range[0]),
                ):
                    ids = selected[self._commands[selected, 0] > 0]
                    if len(ids):
                        upper = self.command_curriculum.upper[self.command_curriculum.groups[ids], 0]
                        lower = torch.minimum(torch.full_like(upper, minimum), upper)
                        self._commands[ids, 0] = lower + torch.rand(len(ids), device=self.device) * (upper - lower)
            if self.cfg.reward_semantics == "upstream_v3":
                self._commands[directional_ids, :2] *= (
                    self._commands[directional_ids, :2].norm(dim=1) > 0.2
                ).unsqueeze(1)

    def _update_heading_commands(self) -> None:
        """Keep obstacle-course robots aligned with positive world x."""
        heading_error = self._heading_targets - self._robot.data.heading_w
        heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
        yaw_limit = self.cfg.heading_yaw_limit
        yaw_command = (self.cfg.heading_command_gain * heading_error).clamp(-yaw_limit, yaw_limit)
        if self.cfg.enable_command_curriculum:
            self._commands[:, 2] = yaw_command
        else:
            self._commands[self._directional_terrain_mask, 2] = yaw_command[self._directional_terrain_mask]

    def _height(self, scanner: RayCaster, expected_dim: int) -> torch.Tensor:
        if self.reference_mode:
            points = self._height_points if expected_dim == HEIGHT_DIM else self._forward_height_points
            world = rotate_height_points(self._robot.data.root_quat_w, points)
            world += self._robot.data.root_pos_w[:, None]
            generator = self.cfg.terrain.terrain_generator
            terrain_height = sample_heights(
                self._terrain.height_samples,
                world,
                self._terrain.heightfield_border,
                generator.horizontal_scale,
                generator.vertical_scale,
            )
        else:
            terrain_height = scanner.data.ray_hits_w[..., 2]
        height = (self._robot.data.root_pos_w[:, 2:3] - terrain_height - 0.30).clamp(-1.0, 1.0)
        height = torch.nan_to_num(height, nan=0.0, posinf=1.0, neginf=-1.0) * 5.0
        if height.shape[-1] != expected_dim:
            raise RuntimeError(f"ray pattern produced {height.shape[-1]} samples, expected {expected_dim}")
        return height

    def get_forward_height_map(self) -> torch.Tensor:
        return self._height(self._forward_height_scanner, FORWARD_HEIGHT_DIM)

    def get_depth(self) -> torch.Tensor:
        if self._depth_frames is None:
            self._depth_frames = torch.full(
                (self.num_envs, 2, 64, 64, 1), 0.0 if self.reference_mode else 0.5, device=self.device
            )
        if self._global_step % self.cfg.camera_update_interval == 0 and self._depth_frame_step != self._global_step:
            self._depth_frame_step = self._global_step
            if self._depth_camera is not None:
                if self.cfg.depth_backend == "isaac":
                    self._sync_rtx_camera_pose()
                    for _ in range(3):
                        self.sim.render()
                    # force_recompute 仅计算过时 buffer；显式使新位姿的帧失效。
                    self._depth_camera._is_outdated[:] = True
                    self._depth_camera.update(0.0, force_recompute=True)
                if isinstance(self._depth_camera, WMPWarpDepthCamera):
                    depth = self._depth_camera.render(
                        self._robot.data.root_pos_w[self._camera_env_ids],
                        self._robot.data.root_quat_w[self._camera_env_ids],
                    )
                else:
                    depth = self._depth_camera.data.output[self._depth_output_key][self._camera_env_ids]
                depth = torch.nan_to_num(depth, nan=2.0, posinf=2.0, neginf=0.0).clamp(0, 2) / 2 - 0.5
                self._depth_frames[self._camera_env_ids, 0] = self._depth_frames[self._camera_env_ids, 1].clone()
                self._depth_frames[self._camera_env_ids, 1] = depth
                if self.reference_mode:
                    initial = self._camera_env_ids[self.episode_length_buf[self._camera_env_ids] <= 1]
                    self._depth_frames[initial, 0] = self._depth_frames[initial, 1]
        return self._depth_frames[:, 1 - self.cfg.camera_delay_frames]

    def _sync_rtx_camera_pose(self):
        # Fabric 不保证机器人子相机的 USD 位姿同步；显式设置独立相机世界位姿。
        offset = getattr(
            self,
            "_camera_offset_pos",
            torch.tensor(self.cfg.depth_camera.offset.pos, device=self.device).expand(self.num_envs, -1),
        )
        orientation = getattr(
            self,
            "_camera_offset_quat",
            torch.tensor(self.cfg.depth_camera.offset.rot, device=self.device).expand(self.num_envs, -1),
        )
        root_quat = self._robot.data.root_quat_w
        positions = self._robot.data.root_pos_w + quat_apply(root_quat, offset)
        orientations = quat_mul(root_quat, orientation)
        self._depth_camera.set_world_poses(positions, orientations, convention="ros")

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
        if self.reference_mode:
            # 上游通过前插构造特权信息，保持每个字段的真实语义顺序。
            undesired = (self._contact_sensor.data.net_forces_w[:, self._undesired_ids].norm(dim=-1) > 0.1).float()
            privileged = torch.cat(
                (
                    undesired,
                    contact_force,
                    self._d_gain_scale * 5,
                    self._p_gain_scale * 5,
                    self._com_offset * 20,
                    self._added_mass,
                    self._restitution,
                    self._friction,
                    self._robot.data.root_lin_vel_b,
                ),
                -1,
            )
        proprioception = torch.cat(
            (
                self._robot.data.root_ang_vel_b * 0.25,
                self._robot.data.projected_gravity_b,
                self._commands * torch.tensor((1.0, 1.0, 0.25), device=self.device),
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
        if self.cfg.enable_observation_noise:
            noise = torch.zeros_like(observation)
            noise[:, 53:56] = 0.2 * 0.25
            noise[:, 56:59] = 0.05
            noise[:, 62:74] = 0.01
            noise[:, 74:86] = 1.5 * 0.05
            observation += (torch.rand_like(observation) * 2 - 1) * noise
        return {"policy": observation}

    def _get_rewards(self) -> torch.Tensor:
        upstream = self.cfg.reward_semantics == "upstream_v3"
        progress_status = None
        if getattr(self, "_progress_tracker", None) is not None:
            eligible = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for name in self.cfg.progress_reward_terrains:
                eligible |= self._terrain_type_masks.get(name, torch.zeros_like(eligible))
            progress_status = self._progress_tracker.update(
                self._robot.data.root_pos_w[:, 0], self._commands[:, 0], eligible
            )
        lin_error = (self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]).square().sum(-1)
        yaw_error = (self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2]).square()
        tracking = torch.exp(-lin_error / 0.25)
        if upstream:
            obstacles = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            for name in ("wmp_gap", "wmp_bream", "wmp_continuous_gap", "wmp_climb"):
                obstacles |= self._terrain_type_masks.get(name, torch.zeros_like(obstacles))
            if progress_status is None:
                tracking = tracking_velocity(self._robot.data.root_lin_vel_b, self._commands, obstacles)
            else:
                tracking = tracking_velocity_with_progress(
                    self._robot.data.root_lin_vel_b, self._commands, obstacles,
                    progress_status.speed, self._progress_tracker.direction, self._progress_tracker.active,
                )
        torque = self._robot.data.applied_torque[:, self._joint_ids].square().sum(-1)
        acceleration = self._robot.data.joint_acc[:, self._joint_ids].square().sum(-1)
        if upstream:
            velocity = self._robot.data.joint_vel[:, self._joint_ids]
            acceleration = ((velocity - self._previous_joint_velocity) / self.step_dt).square().sum(-1)
            self._previous_joint_velocity.copy_(velocity)
        action_rate = (self._actions - self._previous_actions).square().sum(-1)
        smoothness = (self._actions - 2 * self._previous_actions + self._last_last_actions).square().sum(-1)
        contact = self._contact_sensor.data.net_forces_w
        collision = (contact[:, self._undesired_ids].norm(dim=-1) > 0.1).float().sum(-1)
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        air_time = ((self._contact_sensor.data.last_air_time[:, self._feet_ids] - 0.5) * first_contact).sum(-1)
        moving = self._commands[:, :2].norm(dim=-1) > 0.1
        foot_force = contact[:, self._feet_ids]
        if upstream:
            contacts = foot_force[..., 2] > 1.0
            filtered = contacts | self._last_foot_contacts
            first_contact = (self._feet_air_time > 0) & filtered
            self._last_foot_contacts.copy_(contacts)
            self._feet_air_time += self.step_dt
            air_time = ((self._feet_air_time - 0.5) * first_contact).sum(-1)
            self._feet_air_time *= ~filtered
        stumble = (foot_force[..., :2].norm(dim=-1) > (4 if upstream else 5) * foot_force[..., 2].abs()).any(-1)
        joint_pos = self._robot.data.joint_pos[:, self._joint_ids]
        limits = self._robot.data.soft_joint_pos_limits[:, self._joint_ids]
        pos_limit = ((limits[..., 0] - joint_pos).clamp_min(0) + (joint_pos - limits[..., 1]).clamp_min(0)).sum(-1)
        vel_limit = (
            (
                self._robot.data.joint_vel[:, self._joint_ids].abs()
                - self._robot.data.joint_vel_limits[:, self._joint_ids] * 0.9
            )
            .clamp(0, 1)
            .sum(-1)
        )
        vertical = self._robot.data.root_lin_vel_b[:, 2].square()
        if not upstream:
            vertical[self._jump_terrain_mask] *= self.cfg.jump_vertical_velocity_penalty_scale
        progress = self._robot.data.root_lin_vel_w[:, 0].clamp(-1, 1) * self._directional_terrain_mask
        stuck = (self._robot.data.root_lin_vel_w[:, 0].abs() < 0.1) & moving & self._directional_terrain_mask
        lateral = (self._robot.data.root_pos_w[:, 1] - self._terrain.env_origins[:, 1]).abs()
        cheat = (lateral > 1.2) & self._directional_terrain_mask
        if upstream:
            stuck = stuck_penalty(self._robot.data.root_lin_vel_b, self._commands)
            if progress_status is not None:
                stuck = torch.where(eligible, progress_status.stalled.float(), stuck)
            cheat = (self._robot.data.heading_w.abs() > 1.0) & self._directional_terrain_mask
        if self.reference_mode:
            generator = self.cfg.terrain.terrain_generator
            on_edge = feet_at_edge(
                self._terrain.x_edge_mask,
                self._robot.data.body_pos_w[:, self._feet_body_ids],
                filtered,
                self._terrain.heightfield_border,
                generator.horizontal_scale,
            )
            edge = on_edge.float().sum(-1)
            edge *= self._terrain_type_masks.get("wmp_gap", torch.zeros_like(moving)) | self._terrain_type_masks.get(
                "wmp_climb", torch.zeros_like(moving)
            )
        else:
            hits = self._body_height_scanner.data.ray_hits_w
            heights = hits[..., 2].reshape(self.num_envs, 17, 11)
            boundary = torch.zeros_like(heights, dtype=torch.bool)
            dx = (heights[:, 1:] - heights[:, :-1]).abs() > 0.3
            dy = (heights[:, :, 1:] - heights[:, :, :-1]).abs() > 0.3
            boundary[:, 1:] |= dx
            boundary[:, :-1] |= dx
            boundary[:, :, 1:] |= dy
            boundary[:, :, :-1] |= dy
            feet = self._robot.data.body_pos_w[:, self._feet_body_ids]
            nearest = (feet[:, :, None, :2] - hits[:, None, :, :2]).square().sum(-1).argmin(-1)
            edge = (boundary.flatten(1).gather(1, nearest) & (foot_force[..., 2] > 1)).float().sum(-1)
            if upstream:
                edge *= self._terrain_type_masks.get(
                    "wmp_gap", torch.zeros_like(moving)
                ) | self._terrain_type_masks.get("wmp_climb", torch.zeros_like(moving))
        stool = self._terrain_type_masks.get("wmp_stool", torch.zeros_like(moving))
        base_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        stool_height = (base_height - 0.25).square() * stool
        if upstream:
            stool_height = (base_height > 0.45).float() * stool
        self._reward_terms = dict(
            tracking=tracking,
            yaw_tracking=0.5 * torch.exp(-yaw_error / 0.25),
            torque=-0.0001 * torque,
            acceleration=-2.5e-7 * acceleration,
            action_rate=-0.03 * action_rate,
            smoothness=-0.01 * smoothness,
            collision=-collision,
            joint_pos_limits=-10 * pos_limit,
            joint_vel_limits=-10 * vel_limit,
            air_time=air_time * moving,
            vertical=-vertical,
            stumble=-stumble.float(),
            edge=-self._reward_curriculum_scale * edge.float(),
            progress=(0.0 if upstream else self.cfg.terrain_progress_reward_scale) * progress,
            stuck=-self.cfg.stuck_penalty_scale * stuck,
            cheat=-cheat.float(),
            stool_height=-stool_height,
        )
        if progress_status is not None:
            self._reward_terms = gate_positive_task_terms(self._reward_terms, progress_status.stalled)
            self.extras["locomotion_reward_gate"] = (~progress_status.stalled).float()
        self._tracking_episode_sum += self._reward_terms["tracking"]
        reward = sum(self._reward_terms.values()) * self.step_dt
        self._unclipped_task_reward = reward.clone()
        self._terminal_amp = self.get_amp_observations().clone()
        self._terminal_observation = self._get_observations()["policy"].clone()
        self.extras["terminal_amp_states"] = self._terminal_amp
        self.extras["terminal_observation"] = self._terminal_observation
        return reward.clamp_min(0)

    def update_reward_curriculum(self, iteration):
        self._reward_curriculum_scale = 0.1 + 0.9 * min(max((iteration - 4000) / 6000, 0), 1)

    def _sync_physical_parameters(self):
        view = self._robot.root_physx_view
        materials = view.get_material_properties().to(self.device)
        self._friction[:] = materials[..., 0].mean(-1, keepdim=True)
        self._restitution[:] = materials[..., 2].mean(-1, keepdim=True)
        masses = view.get_masses().to(self.device)
        self._added_mass[:] = (
            masses[:, self._base_id] - self._robot.data.default_mass[:, self._base_id].to(self.device)
        ).reshape(-1, 1)
        coms = view.get_coms().to(self.device)
        self._com_offset[:] = coms[:, self._base_id[0], :3] - self._nominal_base_com
        self._p_gain_scale[:] = (
            self._robot.actuators["legs"].stiffness[:, self._actuator_joint_ids] / self._nominal_stiffness - 1
        )
        self._d_gain_scale[:] = (
            self._robot.actuators["legs"].damping[:, self._actuator_joint_ids] / self._nominal_damping - 1
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.reference_mode:
            interval = max(1, int(self.cfg.command_resampling_s / self.step_dt))
            ids = (self.episode_length_buf % interval == 0).nonzero(as_tuple=False).flatten()
            if ids.numel():
                self._resample_commands(ids)
            self._update_heading_commands()
        history_force = self._contact_sensor.data.net_forces_w_history
        base_force = torch.max(torch.norm(history_force[:, :, self._base_contact_ids], dim=-1), dim=1).values
        if self.reference_mode:
            base_force = self._contact_sensor.data.net_forces_w[:, self._base_contact_ids].norm(dim=-1)
        terminated = torch.any(base_force > 1.0, dim=1)
        if self.cfg.reward_semantics == "upstream_v3":
            velocity_error = self._robot.data.root_lin_vel_b[:, 0] - self._commands[:, 0]
            violation = ((velocity_error > 1.5) & (self._commands[:, 0] < 0)) | (
                (velocity_error < -1.5) & (self._commands[:, 0] > 0)
            )
            levels = self._terrain.terrain_levels
            terminated |= violation & (levels > 3)
            terminated |= (self._robot.data.root_lin_vel_w[:, 2] < -3) | (
                self._robot.data.projected_gravity_b[:, 2] > 0
            )
        timed_out = (
            self.episode_length_buf > self.max_episode_length
            if self.reference_mode
            else self.episode_length_buf >= self.max_episode_length - 1
        )
        return terminated, timed_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        if self.cfg.enable_curriculum:
            self._update_terrain_curriculum(env_ids)
        if self.cfg.enable_command_curriculum:
            self.command_curriculum.accumulate(
                env_ids, self._tracking_episode_sum, self.episode_length_buf, self.max_episode_length
            )
        self._tracking_episode_sum[env_ids] = 0
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if not self.reference_mode:
            self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._latency_actions[env_ids] = 0.0
        self._last_last_actions[env_ids] = 0
        self._feet_air_time[env_ids] = 0
        self._last_foot_contacts[env_ids] = False
        self._previous_joint_velocity[env_ids] = 0
        if self.cfg.enable_domain_randomization:
            self._motor_strength[env_ids] = torch.empty(len(env_ids), 12, device=self.device).uniform_(0.8, 1.2)
        self._sync_physical_parameters()
        if self._depth_frames is not None and not self.reference_mode:
            self._depth_frames[env_ids] = 0.5
        if self.reference_mode and self.cfg.randomize_camera and not self._camera_randomized:
            self._randomize_reference_camera()
            self._camera_randomized = True
        if (
            not self.reference_mode
            and self.cfg.randomize_camera
            and self._depth_camera is not None
            and self.cfg.depth_backend == "warp"
        ):
            sensor_ids = env_ids[self.camera_mask[env_ids]]
            position = torch.tensor(self.cfg.depth_camera.offset.pos, device=self.device).expand(len(sensor_ids), -1)
            self._depth_camera._offset_pos[sensor_ids] = position + (torch.rand_like(position) * 2 - 1) * 0.01
            angle = torch.empty(len(sensor_ids), 3, device=self.device).uniform_(-2, 2) * torch.pi / 180
            angle[:, 1] *= 2.5
            base = torch.tensor(self.cfg.depth_camera.offset.rot, device=self.device).expand(len(sensor_ids), -1)
            # RayCasterCamera 内部使用 world 相机坐标。
            from isaaclab.utils.math import convert_camera_frame_orientation_convention

            base = convert_camera_frame_orientation_convention(base, origin="ros", target="world")
            self._depth_camera._offset_quat[sensor_ids] = quat_mul(quat_from_euler_xyz(*angle.unbind(-1)), base)
        self._resample_commands(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        if self.cfg.enable_domain_randomization:
            if self.cfg.reward_semantics == "upstream_v3":
                joint_pos = joint_pos * torch.empty_like(joint_pos).uniform_(0.5, 1.5)
                root_state[:, 7:13] = torch.empty(len(env_ids), 6, device=self.device).uniform_(-0.5, 0.5)
                jitter = torch.empty(len(env_ids), 2, device=self.device).uniform_(-1.0, 1.0)
                x_only = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                for name in ("wmp_gap", "wmp_continuous_gap", "wmp_bream", "wmp_stepping_one_bridge", "wmp_tilt"):
                    x_only |= self._terrain_type_masks.get(name, torch.zeros_like(x_only))
                half_range = x_only.clone()
                for name in ("wmp_stairs_cliff", "wmp_slope_cliff", "wmp_stepping_stones", "wmp_climb", "wmp_crawl"):
                    half_range |= self._terrain_type_masks.get(name, torch.zeros_like(x_only))
                jitter[half_range[env_ids]] *= 0.5
                jitter[x_only[env_ids], 1] = 0
                stool = self._terrain_type_masks.get("wmp_stool", torch.zeros_like(x_only))[env_ids]
                jitter[stool, 0] *= 0.25
                jitter[stool, 1] = 0
            else:
                jitter = torch.empty(len(env_ids), 2, device=self.device).uniform_(-0.25, 0.25)
                jitter[self._directional_terrain_mask[env_ids], 1] = 0
            root_state[:, :2] += jitter
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._episode_start_pos[env_ids] = root_state[:, :2]
        if getattr(self, "_progress_tracker", None) is not None:
            eligible = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
            for name in self.cfg.progress_reward_terrains:
                mask = self._terrain_type_masks.get(name)
                if mask is not None:
                    eligible |= mask[env_ids]
            self._progress_tracker.reset(env_ids, root_state[:, 0], self._commands[env_ids, 0], eligible)

    def _randomize_reference_camera(self):
        from isaaclab.utils.math import convert_camera_frame_orientation_convention

        position = torch.tensor(self.cfg.depth_camera.offset.pos, device=self.device).expand(self.num_envs, -1)
        self._camera_offset_pos = position + (torch.rand_like(position) * 2 - 1) * 0.01
        angle = (torch.rand(self.num_envs, 3, device=self.device) * 2 - 1) * torch.pi / 180
        angle *= torch.tensor((2.0, 5.0, 2.0), device=self.device)
        angle[:, 1] += torch.tensor(self.cfg.depth_camera.offset.rot, device=self.device).new_tensor(
            20 * torch.pi / 180
        )
        optical = torch.tensor((0.5, -0.5, 0.5, -0.5), device=self.device).expand(self.num_envs, -1)
        self._camera_offset_quat = quat_mul(quat_from_euler_xyz(*angle.unbind(-1)), optical)
        if self._depth_camera is not None and self.cfg.depth_backend == "warp":
            ids = self._camera_env_ids
            if isinstance(self._depth_camera, WMPWarpDepthCamera):
                self._depth_camera.set_local_poses(self._camera_offset_pos[ids], self._camera_offset_quat[ids])
            else:
                self._depth_camera._offset_pos[ids] = self._camera_offset_pos[ids]
                self._depth_camera._offset_quat[ids] = convert_camera_frame_orientation_convention(
                    self._camera_offset_quat[ids], origin="ros", target="world"
                )

    def _update_terrain_curriculum(self, env_ids: torch.Tensor) -> None:
        """按所选版本的前进距离规则更新地形等级。"""
        if not hasattr(self._terrain, "terrain_levels"):
            return
        valid = self.episode_length_buf[env_ids] > 0
        curriculum_ids = env_ids[valid]
        if curriculum_ids.numel() == 0:
            return

        if self.cfg.curriculum_semantics == "upstream_v3":
            distance = (
                self._robot.data.root_pos_w[curriculum_ids, :2] - self._terrain.env_origins[curriculum_ids, :2]
            ).norm(dim=-1)
            move_up = distance > self.cfg.terrain.terrain_generator.size[0] / 2
            move_down = (
                distance < self._commands[curriculum_ids, :2].norm(dim=-1) * self.cfg.episode_length_s * 0.5
            ) & ~move_up
            levels = self._terrain.terrain_levels[curriculum_ids] + move_up.long() - move_down.long()
            maximum = self._terrain.max_terrain_level
            levels = torch.where(levels >= maximum, torch.randint_like(levels, maximum), levels.clamp_min(0))
            slots = getattr(self, "_terrain_curriculum_slots", None)
            if slots is not None:
                levels = slots.advance(curriculum_ids, levels)
            self._terrain.terrain_levels[curriculum_ids] = levels
            self._terrain.env_origins[curriculum_ids] = self._terrain.terrain_origins[
                self._terrain.terrain_levels[curriculum_ids], self._terrain.terrain_types[curriculum_ids]
            ]
            self._log_curriculum_metrics()
            return

        origin_xy = self._terrain.env_origins[curriculum_ids, :2]
        root_xy = self._robot.data.root_pos_w[curriculum_ids, :2]
        directional = self._directional_terrain_mask[curriculum_ids]
        course_progress = torch.where(
            directional,
            root_xy[:, 0] - origin_xy[:, 0],
            torch.norm(root_xy - origin_xy, dim=1),
        )
        travel_progress = torch.where(
            directional,
            root_xy[:, 0] - self._episode_start_pos[curriculum_ids, 0],
            torch.norm(root_xy - self._episode_start_pos[curriculum_ids], dim=1),
        )
        lateral_error = torch.abs(root_xy[:, 1] - origin_xy[:, 1])
        stayed_on_course = ~directional | (lateral_error < self.cfg.terrain_curriculum_max_lateral_error)
        success_distance = torch.full_like(course_progress, self.cfg.terrain_curriculum_success_distance)
        plum_mask = self._terrain_type_masks.get("wmp_plum_piles")
        if plum_mask is not None:
            success_distance[plum_mask[curriculum_ids]] = self.cfg.plum_curriculum_success_distance
        move_up = (course_progress > success_distance) & stayed_on_course
        elapsed = self.episode_length_buf[curriculum_ids] * self.step_dt
        commanded_distance = self._commands[curriculum_ids, :2].norm(dim=1) * elapsed
        move_down = (
            (commanded_distance > 0.1)
            & (travel_progress < commanded_distance * self.cfg.terrain_curriculum_down_fraction)
            & ~move_up
        )
        levels = self._terrain.terrain_levels[curriculum_ids]
        self._terrain.terrain_levels[curriculum_ids] = next_terrain_levels(
            levels, move_up, move_down, self._terrain.max_terrain_level
        )
        self._terrain.env_origins[curriculum_ids] = self._terrain.terrain_origins[
            self._terrain.terrain_levels[curriculum_ids], self._terrain.terrain_types[curriculum_ids]
        ]
        self._log_curriculum_metrics()

    def _log_curriculum_metrics(self) -> None:
        log = self.extras.setdefault("log", {})
        log.update({f"Curriculum/{key}": value for key, value in self.get_curriculum_metrics().items()})

    def get_curriculum_metrics(self) -> dict[str, torch.Tensor]:
        """Return current per-terrain levels for the text training logger."""
        if not hasattr(self._terrain, "terrain_levels"):
            return {}
        slots = getattr(self, "_terrain_curriculum_slots", None)
        levels = (self._terrain.terrain_levels if slots is None else slots.levels.flatten()).float()
        metrics = {"terrain_level_mean": levels.mean()}
        for name in self._terrain_type_masks:
            mask = self._terrain_type_masks.get(name)
            if slots is not None and mask is not None:
                mask = mask.repeat_interleave(slots.levels.shape[1])
            if mask is not None and torch.any(mask):
                metrics[f"terrain_level_{name.removeprefix('wmp_')}"] = levels[mask].mean()
        tracker = getattr(self, "_progress_tracker", None)
        if tracker is not None:
            count = tracker.active.sum().clamp_min(1)
            metrics["reward_progress/stalled_fraction"] = tracker.stalled.sum() / count
            metrics["reward_progress/frontier_speed"] = (tracker.speed * tracker.active).sum() / count
        return metrics

    def curriculum_state_dict(self):
        return {
            "command": self.command_curriculum.state_dict(),
            "terrain_levels": self._terrain.terrain_levels.clone(),
            "terrain_slots": (
                None if self._terrain_curriculum_slots is None else self._terrain_curriculum_slots.state_dict()
            ),
            "terrain_types": self._terrain.terrain_types.clone(),
            "terrain_names": self.terrain_names,
            "reward_scale": self._reward_curriculum_scale,
            "config": {
                "control_revision": self.cfg.control_revision,
                "reward_revision": getattr(self.cfg, "reward_revision", "upstream_v3"),
                "camera_implementation": self.cfg.camera_implementation,
                "num_envs": self.num_envs,
                "num_rows": self.cfg.terrain.terrain_generator.num_rows,
                "num_cols": self.cfg.terrain.terrain_generator.num_cols,
                "logical_camera_envs": self.cfg.logical_camera_envs,
                "seed": self.cfg.seed,
                "terrain_proportions": {
                    name: cfg.proportion for name, cfg in self.cfg.terrain.terrain_generator.sub_terrains.items()
                },
            },
        }

    def load_curriculum_state_dict(self, state):
        saved_reward = state.get("config", {}).get("reward_revision", "upstream_v3")
        if saved_reward != getattr(self.cfg, "reward_revision", "upstream_v3"):
            raise ValueError("checkpoint reward revision differs; start a separate experiment")
        self.command_curriculum.load_state_dict(state["command"])
        if state.get("terrain_names") == self.terrain_names:
            self._terrain.terrain_levels.copy_(state["terrain_levels"].clamp(0, self._terrain.max_terrain_level - 1))
            if self._terrain_curriculum_slots is not None:
                if state.get("terrain_slots") is None:
                    raise ValueError("checkpoint has no aligned terrain curriculum slots")
                self._terrain_curriculum_slots.load_state_dict(state["terrain_slots"])
                if not torch.equal(self._terrain_curriculum_slots.active_levels, self._terrain.terrain_levels):
                    raise ValueError("active terrain level differs from saved curriculum slot")
            self._terrain.env_origins[:] = self._terrain.terrain_origins[
                self._terrain.terrain_levels, self._terrain.terrain_types
            ]
        self._reward_curriculum_scale = state["reward_scale"]
        if getattr(self, "_progress_tracker", None) is not None:
            self._progress_tracker = ForwardProgressGate(
                self.num_envs, self.step_dt, self.device,
                self.cfg.progress_window_s, self.cfg.progress_min_speed,
            )


WMPGo2Env = WMPEnv
