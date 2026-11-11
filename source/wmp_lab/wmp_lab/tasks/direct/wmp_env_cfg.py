# SPDX-License-Identifier: Apache-2.0
"""IsaacLab 2.3.2 configuration for the WMP Go2 direct environment."""

import math
from wmp_lab.learning.progress_rewards import (
    PROGRESS_REWARD_REVISION, REFERENCE_REWARD_REVISION, REWARD_REVISIONS,
)
from pathlib import Path

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCameraCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from wmp_lab.robots import (
    CAMERA_REVISION_NATIVE,
    COLLISION_REVISION_NATIVE,
    COLLISION_REVISIONS,
    CONTROL_REVISION,
    ENVIRONMENT_REVISION,
    LEGACY_CONTROL_REVISION,
    UNITREE_GO2_REVISIONS,
    default_control_revision,
    robot_profile,
)

from .actuators import UnitreeGo2ActuatorCfg, UpstreamPDActuatorCfg
from .reference_events import randomize_reference_material
from .reference_terrain import REFERENCE_TERRAINS, REFERENCE_WEIGHTS, ReferenceSubTerrainCfg, ReferenceTerrainImporter
from .terrains import (
    WmpCrawlTerrainCfg,
    WmpObstacleTerrainCfg,
    WmpPlumPilesTerrainCfg,
    WmpTiltTerrainCfg,
    wmp_crawl_terrain,
    wmp_obstacle_terrain,
    wmp_plum_piles_terrain,
    wmp_rough_slope_terrain,
    wmp_tilt_terrain,
)

ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets"
GO2_USD = ASSET_ROOT / "robots" / "go2" / "go2.usd"

# Unitree's official Go2 description defines ``front_camera_joint`` at this
# pose relative to ``base``.  A ROS optical camera looks along +Z, so the
# quaternion below maps camera +Z to Go2 +X, +X to -Y, and +Y to -Z.
GO2_FRONT_CAMERA_POS = (0.32715, -0.00003, 0.04297)
GO2_FRONT_CAMERA_ROT_ROS = (0.5, -0.5, 0.5, -0.5)

WMP_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    # More rows make the jump between adjacent Go2 difficulty levels small
    # enough to learn wide gaps instead of creating a cliff around level 5.
    num_rows=16,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    curriculum=True,
    use_cache=False,
    sub_terrains={
        "rough_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(-0.4, 0.4), platform_width=3.0, border_width=0.25
        ),
        "stairs_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.32,
            platform_width=3.0,
            border_width=1.0,
        ),
        "stairs_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.23),
            step_width=0.32,
            platform_width=3.0,
            border_width=1.0,
        ),
        "wmp_gap": terrain_gen.MeshGapTerrainCfg(proportion=0.25, gap_width_range=(0.05, 1.0), platform_width=4.0),
        "wmp_climb": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.20, box_height_range=(0.05, 0.6), platform_width=4.0, double_box=True
        ),
        "wmp_tilt": WmpTiltTerrainCfg(
            function=wmp_tilt_terrain,
            proportion=0.05,
            # Go2's zero-abduction outer foot span is about 0.328 m; the
            # inherited A1 0.28--0.32 m corridor leaves no nominal clearance.
            corridor_width_range=(0.34, 0.42),
            obstacle_length_range=(0.4, 0.8),
        ),
        "wmp_crawl": WmpCrawlTerrainCfg(
            function=wmp_crawl_terrain,
            proportion=0.05,
            clearance_range=(0.20, 0.35),
            bar_length_range=(0.2, 0.4),
        ),
        "wmp_plum_piles": WmpPlumPilesTerrainCfg(
            function=wmp_plum_piles_terrain,
            proportion=0.05,
            pile_diameter_range=(0.48, 0.26),
            pile_gap_range=(0.12, 0.30),
            pit_depth_range=(0.25, 0.60),
            height_variation_range=(0.0, 0.10),
        ),
        "rough_flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.05, noise_range=(-0.05, 0.05), noise_step=0.005, border_width=0.25
        ),
    },
)

SMOKE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=2.0,
    num_rows=1,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    curriculum=True,
    use_cache=False,
    # One column per sensor-critical terrain exercises the custom meshes and
    # stratified real-camera assignment in the real simulator smoke.
    sub_terrains={
        "wmp_tilt": WmpTiltTerrainCfg(
            function=wmp_tilt_terrain,
            proportion=0.25,
            corridor_width_range=(0.34, 0.42),
            obstacle_length_range=(0.4, 0.8),
        ),
        "wmp_crawl": WmpCrawlTerrainCfg(
            function=wmp_crawl_terrain,
            proportion=0.25,
            clearance_range=(0.20, 0.35),
            bar_length_range=(0.2, 0.4),
        ),
        "wmp_plum_piles": WmpPlumPilesTerrainCfg(
            function=wmp_plum_piles_terrain,
            proportion=0.25,
        ),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
    },
)

# USD 基础模板；apply_robot_profile 根据控制版本替换执行器，DC 仅用于历史兼容。
GO2_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(GO2_USD),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        )
    },
)


@configclass
class WMPEventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 2.0),
            "dynamic_friction_range": (0.5, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.0, 3.0),
            "operation": "add",
        },
    )
    link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="^(?!base$).*$"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(15.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


@configclass
class WMPEnvCfg(DirectRLEnvCfg):
    environment_revision = ENVIRONMENT_REVISION
    seed = 1
    randomize_initial_episode_length = True
    control_revision = None

    def __post_init__(self):
        apply_robot_profile(self)

    robot_name = "go1"
    reward_semantics = "upstream_v3"
    reward_revision = REFERENCE_REWARD_REVISION
    progress_window_s = 2.0
    progress_min_speed = 0.1
    progress_reward_terrains = ("wmp_gap", "wmp_continuous_gap", "wmp_bream", "wmp_climb")
    curriculum_semantics = "upstream_v3"
    enable_command_curriculum = True
    zero_command_probability = 0.05
    enable_observation_noise = False
    depth_backend = "warp"
    camera_implementation = "wmp_go1"
    camera_revision = CAMERA_REVISION_NATIVE
    collision_revision = COLLISION_REVISION_NATIVE
    camera_delay_frames = 1
    randomize_camera = True
    camera_mandatory_terrains = ("wmp_tilt", "wmp_stool", "wmp_crawl")
    episode_length_s = 20.0
    decimation = 4
    action_space = 12
    observation_space = 285
    state_space = 0
    action_scale = 0.25
    logical_camera_envs = 1100
    camera_update_interval = 5
    command_resampling_s = 10.0
    command_x_range = (0.0, 0.8)
    command_y_range = (0.0, 0.0)
    command_yaw_range = (-1.0, 1.0)
    enable_domain_randomization = True
    enable_curriculum = True
    terrain_curriculum_success_distance = 3.0
    plum_curriculum_success_distance = 3.4
    terrain_curriculum_max_lateral_error = 1.0
    terrain_curriculum_down_fraction = 0.5
    heading_command_gain = 2.0
    heading_yaw_limit = math.pi
    jump_command_x_range = (0.45, 0.8)
    tilt_command_x_range = (0.30, 0.8)
    terrain_progress_reward_scale = 0.5
    stuck_penalty_scale = 1.0
    jump_vertical_velocity_penalty_scale = 0.25
    camera_priority_terrains = ("wmp_tilt", "wmp_crawl", "wmp_plum_piles")

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            min_position_iteration_count=4,
            max_position_iteration_count=4,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.5,
            # 原版自碰撞与4100环境需要更大的接触栈，避免 PhysX 丢弃接触。
            gpu_collision_stack_size=2**27,
            gpu_max_rigid_patch_count=2**20,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    events: WMPEventCfg | None = WMPEventCfg()
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=WMP_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = GO2_CFG
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )
    body_height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0), ordering="yx"),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    forward_height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(1.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(2.0, 2.4), ordering="yx"),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    depth_camera: RayCasterCameraCfg = RayCasterCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        update_period=0.1,
        data_types=["distance_to_image_plane"],
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=GO2_FRONT_CAMERA_POS,
            rot=GO2_FRONT_CAMERA_ROT_ROS,
            convention="ros",
        ),
        depth_clipping_behavior="max",
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=17.0,
            horizontal_aperture=20.955,
            height=64,
            width=64,
        ),
        mesh_prim_paths=["/World/ground"],
        max_distance=2.0,
    )


@configclass
class WMPSmokeEnvCfg(WMPEnvCfg):
    seed = 7
    reference_terrain_selection = "smoke"
    episode_length_s = 2.0
    logical_camera_envs = 3
    enable_domain_randomization = False
    enable_observation_noise = False
    randomize_camera = False
    enable_command_curriculum = False
    enable_curriculum = False
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=3.0, replicate_physics=True)
    events = None
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SMOKE_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    def __post_init__(self):
        apply_robot_profile(self)


_LEGACY_TERRAINS_CFG = WMP_TERRAINS_CFG.copy()

TERRAIN_NAMES = (
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


def legacy_terrain_config(robot="go2", terrain=None):
    """默认保留上游地形比例；all 启用全部类型，单类型用于评估。"""
    profile = robot_profile(robot)
    generator = _LEGACY_TERRAINS_CFG.copy()
    generator.num_cols = 40 if terrain is None else (36 if terrain == "all" else 1)
    mix = generator.sub_terrains
    mix["rough_slope"].function = wmp_rough_slope_terrain
    mix["rough_slope"].slope_range = (0.0, 0.45)
    mix["wave"] = terrain_gen.HfWaveTerrainCfg(proportion=0.0, amplitude_range=(0.1, 0.28), num_waves=5)
    mix["discrete"] = terrain_gen.HfDiscreteObstaclesTerrainCfg(
        proportion=0.05,
        obstacle_height_mode="choice",
        obstacle_width_range=(1.0, 1.8),
        obstacle_height_range=(0.05, 0.24),
        num_obstacles=20,
        platform_width=3.0,
    )
    for name in (
        "gap",
        "continuous_gap",
        "bream",
        "stairs_cliff",
        "slope_cliff",
        "stepping_stones",
        "stepping_one_bridge",
        "climb",
        "stool",
    ):
        mix[f"wmp_{name}"] = WmpObstacleTerrainCfg(function=wmp_obstacle_terrain, kind=name, proportion=0.05)
    mix["wmp_tilt"].corridor_width_range = profile.tilt_width_range
    mix["wmp_crawl"].clearance_range = (0.27, 0.42)
    defaults = (0, 0.1, 0.1, 0.1, 0.05, 0.1, 0.05, 0.05, 0.1, 0.05, 0, 0.05, 0.1, 0, 0.05, 0.05, 0.05, 0.05)
    ordered = {}
    for name, weight in zip(TERRAIN_NAMES, defaults, strict=True):
        key = (
            name
            if name in {"wave", "rough_slope", "stairs_up", "stairs_down", "discrete", "rough_flat"}
            else f"wmp_{name}"
        )
        cfg = mix[key]
        cfg.proportion = (1.0 if terrain == "all" else float(name == terrain)) if terrain else weight
        ordered[key] = cfg
    if terrain is not None and terrain != "all" and terrain not in TERRAIN_NAMES:
        raise ValueError(f"unknown terrain {terrain!r}")
    total = sum(cfg.proportion for cfg in ordered.values())
    for cfg in ordered.values():
        cfg.proportion /= total
    generator.sub_terrains = ordered
    return generator


def make_env_cfg(
    robot="go1",
    *,
    smoke=False,
    terrain=None,
    num_envs=None,
    evaluation=False,
    control_revision=None,
    environment_revision=ENVIRONMENT_REVISION,
    camera_implementation=None,
    camera_revision=None,
    collision_revision=None,
    reward_revision=None,
):
    if control_revision is None:
        control_revision = (
            default_control_revision(robot) if environment_revision == ENVIRONMENT_REVISION else CONTROL_REVISION
        )
    profile = robot_profile(
        robot, control_revision=control_revision, environment_revision=environment_revision,
        camera_revision=camera_revision,
    )
    cfg = WMPSmokeEnvCfg() if smoke else WMPEnvCfg()
    cfg.robot_name = profile.name
    cfg.environment_revision = environment_revision
    cfg.control_revision = control_revision
    cfg.camera_revision = profile.camera_revision
    cfg.collision_revision = COLLISION_REVISION_NATIVE if collision_revision is None else collision_revision
    if cfg.collision_revision not in COLLISION_REVISIONS:
        raise ValueError(f"unsupported collision revision: {cfg.collision_revision}")
    cfg.reward_revision = (
        PROGRESS_REWARD_REVISION if profile.name == "go2" and environment_revision == ENVIRONMENT_REVISION
        else REFERENCE_REWARD_REVISION
    ) if reward_revision is None else reward_revision
    if cfg.reward_revision not in REWARD_REVISIONS:
        raise ValueError(f"unsupported reward revision: {cfg.reward_revision}")
    if cfg.reward_revision == PROGRESS_REWARD_REVISION and environment_revision != ENVIRONMENT_REVISION:
        raise ValueError("progress rewards require the reference environment")
    apply_robot_profile(cfg)
    if not smoke or terrain is not None:
        cfg.terrain.terrain_generator = (
            terrain_config(robot, terrain)
            if environment_revision == ENVIRONMENT_REVISION
            else legacy_terrain_config(robot, terrain)
        )
    elif environment_revision != ENVIRONMENT_REVISION:
        cfg.terrain.terrain_generator.sub_terrains["wmp_tilt"].corridor_width_range = profile.tilt_width_range
    if environment_revision == ENVIRONMENT_REVISION:
        if smoke and terrain is None:
            cfg.terrain.terrain_generator = terrain_config(robot, "smoke")
        cfg.terrain.class_type = ReferenceTerrainImporter
        cfg.robot.spawn.articulation_props.enabled_self_collisions = True
        cfg.robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0)
        if cfg.events is not None:
            cfg.events.physics_material = EventTerm(func=randomize_reference_material, mode="startup")
        cfg.sim.physics_material.friction_combine_mode = "average"
        cfg.terrain.physics_material.friction_combine_mode = "average"
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    if camera_implementation is not None:
        if camera_implementation not in ("wmp_go1", "isaaclab_sparse"):
            raise ValueError(f"unsupported camera implementation: {camera_implementation}")
        cfg.camera_implementation = camera_implementation
    elif environment_revision != ENVIRONMENT_REVISION:
        cfg.camera_implementation = "isaaclab_sparse"
    if evaluation:
        cfg.randomize_initial_episode_length = False
        cfg.enable_curriculum = cfg.enable_command_curriculum = False
        cfg.enable_domain_randomization = cfg.enable_observation_noise = cfg.randomize_camera = False
        cfg.events = None
        cfg.logical_camera_envs = cfg.scene.num_envs
    return cfg


def apply_robot_profile(cfg):
    if cfg.control_revision is None:
        cfg.control_revision = (
            default_control_revision(cfg.robot_name)
            if cfg.environment_revision == ENVIRONMENT_REVISION else CONTROL_REVISION
        )
    profile = robot_profile(
        cfg.robot_name, control_revision=cfg.control_revision, environment_revision=cfg.environment_revision,
        camera_revision=getattr(cfg, "camera_revision", CAMERA_REVISION_NATIVE),
    )
    legacy = cfg.control_revision == LEGACY_CONTROL_REVISION
    cfg.robot = GO2_CFG.copy()
    cfg.robot.spawn.usd_path = str(profile.usd_path)
    cfg.robot.init_state.pos = (0, 0, profile.spawn_height)
    cfg.robot.init_state.joint_pos = dict(zip(profile.joint_order, profile.default_joint_pos, strict=True))
    cfg.action_scale = profile.action_scale
    cfg.robot.spawn.articulation_props.solver_position_iteration_count = profile.solver_position_iterations
    cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = profile.solver_velocity_iterations
    cfg.sim.physx.min_position_iteration_count = profile.solver_position_iterations
    cfg.sim.physx.max_position_iteration_count = profile.solver_position_iterations
    cfg.sim.physx.min_velocity_iteration_count = profile.solver_velocity_iterations
    cfg.sim.physx.max_velocity_iteration_count = profile.solver_velocity_iterations
    if legacy:
        actuator = cfg.robot.actuators["legs"]
        actuator.effort_limit = actuator.saturation_effort = profile.torque_limits[0]
        actuator.velocity_limit = profile.velocity_limits[0]
    else:
        actuator_type = (
            UnitreeGo2ActuatorCfg if cfg.control_revision in UNITREE_GO2_REVISIONS else UpstreamPDActuatorCfg
        )
        actuator_options = dict(
            joint_names_expr=list(profile.joint_order),
            effort_limit=dict(zip(profile.joint_order, profile.torque_limits, strict=True)),
            velocity_limit=dict(zip(profile.joint_order, profile.velocity_limits, strict=True)),
            velocity_limit_sim=dict(zip(profile.joint_order, profile.velocity_limits, strict=True)),
            stiffness=profile.stiffness,
            damping=profile.damping,
            friction=profile.friction,
        )
        if cfg.control_revision in UNITREE_GO2_REVISIONS:
            actuator_options["gear_ratio"] = dict(zip(profile.joint_order, profile.joint_gear_ratios, strict=True))
        cfg.robot.actuators["legs"] = actuator_type(**actuator_options)
    if cfg.environment_revision == ENVIRONMENT_REVISION:
        cfg.terrain.terrain_generator = terrain_config(
            cfg.robot_name, getattr(cfg, "reference_terrain_selection", None)
        )
        cfg.terrain.class_type = ReferenceTerrainImporter
        cfg.robot.spawn.articulation_props.enabled_self_collisions = True
        cfg.robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0)
        cfg.sim.physics_material.friction_combine_mode = "average"
        cfg.terrain.physics_material.friction_combine_mode = "average"
        if cfg.events is not None:
            cfg.events.physics_material = EventTerm(func=randomize_reference_material, mode="startup")
    else:
        cfg.body_height_scanner.pattern_cfg.ordering = "xy"
        cfg.forward_height_scanner.pattern_cfg.ordering = "xy"
    cfg.depth_camera.offset.pos = profile.camera_position
    angle = math.radians(profile.camera_pitch_deg) / 2
    c, s = math.cos(angle), math.sin(angle)
    cfg.depth_camera.offset.rot = ((c - s) / 2, -(c + s) / 2, (c + s) / 2, -(c - s) / 2)
    cfg.depth_camera.pattern_cfg.horizontal_aperture = 2 * 17 * math.tan(math.radians(58) / 2)
    tilt = cfg.terrain.terrain_generator.sub_terrains.get("wmp_tilt")
    if tilt is not None:
        tilt.corridor_width_range = profile.tilt_width_range


@configclass
class WMPGo1RoughEnvCfg(WMPEnvCfg):
    robot_name = "go1"

    def __post_init__(self):
        self.terrain.terrain_generator = terrain_config("go1")
        apply_robot_profile(self)


@configclass
class WMPGo2RoughEnvCfg(WMPEnvCfg):
    robot_name = "go2"

    def __post_init__(self):
        self.terrain.terrain_generator = terrain_config("go2")
        apply_robot_profile(self)


@configclass
class WMPGo1RoughSmokeEnvCfg(WMPSmokeEnvCfg):
    robot_name = "go1"

    def __post_init__(self):
        apply_robot_profile(self)


class WMPGo2RoughSmokeEnvCfg(WMPSmokeEnvCfg):
    robot_name = "go2"


def terrain_config(robot="go1", terrain=None):
    """固定 Go1 原17类比例；all/梅花桩为显式扩展，smoke只缩小地图。"""
    if terrain not in (None, "all", "smoke", *REFERENCE_TERRAINS):
        raise ValueError(f"unknown terrain {terrain!r}")
    names = REFERENCE_TERRAINS if terrain == "all" else REFERENCE_TERRAINS[:17]
    if terrain == "plum_piles":
        names = REFERENCE_TERRAINS
    weights = dict(zip(REFERENCE_TERRAINS, REFERENCE_WEIGHTS, strict=True))
    if terrain == "smoke":
        names = ("tilt", "crawl", "plum_piles", "rough_flat")
    chosen = {
        name: (1.0 if terrain in ("all", "smoke") else float(name == terrain) if terrain else weights[name])
        for name in names
    }
    total = sum(chosen.values())
    sub = {}
    for name, weight in chosen.items():
        key = (
            name
            if name in {"wave", "rough_slope", "stairs_up", "stairs_down", "discrete", "rough_flat"}
            else f"wmp_{name}"
        )
        sub[key] = ReferenceSubTerrainCfg(
            kind=name, proportion=weight / total, tilt_width_range=robot_profile(robot).tilt_width_range
        )
    return TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=25.0,
        num_rows=1 if terrain == "smoke" else 10,
        num_cols=4 if terrain == "smoke" else 18 if terrain == "all" else 20 if terrain is None else 1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        curriculum=True,
        use_cache=False,
        sub_terrains=sub,
    )


# 公开默认值与训练工厂一致，历史配置单独保留。
WMP_TERRAINS_CFG = terrain_config("go1")
