# SPDX-License-Identifier: Apache-2.0
"""IsaacLab 2.3.2 configuration for the WMP Go2 direct environment."""

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

from .terrains import WmpCrawlTerrainCfg, WmpTiltTerrainCfg, wmp_crawl_terrain, wmp_tilt_terrain

ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets"
GO2_USD = ASSET_ROOT / "robots" / "go2" / "go2.usd"

WMP_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
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
            proportion=0.25, box_height_range=(0.05, 0.6), platform_width=4.0, double_box=True
        ),
        "wmp_tilt": WmpTiltTerrainCfg(
            function=wmp_tilt_terrain,
            proportion=0.05,
            corridor_width_range=(0.28, 0.32),
            obstacle_length_range=(0.4, 0.8),
        ),
        "wmp_crawl": WmpCrawlTerrainCfg(
            function=wmp_crawl_terrain,
            proportion=0.05,
            clearance_range=(0.20, 0.35),
            bar_length_range=(0.2, 0.4),
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
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    curriculum=False,
    use_cache=False,
    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
)

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
class WMPGo2RoughEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 4
    action_space = 12
    observation_space = 285
    state_space = 0
    action_scale = 0.25
    logical_camera_envs = 1024
    camera_update_interval = 5
    command_resampling_s = 10.0
    command_x_range = (0.0, 0.8)
    command_y_range = (0.0, 0.0)
    command_yaw_range = (-1.0, 1.0)
    enable_domain_randomization = True
    enable_curriculum = True

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
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(1.6, 1.0), ordering="xy"),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    forward_height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(1.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(2.0, 2.4), ordering="xy"),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    depth_camera: RayCasterCameraCfg = RayCasterCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        update_period=0.1,
        data_types=["distance_to_camera"],
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=(0.27, 0.0, 0.03),
            rot=(-0.4055798, 0.5792280, -0.5792280, -0.4055798),
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
class WMPGo2RoughSmokeEnvCfg(WMPGo2RoughEnvCfg):
    seed = 7
    episode_length_s = 2.0
    logical_camera_envs = 2
    enable_domain_randomization = False
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
