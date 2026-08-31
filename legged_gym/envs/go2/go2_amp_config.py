from legged_gym.envs.a1.a1_amp_config import A1AMPCfg, A1AMPCfgPPO
from legged_gym.envs.go2.go2_config import GO2_DEFAULT_JOINT_ANGLES

GO2_FRONT_CAMERA_POSITION = [0.32715, -0.00003, 0.04297]


class Go2AMPCfg(A1AMPCfg):
    """WMP/AMP configuration adapted to the Unitree Go2 asset."""

    class env(A1AMPCfg.env):
        # The bundled motions use [FR, FL, RR, RL], while the Go2 URDF uses
        # [FL, FR, RL, RR]. Keep the conversion local to the Go2 task.
        amp_reorder_from_pybullet_to_isaac = True

    class init_state(A1AMPCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = GO2_DEFAULT_JOINT_ANGLES

    class control(A1AMPCfg.control):
        control_type = "P"
        stiffness = {"joint": 40.0}
        damping = {"joint": 1.0}
        action_scale = 0.25
        decimation = 4

    class terrain(A1AMPCfg.terrain):
        # Finer steps avoid the abrupt gap-width jump around the old level 5.
        num_rows = 16
        # [wave, slope, stairs down, stairs up, discrete, gap, climb,
        #  tilt, crawl, plum piles, rough flat]
        terrain_proportions = [0.0, 0.05, 0.15, 0.15, 0.0, 0.25, 0.20, 0.05, 0.05, 0.05, 0.05]
        gap_width_range = [0.0, 1.0]
        # Go2's zero-abduction outer foot span is about 0.328 m, so A1's
        # 0.28--0.32 m corridor has no nominal collision margin.
        tilt_width_range = [0.42, 0.34]
        signed_course_curriculum = True
        retain_max_terrain_level = True
        curriculum_success_distance = 3.0
        plum_curriculum_success_distance = 3.4
        curriculum_max_lateral_error = 1.0
        curriculum_down_fraction = 0.5
        plum_pile_diameter_range = [0.48, 0.26]
        plum_pile_gap_range = [0.12, 0.30]
        plum_pit_depth_range = [0.25, 0.60]
        plum_height_variation_range = [0.0, 0.10]

    class depth(A1AMPCfg.depth):
        # Official Unitree ``front_camera_joint`` pose relative to ``base``.
        position = GO2_FRONT_CAMERA_POSITION

    class asset(A1AMPCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        # Isaac Gym uses bit 0 to disable self-collision.  This matches the A1
        # WMP task and avoids overlapping Go2 fixed-link shapes fighting PhysX.
        self_collisions = 1

    class normalization(A1AMPCfg.normalization):
        base_height = 0.38

    class rewards(A1AMPCfg.rewards):
        base_height_target = 0.38
        foot_height_target = 0.15
        jump_vertical_velocity_penalty_scale = 0.25

        class scales(A1AMPCfg.rewards.scales):
            terrain_progress = 0.5

    class commands(A1AMPCfg.commands):
        jump_lin_vel_x = [0.45, 0.8]
        tilt_lin_vel_x = [0.30, 0.8]


class Go2AMPCfgPPO(A1AMPCfgPPO):
    class runner(A1AMPCfgPPO.runner):
        experiment_name = "go2_amp_example"
        amp_reorder_from_pybullet_to_isaac = True
