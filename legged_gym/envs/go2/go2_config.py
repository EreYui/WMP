from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


GO2_DEFAULT_JOINT_ANGLES = {
    "FL_hip_joint": 0.1,
    "RL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "FR_thigh_joint": 0.8,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,
    "RL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RR_calf_joint": -1.5,
}


class Go2RoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 5480
        include_history_steps = None
        num_observations = 235
        num_privileged_obs = 235
        reference_state_initialization = False

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = GO2_DEFAULT_JOINT_ANGLES

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"joint": 20.0}
        damping = {"joint": 0.5}
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1

    class normalization(LeggedRobotCfg.normalization):
        base_height = 0.38

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.38

        class scales(LeggedRobotCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0


class Go2RoughCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_go2"
