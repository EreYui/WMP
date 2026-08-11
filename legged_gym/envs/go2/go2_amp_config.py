from legged_gym.envs.a1.a1_amp_config import A1AMPCfg, A1AMPCfgPPO
from legged_gym.envs.go2.go2_config import GO2_DEFAULT_JOINT_ANGLES


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

    class depth(A1AMPCfg.depth):
        # Go2's nose extends farther forward than A1's trunk.
        position = [0.30, 0.0, 0.03]

    class asset(A1AMPCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2.urdf"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0

    class normalization(A1AMPCfg.normalization):
        base_height = 0.38

    class rewards(A1AMPCfg.rewards):
        base_height_target = 0.38
        foot_height_target = 0.15


class Go2AMPCfgPPO(A1AMPCfgPPO):
    class runner(A1AMPCfgPPO.runner):
        experiment_name = "go2_amp_example"
        amp_reorder_from_pybullet_to_isaac = True
