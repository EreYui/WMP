# SPDX-License-Identifier: Apache-2.0
"""Go1/Go2 的离线资产、相机和部署参数，不依赖模拟器导入。"""

from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .constants import GO2_JOINT_ORDER

ENVIRONMENT_REVISION = "go1_reference_v6"
LEGACY_ENVIRONMENT_REVISION = "legacy_v5"
CONTROL_REVISION = "upstream_pd_v1"
GO2_CONTROL_REVISION = "unitree_go2_v1"
GO2_JOINT_CONTROL_REVISION = "unitree_go2_joint_v2"
GO2_CALF_GEAR_RATIO = 45.43 / 23.7
# 诊断用：Go2 资产 + 按 Go1 gym 力矩/质量比（33.5 N·m / 11.81 kg）缩放到 16.09 kg 的全关节力矩。
GO2_CAPACITY_PROBE_REVISION = "go2_go1gym_capacity_probe_v1"
GO2_CAPACITY_PROBE_TORQUE = round(33.5 * 16.087 / 11.81, 2)
# 诊断用：保留 Go2 相机水平位置，仅把高度抬到与 Go1 相同的 0.12 m。
CAMERA_REVISION_NATIVE = "native_v1"
GO2_CAMERA_GO1_HEIGHT_REVISION = "go1_height_v1"
CAMERA_REVISIONS = (CAMERA_REVISION_NATIVE, GO2_CAMERA_GO1_HEIGHT_REVISION)
# 诊断用：关闭 Head_upper/Head_lower 前伸碰撞，机身盒与腿碰撞保留。
COLLISION_REVISION_NATIVE = "native_v1"
GO2_NO_HEAD_COLLISION_REVISION = "no_head_collision_v1"
COLLISION_REVISIONS = (COLLISION_REVISION_NATIVE, GO2_NO_HEAD_COLLISION_REVISION)
GO2_HEAD_COLLISION_BODIES = ("Head_upper", "Head_lower")
UNITREE_GO2_REVISIONS = (GO2_CONTROL_REVISION, GO2_JOINT_CONTROL_REVISION)
EXPLICIT_PD_REVISIONS = (CONTROL_REVISION, GO2_CAPACITY_PROBE_REVISION, *UNITREE_GO2_REVISIONS)
UNITREE_LAB_COMMIT = "4960b84732b0c2ec593dccbfe963fda1bcd7b1e3"
UNITREE_ROS_COMMIT = "3ae893d905d11795ddf1075825ee5010364a1109"

LEGACY_CONTROL_REVISION = "legacy_dc_v3"


def checkpoint_control_revision(payload):
    extra = payload.get("extra", {})
    environment = extra.get("environment") or {}
    return extra.get("control_revision", environment.get("config", {}).get("control_revision", LEGACY_CONTROL_REVISION))


@dataclass(frozen=True)
class RobotProfile:
    name: str
    camera_position: tuple[float, float, float]
    camera_pitch_deg: float
    torque_limits: tuple[float, ...]
    velocity_limits: tuple[float, ...]
    spawn_height: float
    tilt_width_range: tuple[float, float]
    asset_revision: str = ENVIRONMENT_REVISION
    joint_order: tuple[str, ...] = GO2_JOINT_ORDER
    default_joint_pos: tuple[float, ...] = (0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5)
    stiffness: float = 40.0
    damping: float = 1.0
    action_scale: float = 0.25
    control_revision: str = CONTROL_REVISION
    actuator_model: str = "fixed_pd"
    friction: float = 0.0
    solver_position_iterations: int = 4
    solver_velocity_iterations: int = 0
    torque_speed_curve: dict[str, float] | None = None
    asset_effort_limits: tuple[float, ...] | None = None
    joint_gear_ratios: tuple[float, ...] = (1.0,) * 12
    control_adaptation: str | None = None
    camera_revision: str = CAMERA_REVISION_NATIVE
    collision_revision: str = COLLISION_REVISION_NATIVE
    reference_source: str | None = None
    asset_source: str | None = None

    @property
    def usd_path(self) -> Path:
        root = Path(__file__).parent / "assets" / "robots" / self.name
        if self.asset_revision == ENVIRONMENT_REVISION:
            root = root / "reference"
        return root / f"{self.name}.usd"


ROBOTS = {
    "go1": RobotProfile("go1", (0.28, 0.0, 0.12), 20.0, (33.5,) * 12, (30.0,) * 12, 0.35, (0.28, 0.32)),
    "go2": RobotProfile(
        "go2", (0.32715, -0.00003, 0.04297), 20.0, (23.7, 23.7, 35.55) * 4, (30.1, 30.1, 20.07) * 4, 0.4, (0.34, 0.42)
    ),
}


def default_control_revision(name: str) -> str:
    name = name.lower()
    if name not in ROBOTS:
        raise ValueError(f"unsupported robot {name!r}; choose go1 or go2")
    return GO2_JOINT_CONTROL_REVISION if name == "go2" else CONTROL_REVISION


def apply_camera_revision(profile: RobotProfile, camera_revision=None) -> RobotProfile:
    """Keep native camera by default; optionally raise Go2 height to the Go1 value."""
    revision = CAMERA_REVISION_NATIVE if camera_revision is None else camera_revision
    if revision not in CAMERA_REVISIONS:
        raise ValueError(f"unsupported camera revision {revision!r}")
    if revision == CAMERA_REVISION_NATIVE:
        return replace(profile, camera_revision=revision)
    if profile.name != "go2":
        raise ValueError("go1_height camera revision is defined only for Go2")
    x, y, _ = profile.camera_position
    return replace(profile, camera_position=(x, y, 0.12), camera_revision=revision)


def robot_profile(
    name: str, *, legacy=False, environment_revision=ENVIRONMENT_REVISION, control_revision=None,
    camera_revision=None,
) -> RobotProfile:
    name = name.lower()
    if name not in ROBOTS:
        raise ValueError(f"unsupported robot {name!r}; choose go1 or go2")
    if legacy and control_revision not in (None, LEGACY_CONTROL_REVISION):
        raise ValueError("legacy flag and control revision disagree")
    revision = LEGACY_CONTROL_REVISION if legacy else control_revision
    if revision is None:
        revision = default_control_revision(name) if environment_revision == ENVIRONMENT_REVISION else CONTROL_REVISION
    profile = replace(ROBOTS[name], asset_revision=environment_revision, control_revision=revision)
    if revision == LEGACY_CONTROL_REVISION:
        profile = replace(
            profile, actuator_model="legacy_dc",
            torque_limits=((23.7 if name == "go1" else 23.5),) * 12, velocity_limits=(30.0,) * 12,
        )
    elif revision == CONTROL_REVISION:
        pass
    elif revision == GO2_CAPACITY_PROBE_REVISION:
        if name != "go2" or environment_revision != ENVIRONMENT_REVISION:
            raise ValueError("capacity probe is defined only for the Go2 reference environment")
        profile = replace(profile, torque_limits=(GO2_CAPACITY_PROBE_TORQUE,) * 12, velocity_limits=(30.0,) * 12)
    elif revision not in UNITREE_GO2_REVISIONS or name != "go2" or environment_revision != ENVIRONMENT_REVISION:
        raise ValueError(f"unsupported control revision {revision!r} for {name}/{environment_revision}")
    else:
        # v1 preserves the official direct joint-space use of the shared Go2HV curve.
        # v2 is a separately versioned asset-based transmission adaptation hypothesis.
        official = replace(
            profile,
            torque_limits=(23.4,) * 12,
            velocity_limits=(30.1, 30.1, 15.7) * 4,
            stiffness=25.0,
            damping=0.5,
            friction=0.01,
            actuator_model="unitree_go2hv",
            solver_position_iterations=8,
            solver_velocity_iterations=4,
            torque_speed_curve={"X1": 13.5, "X2": 30.0, "Y1": 20.2, "Y2": 23.4},
            asset_effort_limits=(23.7, 23.7, 45.43) * 4,
            reference_source=(
                f"https://github.com/unitreerobotics/unitree_rl_lab/blob/{UNITREE_LAB_COMMIT}/"
                "source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py"
            ),
            asset_source=(
                f"https://github.com/unitreerobotics/unitree_ros/blob/{UNITREE_ROS_COMMIT}/"
                "robots/go2_description/urdf/go2_description.urdf"
            ),
        )
        if revision == GO2_CONTROL_REVISION:
            profile = official
        else:
            gears = (1.0, 1.0, GO2_CALF_GEAR_RATIO) * 4
            profile = replace(
                official,
                actuator_model="unitree_go2hv_geared",
                torque_limits=tuple(23.4 * gear for gear in gears),
                joint_gear_ratios=gears,
                control_adaptation=(
                    "Asset-based calf transmission hypothesis, g=45.43/23.7. "
                    "Joint velocity times g and requested joint torque divided by g enter the original motor curve; "
                    "motor torque times g returns to the joint. Not a claim of a missing official code branch."
                ),
            )
    return apply_camera_revision(profile, camera_revision)


def robot_metadata(name, control_revision, environment_revision=ENVIRONMENT_REVISION, camera_revision=None):
    legacy = control_revision == LEGACY_CONTROL_REVISION
    profile = robot_profile(
        name, legacy=legacy, environment_revision=environment_revision, control_revision=control_revision,
        camera_revision=camera_revision,
    )
    result = asdict(profile)
    if legacy:
        # 历史导出的标量字段仍保留，逐关节字段供新版部署读取。
        result.update(torque_limit=profile.torque_limits[0], velocity_limit=profile.velocity_limits[0])
    return result
