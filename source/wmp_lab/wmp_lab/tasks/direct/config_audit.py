# SPDX-License-Identifier: Apache-2.0
"""枚举两种机器人的有效配置差异，防止额外训练参数悄悄分叉。"""

from dataclasses import asdict, replace

from wmp_lab.learning.config import PAPER_CFG
from wmp_lab.learning.contracts import canonical

from .wmp_env_cfg import make_env_cfg


def differences(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            result += differences(left.get(key), right.get(key), f"{path}.{key}".lstrip("."))
        return result
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [
            item
            for index, (a, b) in enumerate(zip(left, right, strict=True))
            for item in differences(a, b, f"{path}.{index}")
        ]
    return [] if left == right else [dict(path=path, go1=left, go2=right)]


def audit_robot_configs():
    configs = {robot: canonical(make_env_cfg(robot)) for robot in ("go1", "go2")}
    delta = differences(configs["go1"], configs["go2"])
    exact = {
        "robot_name",
        "robot.spawn.usd_path",
        "robot.init_state.pos.2",
        *(f"depth_camera.offset.pos.{i}" for i in range(3)),
    }
    prefixes = tuple(
        f"robot.actuators.legs.{field}." for field in ("effort_limit", "velocity_limit", "velocity_limit_sim")
    )
    unexpected = [
        row
        for row in delta
        if row["path"] not in exact
        and not row["path"].startswith(prefixes)
        and not (
            row["path"].startswith("terrain.terrain_generator.sub_terrains.") and ".tilt_width_range." in row["path"]
        )
    ]
    assert not unexpected, unexpected
    learning = {robot: asdict(replace(PAPER_CFG, robot=robot)) for robot in ("go1", "go2")}
    assert [row["path"] for row in differences(learning["go1"], learning["go2"])] == ["robot"]
    return dict(passed=True, environment_differences=delta, environment_configs=configs, learning_configs=learning)
