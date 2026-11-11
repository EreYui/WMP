# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

pytest.importorskip("carb", reason="terrain configs require an Isaac Sim Python launch")

from wmp_lab.tasks.direct.go2_env_cfg import (  # noqa: E402
    GO2_FRONT_CAMERA_POS,
    GO2_FRONT_CAMERA_ROT_ROS,
    WMP_TERRAINS_CFG,
)
from wmp_lab.tasks.direct.terrains import (  # noqa: E402
    WmpPlumPilesTerrainCfg,
    WmpTiltTerrainCfg,
    next_terrain_levels,
    wmp_plum_piles_terrain,
    wmp_tilt_terrain,
)


def _rotation_matrix(quaternion: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
    )


def test_go2_camera_uses_official_pose_and_faces_forward():
    np.testing.assert_allclose(GO2_FRONT_CAMERA_POS, (0.32715, -0.00003, 0.04297), atol=0.0)
    rotation = _rotation_matrix(GO2_FRONT_CAMERA_ROT_ROS)
    # ROS optical-frame +Z is forward and +X is image-right.
    np.testing.assert_allclose(rotation @ (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), atol=1.0e-7)
    np.testing.assert_allclose(rotation @ (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), atol=1.0e-7)


def test_reference_terrain_mix_keeps_upstream_defaults():
    assert WMP_TERRAINS_CFG.num_rows == 10
    assert "wmp_plum_piles" not in WMP_TERRAINS_CFG.sub_terrains
    total = sum(cfg.proportion for cfg in WMP_TERRAINS_CFG.sub_terrains.values())
    assert np.isclose(total, 1.0)


def test_tilt_corridor_keeps_go2_collision_margin():
    cfg = WmpTiltTerrainCfg(
        function=wmp_tilt_terrain,
        size=(8.0, 8.0),
        corridor_width_range=(0.34, 0.42),
        obstacle_length_range=(0.4, 0.8),
    )
    easy_meshes, _ = wmp_tilt_terrain(0.0, cfg)
    hard_meshes, _ = wmp_tilt_terrain(1.0, cfg)

    def corridor_width(meshes) -> float:
        negative_wall, positive_wall = meshes[1:3]
        return positive_wall.bounds[0, 1] - negative_wall.bounds[1, 1]

    assert np.isclose(corridor_width(easy_meshes), 0.42)
    assert np.isclose(corridor_width(hard_meshes), 0.34)


def test_plum_pile_difficulty_changes_geometry_without_removing_safe_platforms():
    cfg = WmpPlumPilesTerrainCfg(function=wmp_plum_piles_terrain, size=(8.0, 8.0))
    easy_meshes, easy_origin = wmp_plum_piles_terrain(0.0, cfg)
    hard_meshes, hard_origin = wmp_plum_piles_terrain(1.0, cfg)

    np.testing.assert_allclose(easy_origin, (4.0, 4.0, 0.0))
    np.testing.assert_allclose(hard_origin, easy_origin)
    assert len(easy_meshes) > 3
    assert len(hard_meshes) > 3
    assert np.isclose(easy_meshes[0].bounds[1, 2], -0.25)
    assert np.isclose(hard_meshes[0].bounds[1, 2], -0.60)
    assert np.isclose(easy_meshes[1].bounds[1, 2], 0.0)
    assert np.isclose(hard_meshes[2].bounds[1, 2], 0.0)
    assert np.isclose(easy_meshes[3].extents[0], 0.48)
    assert np.isclose(hard_meshes[3].extents[0], 0.26)


def test_curriculum_clamps_at_hardest_level_instead_of_random_recycling():
    levels = torch.tensor([0, 5, 15, 15])
    move_up = torch.tensor([False, True, True, False])
    move_down = torch.tensor([True, False, False, True])
    actual = next_terrain_levels(levels, move_up, move_down, max_terrain_level=16)
    torch.testing.assert_close(actual, torch.tensor([0, 6, 15, 14]))
