# SPDX-License-Identifier: MIT
"""WMP obstacle terrain primitives adapted to IsaacLab's terrain generator API."""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import torch
import trimesh
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


@configclass
class WmpTiltTerrainCfg(SubTerrainBaseCfg):
    function = None
    corridor_width_range: tuple[float, float] = MISSING
    obstacle_length_range: tuple[float, float] = (0.4, 0.8)


@configclass
class WmpCrawlTerrainCfg(SubTerrainBaseCfg):
    function = None
    clearance_range: tuple[float, float] = MISSING
    bar_length_range: tuple[float, float] = (0.2, 0.4)


@configclass
class WmpPlumPilesTerrainCfg(SubTerrainBaseCfg):
    """Difficulty curriculum for staggered circular stepping piles."""

    function = None
    pile_diameter_range: tuple[float, float] = (0.48, 0.26)
    pile_gap_range: tuple[float, float] = (0.12, 0.30)
    pit_depth_range: tuple[float, float] = (0.25, 0.60)
    height_variation_range: tuple[float, float] = (0.0, 0.10)
    approach_length: float = 5.0
    landing_length: float = 0.8


def _ground(size: tuple[float, float]) -> trimesh.Trimesh:
    return trimesh.creation.box(
        (size[0], size[1], 1.0),
        trimesh.transformations.translation_matrix((size[0] / 2.0, size[1] / 2.0, -0.5)),
    )


def wmp_tilt_terrain(difficulty: float, cfg: WmpTiltTerrainCfg):
    """Create WMP's narrow alternating corridor obstacle."""
    corridor = cfg.corridor_width_range[1] - difficulty * (cfg.corridor_width_range[1] - cfg.corridor_width_range[0])
    length = cfg.obstacle_length_range[0] + difficulty * (cfg.obstacle_length_range[1] - cfg.obstacle_length_range[0])
    center_x, center_y = cfg.size[0] / 2.0, cfg.size[1] / 2.0
    side_width = (cfg.size[1] - corridor) / 2.0
    meshes = [_ground(cfg.size)]
    for x_offset in (-2.0, 2.0):
        for y_sign in (-1.0, 1.0):
            y = center_y + y_sign * (corridor / 2.0 + side_width / 2.0)
            meshes.append(
                trimesh.creation.box(
                    (length, side_width, 1.0),
                    trimesh.transformations.translation_matrix((center_x + x_offset, y, 0.5)),
                )
            )
    return meshes, np.asarray((center_x, center_y, 0.0))


def wmp_crawl_terrain(difficulty: float, cfg: WmpCrawlTerrainCfg):
    """Create WMP's overhead crawl bars with difficulty-dependent clearance."""
    clearance = cfg.clearance_range[1] - difficulty * (cfg.clearance_range[1] - cfg.clearance_range[0])
    length = cfg.bar_length_range[0] + difficulty * (cfg.bar_length_range[1] - cfg.bar_length_range[0])
    center_x, center_y = cfg.size[0] / 2.0, cfg.size[1] / 2.0
    meshes = [_ground(cfg.size)]
    for x_offset in (-2.0, 2.0):
        meshes.append(
            trimesh.creation.box(
                (length, cfg.size[1], 1.0),
                trimesh.transformations.translation_matrix((center_x + x_offset, center_y, clearance + 0.5)),
            )
        )
    return meshes, np.asarray((center_x, center_y, 0.0))


def wmp_plum_piles_terrain(difficulty: float, cfg: WmpPlumPilesTerrainCfg):
    """Create a staggered field of round piles over a difficulty-scaled pit.

    The robot starts on the approach platform around the terrain origin and must
    traverse the piles in positive x before reaching the landing platform.  The
    pile diameter shrinks while the inter-pile gap, pit depth, and height
    variation grow with difficulty.
    """
    diameter = cfg.pile_diameter_range[0] + difficulty * (cfg.pile_diameter_range[1] - cfg.pile_diameter_range[0])
    gap = cfg.pile_gap_range[0] + difficulty * (cfg.pile_gap_range[1] - cfg.pile_gap_range[0])
    pit_depth = cfg.pit_depth_range[0] + difficulty * (cfg.pit_depth_range[1] - cfg.pit_depth_range[0])
    height_variation = cfg.height_variation_range[0] + difficulty * (
        cfg.height_variation_range[1] - cfg.height_variation_range[0]
    )
    radius = 0.5 * diameter
    pitch = diameter + gap
    center_x, center_y = cfg.size[0] / 2.0, cfg.size[1] / 2.0
    landing_start = cfg.size[0] - cfg.landing_length

    # A low floor keeps falls finite while the two full-width platforms prevent
    # bypassing the pile field around its sides.
    meshes = [
        trimesh.creation.box(
            (cfg.size[0], cfg.size[1], 1.0),
            trimesh.transformations.translation_matrix((center_x, center_y, -pit_depth - 0.5)),
        ),
        trimesh.creation.box(
            (cfg.approach_length, cfg.size[1], 1.0),
            trimesh.transformations.translation_matrix((0.5 * cfg.approach_length, center_y, -0.5)),
        ),
        trimesh.creation.box(
            (cfg.landing_length, cfg.size[1], 1.0),
            trimesh.transformations.translation_matrix((landing_start + 0.5 * cfg.landing_length, center_y, -0.5)),
        ),
    ]

    row = 0
    x = cfg.approach_length + radius
    while x <= landing_start - radius + 1.0e-6:
        # Alternating half-pitch offsets form the characteristic plum-blossom
        # pattern and prevent the policy from learning one straight rail.
        y = radius + (0.5 * pitch if row % 2 else 0.0)
        column = 0
        while y <= cfg.size[1] - radius + 1.0e-6:
            phase = (row * 3 + column * 5) % 7 / 6.0
            top = height_variation * phase
            height = pit_depth + top
            meshes.append(
                trimesh.creation.cylinder(
                    radius=radius,
                    height=height,
                    sections=12,
                    transform=trimesh.transformations.translation_matrix((x, y, -pit_depth + 0.5 * height)),
                )
            )
            y += pitch
            column += 1
        x += pitch
        row += 1

    return meshes, np.asarray((center_x, center_y, 0.0))


def next_terrain_levels(
    levels: torch.Tensor,
    move_up: torch.Tensor,
    move_down: torch.Tensor,
    max_terrain_level: int,
) -> torch.Tensor:
    """Advance curriculum levels and keep solved robots at the hardest level."""
    return (levels + move_up.to(levels.dtype) - move_down.to(levels.dtype)).clamp(0, max_terrain_level - 1)


WmpTiltTerrainCfg.function = wmp_tilt_terrain
WmpCrawlTerrainCfg.function = wmp_crawl_terrain
WmpPlumPilesTerrainCfg.function = wmp_plum_piles_terrain


@configclass
class WmpObstacleTerrainCfg(SubTerrainBaseCfg):
    """IsaacLab 网格地形；所有几何保持在所属单元内，避免跨等级重叠。"""

    function = None
    kind: str = "gap"
    channel_width: float = 2.4


def _box(size, position):
    return trimesh.creation.box(size, trimesh.transformations.translation_matrix(position))


def wmp_obstacle_terrain(difficulty: float, cfg: WmpObstacleTerrainCfg):
    d = float(np.clip(difficulty, 0.0, 1.0))
    length, width = cfg.size
    center_y = width / 2
    kind = cfg.kind
    meshes = []
    origin = np.array((1.0, center_y, 0.0))
    pit = kind in {
        "gap",
        "continuous_gap",
        "bream",
        "stepping_stones",
        "stepping_one_bridge",
        "stairs_cliff",
        "slope_cliff",
        "stool",
    }
    meshes.append(_box((length, width, 1.0), (length / 2, center_y, -1.5 if pit else -0.5)))
    if pit:
        meshes.append(_box((2.0, width, 1.0), (1.0, center_y, -0.5)))
        meshes.append(_box((0.8, width, 1.0), (length - 0.4, center_y, -0.5)))
    if kind == "gap":
        gap = 0.10 + 0.80 * d
        meshes.append(_box((length - 2.0 - gap, cfg.channel_width, 1.0), ((length + 2.0 + gap) / 2, center_y, -0.5)))
    elif kind == "continuous_gap":
        gap = 0.08 + 0.42 * d
        x = 2.0 + gap
        while x + 0.4 < length - 0.8:
            top = 0.2 + 0.1 * np.random.random()
            meshes.append(_box((0.4, cfg.channel_width, 1.0 + top), (x + 0.2, center_y, (top - 1) / 2)))
            x += 0.4 + gap
    elif kind in {"bream", "stepping_stones", "stepping_one_bridge"}:
        block = {"bream": 0.9 - 0.4 * d, "stepping_stones": 0.4 - 0.15 * d, "stepping_one_bridge": 0.5 - 0.3 * d}[kind]
        gap = 0.1 + (0.4 if kind == "bream" else 0.05) * d
        x = 2.0 + gap
        while x + block < length - 0.8:
            if kind == "stepping_stones":
                for y in np.arange(center_y - 1, center_y + 1, block + gap):
                    top = np.random.uniform(0, 0.05 + 0.1 * d)
                    meshes.append(_box((block, block, 1 + top), (x + block / 2, y, (top - 1) / 2)))
            else:
                w = np.random.uniform(0.5, 1.0) if kind == "bream" else block
                top = np.random.uniform(0, 0.05 + 0.18 * d) if kind == "bream" else 0.0
                meshes.append(_box((block, w, 1 + top), (x + block / 2, center_y, (top - 1) / 2)))
            x += block + gap
    elif kind == "stairs_cliff":
        step_width, n = 0.25, 10
        step_height = 0.05 + 0.21 * d
        for i in range(n * 2):
            top = (min(i + 1, n * 2 - i)) * step_height
            meshes.append(
                _box((step_width, cfg.channel_width, 1 + top), (2.0 + (i + 0.5) * step_width, center_y, (top - 1) / 2))
            )
    elif kind == "slope_cliff":
        ramp_length = 2.4
        top = np.tan(np.deg2rad(10 + 25 * d)) * ramp_length
        x = np.array((2, 2 + ramp_length, 2 + ramp_length + 0.4, 2 + 2 * ramp_length + 0.4))
        # 两侧三角面和封闭底面共同组成坡道，保留真实斜面而非阶梯近似。
        vertices = np.array(
            [
                (xx, yy, zz)
                for yy in (center_y - cfg.channel_width / 2, center_y + cfg.channel_width / 2)
                for xx, zz in zip(x, (0, top, top, 0), strict=True)
            ]
            + [
                (2, center_y - cfg.channel_width / 2, -1),
                (x[-1], center_y - cfg.channel_width / 2, -1),
                (2, center_y + cfg.channel_width / 2, -1),
                (x[-1], center_y + cfg.channel_width / 2, -1),
            ]
        )
        faces = [
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (0, 8, 9),
            (0, 9, 3),
            (0, 3, 2),
            (0, 2, 1),
            (4, 7, 11),
            (4, 11, 10),
            (4, 5, 6),
            (4, 6, 7),
            (8, 10, 11),
            (8, 11, 9),
            (0, 4, 10),
            (0, 10, 8),
            (3, 9, 11),
            (3, 11, 7),
        ]
        meshes.append(trimesh.Trimesh(vertices=vertices, faces=faces, process=False))
    elif kind == "climb":
        for x in (3.3, 6.8):
            height = 0.15 + 0.45 * d
            meshes.append(_box((1.0, width, height), (min(x + 0.5, length - 0.5), center_y, height / 2)))
    elif kind == "stool":
        clear, thick, stool_length = 0.42 - 0.15 * d, 0.06, 0.6
        stool_width = 1.8
        meshes.append(_box((length - 2, stool_width, 1), ((length + 2) / 2, center_y, -0.5)))
        for x in (2.4, 4.8, 7.2):
            meshes.append(_box((stool_length, stool_width, thick), (x, center_y, clear + thick / 2)))
            for sign in (-1, 1):
                meshes.append(
                    _box((stool_length, thick, clear), (x, center_y + sign * (stool_width - thick) / 2, clear / 2))
                )
    else:
        raise ValueError(f"unsupported custom terrain {kind}")
    return meshes, origin


WmpObstacleTerrainCfg.function = wmp_obstacle_terrain


def wmp_rough_slope_terrain(difficulty, cfg):
    """难度从平地向 ±0.45 坡比增加，升降方向独立采样。"""
    from isaaclab.terrains.height_field.hf_terrains import pyramid_sloped_terrain

    local = cfg.copy()
    local.slope_range = (0.0, 0.45)
    local.inverted = bool(np.random.random() < 0.5)
    return pyramid_sloped_terrain(difficulty, local)
