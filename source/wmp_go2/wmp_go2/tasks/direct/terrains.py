# SPDX-License-Identifier: MIT
"""WMP obstacle terrain primitives adapted to IsaacLab's terrain generator API."""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
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


WmpTiltTerrainCfg.function = wmp_tilt_terrain
WmpCrawlTerrainCfg.function = wmp_crawl_terrain
