# SPDX-License-Identifier: Apache-2.0
"""将固定 Go1 高度场、附加障碍和边缘掩码接入 IsaacLab。"""

from types import SimpleNamespace

import numpy as np
import torch
import trimesh
from isaaclab import sim as sim_utils
from isaaclab.terrains import SubTerrainBaseCfg, TerrainImporter
from isaaclab.utils import configclass

from wmp_lab.reference import meshes
from wmp_lab.reference.primitives import SubTerrain
from wmp_lab.reference.terrain import Terrain, convert_heightfield_to_trimesh

from .terrains import WmpPlumPilesTerrainCfg, wmp_plum_piles_terrain

REFERENCE_TERRAINS = (
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
REFERENCE_WEIGHTS = (0, 0.1, 0.1, 0.1, 0.05, 0.1, 0.05, 0.05, 0.1, 0.05, 0, 0.05, 0.1, 0, 0.05, 0.05, 0.05, 0)


def _core_cfg(generator):
    weights = {name.removeprefix("wmp_"): item.proportion for name, item in generator.sub_terrains.items()}
    first = next(iter(generator.sub_terrains.values()))
    return SimpleNamespace(
        mesh_type="trimesh",
        terrain_length=generator.size[0],
        terrain_width=generator.size[1],
        horizontal_scale=generator.horizontal_scale,
        vertical_scale=generator.vertical_scale,
        border_size=generator.border_width,
        num_rows=generator.num_rows,
        num_cols=generator.num_cols,
        terrain_proportions=[weights.get(name, 0) for name in REFERENCE_TERRAINS],
        slope_threshold=generator.slope_threshold,
        curriculum=True,
        selected=False,
        tilt_width_range=getattr(first, "tilt_width_range", (0.28, 0.32)),
    )


class ExtendedReferenceTerrain(Terrain):
    def __init__(self, *args, **kwargs):
        self._plum_tiles = []
        super().__init__(*args, **kwargs)
        if self._plum_tiles:
            # 扩展圆桩使用精确网格，去掉该单元栅格近似产生的碰撞三角面。
            nx, ny = self.height_field_raw.shape
            count = 2 * (nx - 1) * (ny - 1)
            ids = np.arange(count)
            x, y = ids // (2 * (ny - 1)), (ids // 2) % (ny - 1)
            keep = np.ones(count, dtype=bool)
            for row, column in self._plum_tiles:
                x0 = self.border + row * self.length_per_env_pixels
                y0 = self.border + column * self.width_per_env_pixels
                keep &= ~(
                    (x >= x0) & (x < x0 + self.length_per_env_pixels) & (y >= y0) & (y < y0 + self.width_per_env_pixels)
                )
            self.triangles = np.concatenate((self.triangles[:count][keep], self.triangles[count:]))

    def make_terrain(self, choice, difficulty, i=0, j=0):
        if choice < self.proportions[16]:
            return super().make_terrain(choice, difficulty, i, j)
        # 梅花桩保持显式扩展；参考默认比例为零。
        cfg = WmpPlumPilesTerrainCfg(size=(self.env_length, self.env_width))
        parts, _ = wmp_plum_piles_terrain(difficulty, cfg)
        surface = SubTerrain(
            width=self.width_per_env_pixels,
            length=self.length_per_env_pixels,
            horizontal_scale=self.cfg.horizontal_scale,
            vertical_scale=self.cfg.vertical_scale,
        )
        xy = np.stack(
            np.meshgrid(
                np.arange(surface.width) * surface.horizontal_scale,
                np.arange(surface.length) * surface.horizontal_scale,
                indexing="ij",
            ),
            -1,
        )
        top = np.full(xy.shape[:-1], parts[0].bounds[1, 2])
        for index, mesh in enumerate(parts[1:], 1):
            lo, hi = mesh.bounds
            inside = ((xy >= lo[:2]) & (xy <= hi[:2])).all(-1)
            if index >= 3:
                center, radius = (lo[:2] + hi[:2]) / 2, (hi[0] - lo[0]) / 2
                inside &= ((xy - center) ** 2).sum(-1) <= radius**2
            top[inside] = np.maximum(top[inside], hi[2])
        surface.height_field_raw[:] = np.rint(top / surface.vertical_scale).astype(np.int16)
        # 高度图只用于查询；实际碰撞使用原来的圆桩网格。
        offset = np.array((i * self.env_length + self.cfg.border_size, j * self.env_width + self.cfg.border_size, 0))
        for part in parts:
            pair = (np.asarray(part.vertices + offset, dtype=np.float32), np.asarray(part.faces, dtype=np.uint32))
            self.added_trimesh = (
                pair if self.added_trimesh is None else meshes.combine_trimeshes(self.added_trimesh, pair)
            )
        if hasattr(self, "_plum_tiles"):
            self._plum_tiles.append((i, j))
        surface.env_origin_z = 0.0
        return surface


def reference_tile(difficulty, cfg):
    """单地形预览接口；正式地图由整个原始高度场一次性转网格。"""
    if cfg.kind == "plum_piles":
        return wmp_plum_piles_terrain(difficulty, WmpPlumPilesTerrainCfg(size=cfg.size))
    core = ExtendedReferenceTerrain.__new__(ExtendedReferenceTerrain)
    index = REFERENCE_TERRAINS.index(cfg.kind)
    weights = [0.0] * len(REFERENCE_TERRAINS)
    weights[index] = 1.0
    core.cfg = SimpleNamespace(
        mesh_type="trimesh",
        terrain_length=cfg.size[0],
        terrain_width=cfg.size[1],
        horizontal_scale=cfg.horizontal_scale,
        vertical_scale=cfg.vertical_scale,
        border_size=0.0,
        num_rows=cfg.reference_rows,
        num_cols=1,
        terrain_proportions=weights,
        slope_threshold=0.75,
        tilt_width_range=cfg.tilt_width_range,
    )
    core.env_length, core.env_width = cfg.size
    core.width_per_env_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    core.length_per_env_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    core.proportions = np.cumsum(weights)
    core.added_trimesh = None
    surface = core.make_terrain(0.5, difficulty)
    verts, faces, _ = convert_heightfield_to_trimesh(
        surface.height_field_raw, cfg.horizontal_scale, cfg.vertical_scale, 0.75
    )
    pair = (verts, faces)
    if core.added_trimesh is not None:
        pair = meshes.combine_trimeshes(pair, core.added_trimesh)
    x0 = cfg.size[0] / 2 + getattr(surface, "env_origin_x_offset", 0)
    z0 = getattr(surface, "env_origin_z", None)
    if z0 is None:
        x, y = surface.height_field_raw.shape
        z0 = surface.height_field_raw[x // 2 - 10 : x // 2 + 10, y // 2 - 10 : y // 2 + 10].max() * cfg.vertical_scale
    return [trimesh.Trimesh(vertices=pair[0], faces=pair[1], process=False)], np.array((x0, cfg.size[1] / 2, z0))


@configclass
class ReferenceSubTerrainCfg(SubTerrainBaseCfg):
    function = reference_tile
    kind: str = "rough_flat"
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    reference_rows: int = 10
    tilt_width_range: tuple[float, float] = (0.28, 0.32)


class ReferenceTerrainImporter(TerrainImporter):
    def __init__(self, cfg):
        cfg.validate()
        self.cfg, self.device = cfg, sim_utils.SimulationContext.instance().device
        self.terrain_prim_paths = []
        self.terrain_origins = self.env_origins = None
        self._terrain_flat_patches = {}
        terrain = ExtendedReferenceTerrain(_core_cfg(cfg.terrain_generator), cfg.num_envs)
        vertices = terrain.vertices.copy()
        vertices[:, :2] -= terrain.cfg.border_size
        self.render_vertices, self.render_triangles = vertices, terrain.triangles
        self.import_mesh("terrain", trimesh.Trimesh(vertices=vertices, faces=terrain.triangles, process=False))
        self.configure_env_origins(terrain.env_origins)
        self.height_samples = torch.as_tensor(terrain.height_field_raw, device=self.device)
        self.x_edge_mask = torch.as_tensor(terrain.x_edge_mask, device=self.device)
        self.heightfield_border = terrain.cfg.border_size
        self.set_debug_vis(cfg.debug_vis)
