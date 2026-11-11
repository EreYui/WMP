# SPDX-License-Identifier: Apache-2.0
"""用 NumPy 实现参考任务需要的基础高度场，不导入 Isaac Gym。"""

import numpy as np
from scipy.interpolate import RectBivariateSpline


class SubTerrain:
    def __init__(self, terrain_name="terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name, self.width, self.length = terrain_name, width, length
        self.vertical_scale, self.horizontal_scale = vertical_scale, horizontal_scale
        self.height_field_raw = np.zeros((width, length), dtype=np.int16)


def random_uniform_terrain(terrain, min_height, max_height, step=1, downsampled_scale=None):
    scale = downsampled_scale or terrain.horizontal_scale
    low, high, delta = (int(value / terrain.vertical_scale) for value in (min_height, max_height, step))
    nx, ny = (int(n * terrain.horizontal_scale / scale) for n in (terrain.width, terrain.length))
    samples = np.random.choice(np.arange(low, high + delta, delta), (nx, ny))
    xmax, ymax = terrain.width * terrain.horizontal_scale, terrain.length * terrain.horizontal_scale
    # 线性样条保留旧 interp2d 的规则网格插值和舍入定义，兼容新版 SciPy。
    spline = RectBivariateSpline(np.linspace(0, xmax, nx), np.linspace(0, ymax, ny), samples, kx=1, ky=1)
    terrain.height_field_raw += np.rint(
        spline(np.linspace(0, xmax, terrain.width), np.linspace(0, ymax, terrain.length))
    ).astype(np.int16)
    return terrain


def pyramid_sloped_terrain(terrain, slope=1, platform_size=1.0):
    cx, cy = terrain.width // 2, terrain.length // 2
    x = ((cx - np.abs(cx - np.arange(terrain.width))) / cx)[:, None]
    y = ((cy - np.abs(cy - np.arange(terrain.length))) / cy)[None, :]
    peak = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2))
    terrain.height_field_raw += (peak * x * y).astype(np.int16)
    radius = int(platform_size / terrain.horizontal_scale / 2)
    height = terrain.height_field_raw[cx - radius, cy - radius]
    terrain.height_field_raw = np.clip(terrain.height_field_raw, min(height, 0), max(height, 0))
    return terrain


def discrete_obstacles_terrain(terrain, max_height, min_size, max_size, num_rects, platform_size=1.0):
    high = int(max_height / terrain.vertical_scale)
    sizes = range(int(min_size / terrain.horizontal_scale), int(max_size / terrain.horizontal_scale), 4)
    heights = [-high, -high // 2, high // 2, high]
    for _ in range(num_rects):
        nx, ny = np.random.choice(sizes), np.random.choice(sizes)
        x, y = np.random.choice(range(0, terrain.width - nx, 4)), np.random.choice(range(0, terrain.length - ny, 4))
        terrain.height_field_raw[x : x + nx, y : y + ny] = np.random.choice(heights)
    _clear_platform(terrain, platform_size)
    return terrain


def wave_terrain(terrain, num_waves=1, amplitude=1.0):
    if num_waves:
        height = int(0.5 * amplitude / terrain.vertical_scale)
        period = terrain.length / (2 * np.pi * num_waves)
        x, y = np.arange(terrain.width)[:, None], np.arange(terrain.length)[None, :]
        terrain.height_field_raw += (height * np.cos(y / period) + height * np.sin(x / period)).astype(np.int16)
    return terrain


def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1.0):
    width = int(step_width / terrain.horizontal_scale)
    height = int(step_height / terrain.vertical_scale)
    platform = int(platform_size / terrain.horizontal_scale)
    x0 = y0 = top = 0
    x1, y1 = terrain.width, terrain.length
    while x1 - x0 > platform and y1 - y0 > platform:
        x0, x1, y0, y1 = x0 + width, x1 - width, y0 + width, y1 - width
        top += height
        terrain.height_field_raw[x0:x1, y0:y1] = top
    return terrain


def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1.0, depth=-10):
    size, gap = (int(value / terrain.horizontal_scale) for value in (stone_size, stone_distance))
    amplitude = int(max_height / terrain.vertical_scale)
    heights = np.arange(-amplitude - 1, amplitude)
    terrain.height_field_raw.fill(int(depth / terrain.vertical_scale))
    along_y = terrain.length >= terrain.width
    outer_limit = terrain.length if along_y else terrain.width
    inner_limit = terrain.width if along_y else terrain.length
    for outer in range(0, outer_limit, size + gap):
        stop_outer = min(outer + size, outer_limit)
        inner = np.random.randint(0, size)

        def fill(start, stop, outer=outer, stop_outer=stop_outer):
            region = (slice(start, stop), slice(outer, stop_outer))
            if not along_y:
                region = region[::-1]
            terrain.height_field_raw[region] = np.random.choice(heights)

        fill(0, max(0, inner - gap))
        while inner < inner_limit:
            fill(inner, min(inner_limit, inner + size))
            inner += size + gap
    _clear_platform(terrain, platform_size)
    return terrain


def _clear_platform(terrain, size):
    width = int(size / terrain.horizontal_scale)
    terrain.height_field_raw[
        (terrain.width - width) // 2 : (terrain.width + width) // 2,
        (terrain.length - width) // 2 : (terrain.length + width) // 2,
    ] = 0
