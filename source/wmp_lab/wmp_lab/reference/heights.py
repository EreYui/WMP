# SPDX-License-Identifier: Apache-2.0
"""参考高度点、三邻格查询与全局脚边查表，支持独立数值验证。"""

import torch


def rotate_height_points(quaternion, points):
    # 原版先清除 quaternion 的 x/y 再归一化；这与一般 Euler yaw 提取不同。
    wz = quaternion[:, (0, 3)]
    wz = wz / wz.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    w, z = wz[:, 0:1], wz[:, 1:2]
    cosine, sine = 1 - 2 * z.square(), 2 * w * z
    x = cosine * points[None, :, 0] - sine * points[None, :, 1]
    y = sine * points[None, :, 0] + cosine * points[None, :, 1]
    return torch.stack((x, y, points[None, :, 2].expand_as(x)), -1)


def height_points(forward=False, device="cpu"):
    x = torch.tensor([i / 10 for i in (range(21) if forward else range(-8, 9))], device=device)
    y = torch.tensor([i / 10 for i in (range(-12, 13) if forward else range(-5, 6))], device=device)
    gx, gy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((gx.flatten(), gy.flatten(), torch.zeros(gx.numel(), device=device)), -1)


def sample_heights(samples, world_points, border, horizontal_scale, vertical_scale):
    indices = ((world_points[..., :2] + border) / horizontal_scale).long()
    x = indices[..., 0].clamp(0, samples.shape[0] - 2)
    y = indices[..., 1].clamp(0, samples.shape[1] - 2)
    return torch.minimum(torch.minimum(samples[x, y], samples[x + 1, y]), samples[x, y + 1]) * vertical_scale


def feet_at_edge(mask, positions, contacts, border, horizontal_scale):
    indices = ((positions[..., :2] + border) / horizontal_scale).round().long()
    x = indices[..., 0].clamp(0, mask.shape[0] - 1)
    y = indices[..., 1].clamp(0, mask.shape[1] - 1)
    return mask[x, y] & contacts
