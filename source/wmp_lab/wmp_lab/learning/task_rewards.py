# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# Portions Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES and ETH Zurich, Nikita Rudin.
# Portions Copyright Bytedance Ltd. and/or its affiliates.
"""不依赖模拟器的上游任务奖励，便于与原函数直接差分。"""

import torch


def tracking_velocity(velocity, command, obstacle_mask, sigma=0.25, clip=0.1):
    value = velocity[:, :2].clone()
    selected = command[obstacle_mask, :2]
    upper = torch.where(selected < 0, 1.0e5, selected + clip)
    lower = torch.where(selected > 0, -1.0e5, selected - clip)
    value[obstacle_mask] = value[obstacle_mask].clamp(lower, upper)
    return torch.exp(-(command[:, :2] - value).square().sum(-1) / sigma)


def stuck_penalty(velocity, command):
    return ((velocity[:, 0].abs() < 0.1) & (command[:, 0].abs() > 0.1)).float()
