# SPDX-License-Identifier: Apache-2.0
"""覆盖完成集隔离、长度加权、相机稀疏映射与无损深度。"""

import torch
from wmp_lab.learning.sequence_replay import EpisodeReplay, SequenceReplay


def frame(step, num_envs=3):
    return dict(
        prop=torch.arange(num_envs).float()[:, None] * 100 + step,
        real=torch.arange(num_envs).float()[:, None, None, None] + step * 0.0012345,
        is_first=torch.zeros(num_envs),
    )


def test_completed_episode_isolation_replacement_and_sparse_depth():
    replay = EpisodeReplay(3, 12, torch.tensor([False, True, True]))
    for step in range(4):
        replay.insert(frame(step))
    assert replay.total_samples == 0
    replay.finish_episodes(torch.tensor([1, 2]))
    assert replay.lengths.tolist() == [0, 4, 4]
    for step in range(4, 8):
        replay.insert(frame(step))
    batch = replay.sample(20, 4)
    assert batch["prop"][..., 0].remainder(100).max() == 3
    expected = batch["prop"] // 100 + batch["prop"].remainder(100) * 0.0012345
    torch.testing.assert_close(batch["real"].flatten(2), expected, rtol=0, atol=0)
    replay.finish_episodes(torch.tensor([1]))
    batch = replay.sample(8, 4, env_ids=torch.tensor([1]))
    torch.testing.assert_close(batch["prop"][..., 0], torch.arange(104, 108).float().expand(8, -1))
    assert torch.all(batch["is_first"][:, 0] == 1)
    assert torch.all(batch["is_first"][:, 1:] == 0)


def test_world_sampling_weights_episode_lengths_and_depth_sampling_is_uniform():
    torch.manual_seed(42)
    replay = EpisodeReplay(3, 12, torch.ones(3, dtype=torch.bool))
    for step in range(8):
        replay.insert(frame(step))
        if step == 1:
            replay.finish_episodes(torch.tensor([0]))
    replay.finish_episodes(torch.tensor([1, 2]))
    # 有效长度为 2、8、8；当前未完成 episode 不进入采样。
    batch = replay.sample(9000, 1)
    counts = torch.bincount((batch["prop"][:, 0, 0] // 100).long(), minlength=3)
    torch.testing.assert_close(counts.float() / len(batch["prop"]), torch.tensor([2, 8, 8]) / 18, atol=0.02, rtol=0)
    batch = replay.sample(9000, 1, env_ids=torch.tensor([0, 1]))
    counts = torch.bincount((batch["prop"][:, 0, 0] // 100).long(), minlength=3)
    torch.testing.assert_close(counts.float() / len(batch["prop"]), torch.tensor([0.5, 0.5, 0]), atol=0.02, rtol=0)


def test_sparse_ring_retains_float_depth_without_clipping_or_quantization():
    replay = SequenceReplay(3, 12, torch.tensor([False, True, False]))
    for step in range(4):
        replay.insert(frame(step))
    batch = replay.sample(8, 4, env_ids=torch.tensor([1]))
    assert replay.storage["real"].shape == (4, 1, 1, 1, 1)
    expected = torch.stack([frame(step)["real"][1] for step in range(4)]).expand(8, -1, -1, -1, -1)
    torch.testing.assert_close(batch["real"], expected, rtol=0, atol=0)
