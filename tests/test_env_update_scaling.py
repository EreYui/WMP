# SPDX-License-Identifier: Apache-2.0
"""核对完整更新批次，禁止恢复旧版按环境数取整的近似配置。"""

from dataclasses import replace

import pytest
from wmp_lab.learning.config import PAPER_CFG, SMOKE_CFG, get_learning_cfg
from wmp_lab.learning.runner import WMPRunner


@pytest.mark.parametrize("num_envs,chunks", [(4096, 1), (2048, 2), (1024, 4), (512, 8)])
def test_complete_reference_update_batch(num_envs, chunks):
    cfg = get_learning_cfg(num_envs=num_envs)
    cfg.validate_schedule()
    assert cfg.collection_chunks == cfg.completed_episodes_per_env == chunks
    assert cfg.batch_samples == 4096 * 24
    assert cfg.batch_samples // cfg.mini_batches == 24576
    assert cfg.rollout_steps == 24
    assert (cfg.world_train_steps, cfg.depth_train_steps) == (10, 50)
    assert cfg.learning_epochs * cfg.mini_batches == 20


def test_reference_smoke_and_unsupported_counts():
    assert get_learning_cfg() is PAPER_CFG
    assert get_learning_cfg(True) is SMOKE_CFG
    assert get_learning_cfg(True, num_envs=32) == replace(SMOKE_CFG, num_envs=32)
    for count in (1023, 4100, 8200):
        with pytest.raises(ValueError, match="divide 4096"):
            get_learning_cfg(num_envs=count)
    with pytest.raises(ValueError, match="positive"):
        get_learning_cfg(num_envs=0)


def test_clock_and_configuration_survive_checkpoint(tmp_path):
    cfg = replace(SMOKE_CFG, collection_chunks=4, completed_episodes_per_env=4)
    runner = WMPRunner(cfg)
    runner.total_steps = 123456
    runner.iteration = 23
    path = runner.save(tmp_path / "aligned.pt")
    loaded = WMPRunner.from_checkpoint(path)
    assert loaded.cfg == cfg
    assert loaded.total_steps == 123456 and loaded.iteration == 23
    clone = WMPRunner(cfg)
    clone.load(path)
    assert clone.cfg.collection_chunks == 4
    with pytest.raises(ValueError, match="configuration differs"):
        WMPRunner(replace(cfg, collection_chunks=2)).load(path)
