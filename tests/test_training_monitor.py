# SPDX-License-Identifier: Apache-2.0
"""独立实际步数的episode统计必须排除首段，并区分超时与失败。"""

import runpy
from pathlib import Path

import torch

EpisodeMonitor = runpy.run_path(str(Path(__file__).resolve().parents[1] / "tools/training_monitor.py"))[
    "EpisodeMonitor"
]


def test_episode_monitor_discards_initial_partial_and_accumulates_across_calls():
    monitor = EpisodeMonitor(2, 0.02)
    monitor.update(torch.tensor([1.0, 2.0]), torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool))
    assert monitor.metrics()["episode_window_count"] == 0
    monitor.update(torch.tensor([3.0, 4.0]), torch.tensor([False, True]), torch.zeros(2, dtype=torch.bool))
    assert monitor.discarded == 2
    monitor.update(torch.tensor([5.0, 6.0]), torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool))
    metrics = monitor.metrics()
    assert metrics["episodes_completed"] == 1
    assert abs(metrics["episode_length_mean_s"] - 0.04) < 1e-6
    assert metrics["episode_task_return_mean"] == 8


def test_episode_monitor_window_and_timeout_excludes_simultaneous_failure():
    monitor = EpisodeMonitor(1, 0.02, window=2)
    one, yes, no = torch.ones(1), torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
    monitor.update(one, yes, no)
    monitor.update(one, no, yes, -one)
    monitor.update(one, yes, yes, -one * 2)
    metrics = monitor.metrics()
    assert metrics["episode_timeout_fraction"] == 0.5
    assert metrics["episode_unclipped_task_return_mean"] == -1.5
    monitor.update(one * 3, yes, no)
    assert monitor.metrics()["episodes_completed"] == 3
    assert monitor.metrics()["episode_window_count"] == 2
    assert monitor.metrics()["episode_task_return_mean"] == 2
