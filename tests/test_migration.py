# SPDX-License-Identifier: Apache-2.0
"""验证连续训练、reset 边界、导出契约和有界缓存。"""

import json
from dataclasses import replace

import torch
from wmp_lab.learning.config import SMOKE_CFG
from wmp_lab.learning.curriculum import CommandCurriculum
from wmp_lab.learning.deployment import export_policy
from wmp_lab.learning.runner import WMPRunner, generalized_advantage
from wmp_lab.learning.sequence_replay import SequenceReplay
from wmp_lab.robots import default_control_revision, robot_profile


def test_robot_contracts_and_offline_assets():
    for robot in ("go1", "go2"):
        profile = robot_profile(robot)
        assert profile.usd_path.is_file()
        assert len(profile.joint_order) == 12
    assert robot_profile("go1").camera_position != robot_profile("go2").camera_position


def test_gae_respects_termination_and_bootstrap():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[0.2], [0.3], [0.4]])
    dones = torch.tensor([[False], [True], [False]])
    advantage, returns = generalized_advantage(rewards, values, dones, torch.tensor([0.5]), 0.9, 1.0)
    torch.testing.assert_close(returns, torch.tensor([[2.8], [2.0], [3.45]]))
    torch.testing.assert_close(advantage, returns - values)


def test_command_course_groups_and_early_termination():
    curriculum = CommandCurriculum(["wave", "gap", "crawl", "rough_flat", "slope_cliff"])
    ids = torch.arange(5)
    curriculum.accumulate(ids, torch.full((5,), 95.0), torch.full((5,), 100), 100)
    assert curriculum.upper[0, 0] > 0.5
    assert curriculum.upper[3, 0] > 0.5
    before = curriculum.upper.clone()
    curriculum.accumulate(ids, torch.ones(5), torch.ones(5, dtype=torch.long), 100)
    torch.testing.assert_close(before, curriculum.upper)
    velocity, heading = curriculum.sample(ids, zero_probability=0)
    assert (velocity[[1, 2, 4], 0] >= 0).all()
    assert (velocity[[1, 2, 4], 1] == 0).all()
    assert (heading[[1, 2, 4]] == 0).all()


def test_sequence_replay_wrap_preserves_order():
    replay = SequenceReplay(2, 6)
    for step in range(8):
        replay.insert(
            dict(
                prop=torch.full((2, 33), float(step)),
                image=torch.zeros(2, 64, 64, 1),
                action=torch.zeros(2, 60),
                reward=torch.zeros(2),
                is_first=torch.zeros(2),
            )
        )
    batch = replay.sample(16, 3)
    torch.testing.assert_close(batch["prop"][..., 0], torch.tensor([[5.0, 6.0, 7.0]]).expand(16, -1))
    assert replay.size == replay.capacity == 3
    assert replay.storage["image"].dtype == torch.float32
    assert (batch["is_first"][:, 0] == 1).all()


def test_deployment_serialization_and_robot_metadata(tmp_path):
    for robot in ("go1", "go2"):
        runner = WMPRunner(replace(SMOKE_CFG, robot=robot))
        path, error = export_policy(runner, tmp_path / f"{robot}.pt", batch_size=2)
        assert error < 1e-4
        metadata = json.loads(path.with_suffix(".json").read_text())
        assert metadata["format_version"] == 4
        assert metadata["control_revision"] == default_control_revision(robot)
        assert metadata["robot"]["torque_limits"] == list(robot_profile(robot).torque_limits)
        policy = torch.jit.load(str(path))
        with torch.inference_mode():
            policy.reset()
            output = policy(torch.zeros(2, 45), torch.zeros(2, 64, 64, 1))
        assert output.shape == (2, 12)
        assert torch.isfinite(output).all()


def test_continuous_runner_preserves_rollout_and_terminal_amp(tmp_path):
    from pathlib import Path

    import wmp_lab
    from wmp_lab.learning.amp import AMPMotionDataset

    class Env:
        num_envs = 2
        device = "cpu"
        camera_mask = torch.ones(2, dtype=torch.bool)
        depth_supervision_mask = camera_mask

        def __init__(self):
            self.resets = self.steps = 0
            self.obs = torch.zeros(2, 285)

        def reset(self):
            self.resets += 1
            return {"policy": self.obs.clone()}, {}

        def step(self, action):
            self.steps += 1
            self.obs[:, 86:98] = action.clamp(-6, 6)
            done = torch.tensor([self.steps % 3 == 0, False])
            terminal = self.get_amp_observations() + 7
            return (
                {"policy": self.obs.clone()},
                torch.ones(2) * 0.02,
                done,
                torch.zeros(2, dtype=torch.bool),
                {"terminal_amp_states": terminal, "terminal_observation": self.obs.clone()},
            )

        def get_amp_observations(self):
            return torch.zeros(2, 30)

        def get_forward_height_map(self):
            return torch.zeros(2, 525)

        def get_depth(self):
            return torch.zeros(2, 64, 64, 1)

    env = Env()
    expert = AMPMotionDataset(Path(wmp_lab.__file__).parent / "data/motions")
    runner = WMPRunner(replace(SMOKE_CFG, learning_epochs=2, mini_batches=2))
    batches = []
    actions = []
    original = runner.ppo.update

    def capture(batch, expert, **kwargs):
        batches.append(batch["amp_next_state"].clone())
        actions.append(batch["action"].clone())
        return original(batch, expert, **kwargs)

    runner.ppo.update = capture
    runner.collect_and_update(env, expert, rollout_steps=6)
    assert env.resets == 1
    assert any((batch == 7).any() for batch in batches)
    runner.collect_and_update(env, expert, rollout_steps=5)
    assert env.resets == 1
    assert runner.rollout_step == 11
    assert runner.iteration == 2
    assert runner.world_replay.size == 3
    assert len(batches) == 8
    # 原版 RolloutStorage 在同轮 PPO 的两个 epoch 中复用 minibatch。
    for start in (0, 4):
        torch.testing.assert_close(actions[start], actions[start + 2], rtol=0, atol=0)
        torch.testing.assert_close(actions[start + 1], actions[start + 3], rtol=0, atol=0)
