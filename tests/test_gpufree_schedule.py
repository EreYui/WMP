# SPDX-License-Identifier: Apache-2.0
"""以原版 GAE 和完整梯度为独立对照，验证小环境完整批次与课程时钟。"""

import ast
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from wmp_lab.learning.config import SMOKE_CFG, get_learning_cfg
from wmp_lab.learning.curriculum import CommandCurriculum, TerrainCurriculumSlots, reference_group_sizes
from wmp_lab.learning.networks import ActorCriticWMP, AMPDiscriminator
from wmp_lab.learning.ppo import AMPPPO
from wmp_lab.learning.runner import WMPRunner, generalized_advantage, reference_batch_order, segmented_advantage
from wmp_lab.learning.sequence_replay import EpisodeReplay


def test_segmented_gae_matches_original_parallel_rollout_and_resets():
    torch.manual_seed(17)
    chunks, steps, envs = 4, 24, 3
    rewards, values = torch.randn(2, chunks * steps, envs)
    dones = torch.rand(chunks * steps, envs) < 0.12
    bootstrap = torch.randn(chunks, envs)
    advantage, returns = segmented_advantage(rewards, values, dones, list(bootstrap), 0.99, 0.95)
    source = Path(__file__).resolve().parents[2] / "wmp_go1/rsl_rl/storage/rollout_storage.py"
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute_returns")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    def fold(value):
        return reference_batch_order(value, chunks, steps).reshape(steps, chunks * envs, 1)

    reference = SimpleNamespace(
        num_transitions_per_env=steps, rewards=fold(rewards), values=fold(values), dones=fold(dones),
        returns=torch.empty(steps, chunks * envs, 1),
    )
    scope["compute_returns"](reference, bootstrap.reshape(-1, 1), 0.99, 0.95)
    actual_adv = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    torch.testing.assert_close(fold(returns), reference.returns)
    torch.testing.assert_close(fold(actual_adv), reference.advantages)
    extended, _ = generalized_advantage(rewards, values, dones, bootstrap[-1], 0.99, 0.95)
    assert (extended - advantage).abs().max() > 0.1


def test_micro_batches_match_full_minibatch_gradient_adam_kl_and_amp_statistics():
    torch.manual_seed(31)
    policy = ActorCriticWMP(
        world_feature_dim=8, history_hidden=(16, 8), actor_hidden=(16,), critic_hidden=(16,), world_hidden=(8,),
    )
    disc = AMPDiscriminator((16, 8))
    full = AMPPPO(policy, disc, replay_size=64)
    micro = AMPPPO(copy.deepcopy(policy), copy.deepcopy(disc), replay_size=64, micro_batch_size=6)
    size = 17  # 最后一段不足微批次，必须按样本数加权。
    obs, history, feature = torch.randn(size, 285), torch.randn(size, 210), torch.randn(size, 8)
    with torch.no_grad():
        action = policy.act(obs, history, feature)
        batch = dict(
            observation=obs, critic_observation=obs, history=history, world_feature=feature, action=action,
            old_log_prob=policy.get_actions_log_prob(action), old_value=policy.evaluate(obs, feature).squeeze(-1),
            old_mean=policy.action_mean + 0.1, old_std=policy.action_std.clone(),
            advantage=torch.randn(size), **{"return": torch.randn(size)},
            amp_state=torch.randn(size, 30), amp_next_state=torch.randn(size, 30),
        )
    replay = torch.randn(size, 30), torch.randn(size, 30)
    expert_data = torch.randn(size, 30), torch.randn(size, 30), torch.ones(size)
    expert = SimpleNamespace(sample_transitions=lambda n: expert_data)
    for algo in (full, micro):
        algo.amp_replay.sample = lambda n: replay
    a, b = full.update(batch, expert), micro.update(batch, expert)
    for key in a:
        assert b[key] == pytest.approx(a[key], rel=2e-5, abs=2e-6), key
    assert full.optimizer.param_groups[0]["lr"] == micro.optimizer.param_groups[0]["lr"]
    for module_a, module_b in [(full.actor_critic, micro.actor_critic), (full.discriminator, micro.discriminator)]:
        for pa, pb in zip(module_a.parameters(), module_b.parameters(), strict=True):
            torch.testing.assert_close(pa.grad, pb.grad, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(pa, pb, rtol=2e-5, atol=5e-6)
    for key, value in full.amp_normalizer.state_dict().items():
        torch.testing.assert_close(value, micro.amp_normalizer.state_dict()[key])
    assert {float(v["step"]) for v in micro.optimizer.state.values()} == {1.0}


def test_reference_command_gate_and_pending_score_resume():
    weights = [
        ("wave", 0), ("rough_slope", .1), ("stairs_up", .1), ("stairs_down", .1), ("discrete", .05),
        ("gap", .1), ("continuous_gap", .05), ("bream", .05), ("stairs_cliff", .1), ("slope_cliff", .05),
        ("stepping_stones", 0), ("stepping_one_bridge", .05), ("climb", .1), ("tilt", 0),
        ("stool", .05), ("crawl", .05), ("rough_flat", .05),
    ]
    assert reference_group_sizes(weights, 4096) == [410, 204, 1024, 2253, 205]
    small = CommandCurriculum(["climb"] * 1024)
    small.configure_reference([("climb", 1)], 4096)
    ids, score, lengths = torch.arange(1024), torch.full((1024,), 900.), torch.full((1024,), 1000)
    for _ in range(2):
        small.accumulate(ids, score, lengths, 1000)
    assert small.upper[3, 0].item() == .5
    restored = CommandCurriculum(["climb"] * 1024)
    restored.configure_reference([("climb", 1)], 4096)
    restored.load_state_dict(small.state_dict())
    for _ in range(2):
        restored.accumulate(ids, score, lengths, 1000)
    reference = CommandCurriculum(["climb"] * 4096)
    reference.accumulate(torch.arange(4096), torch.full((4096,), 900.), torch.full((4096,), 1000), 1000)
    torch.testing.assert_close(restored.upper, reference.upper)
    assert restored.episode_count[3] == 0


def test_four_completed_episode_slots_keep_sparse_images_and_exclude_current():
    replay = EpisodeReplay(2, 8, torch.tensor([False, True]), episodes_per_env=4)
    for episode in range(6):
        for step in range(3):
            value = torch.tensor([episode * 10. + step, 100 + episode * 10. + step])
            replay.insert(dict(prop=value[:, None], real=value[:, None, None, None], is_first=torch.zeros(2)))
        replay.finish_episodes(torch.arange(2))
    assert replay.lengths.tolist() == [3] * 8
    assert replay.total_samples == 24
    assert replay.storage["real"].shape == (4, 8, 1, 1, 1)
    replay.insert(dict(prop=torch.full((2, 1), 999.), real=torch.full((2, 1, 1, 1), 999.), is_first=torch.zeros(2)))
    sample = replay.sample(4000, 3, env_ids=torch.tensor([1]))
    seen = set(sample["prop"][:, 0, 0].tolist())
    assert seen == {120., 130., 140., 150.}
    torch.testing.assert_close(sample["prop"], sample["real"].flatten(2))
    assert replay.has_completed(torch.tensor([0, 1])).all()


def test_reward_course_uses_cumulative_samples_not_optimizer_iteration(tmp_path):
    runner = WMPRunner(SMOKE_CFG)
    runner.cfg = get_learning_cfg(num_envs=1024)
    runner.total_steps = 1024 * 24 * 10000
    runner.iteration = 10000
    assert runner.curriculum_iteration == 2500
    runner.total_steps = 4096 * 24 * 4000
    assert runner.curriculum_iteration == 4000
    runner.total_steps = 4096 * 24 * 10000
    assert runner.curriculum_iteration == 10000


def test_old_schedule_checkpoint_cannot_silently_resume(tmp_path):
    old_cfg = replace(SMOKE_CFG, algorithm_revision="upstream_reference_v6")
    old = WMPRunner(old_cfg)
    path = old.save(tmp_path / "old.pt")
    with pytest.raises(ValueError, match="schedule differs"):
        WMPRunner(SMOKE_CFG).load(path)
    # 回放构造仍保持旧网络可读。
    assert WMPRunner.from_checkpoint(path).cfg.algorithm_revision == "upstream_reference_v6"


def test_predicted_depth_cache_keeps_pre_update_parameters_at_batch_boundary():
    class Env:
        num_envs = 2
        camera_mask = torch.zeros(2, dtype=torch.bool)

        def reset(self):
            self._global_step = 1
            return {"policy": torch.zeros(2, 285)}, {}

        def step(self, action):
            self._global_step += 1
            return ({"policy": torch.zeros(2, 285)}, torch.zeros(2), torch.zeros(2, dtype=torch.bool),
                    torch.zeros(2, dtype=torch.bool), {})

        def get_amp_observations(self):
            return torch.zeros(2, 30)

        def get_forward_height_map(self):
            return torch.zeros(2, 525)

        def get_depth(self):
            return torch.zeros(2, 64, 64, 1)

    env = Env()
    runner = WMPRunner(replace(SMOKE_CFG, world_train_start=10**9))
    version = [0.]
    runner.depth_predictor.forward = lambda forward, prop: torch.full((len(prop), 64, 64, 1), version[0])
    runner.ppo.update = lambda *a, **kw: {"test": 0.}
    runner.collect_and_update(env, object(), rollout_steps=4)
    assert runner.rollout_step == 5
    seen = []
    observe = runner.world_model.observe_step

    def capture(*args, **kwargs):
        seen.append(args[1].clone())
        return observe(*args, **kwargs)

    runner.world_model.observe_step = capture
    version[0] = 1.
    runner.collect_and_update(env, object(), rollout_steps=1)
    assert len(seen) == 1 and seen[0].eq(0).all()


def test_rotating_terrain_slots_match_original_independent_environment_updates():
    root = Path(__file__).resolve().parents[1]

    def method(path, name):
        tree = ast.parse(path.read_text())
        node = next(
            n for cls in tree.body if isinstance(cls, ast.ClassDef)
            for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        scope = {"torch": torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        return scope[name]

    original = method(root.parent / "wmp_go1/legged_gym/envs/base/legged_robot.py", "_update_terrain_curriculum")
    actual = method(root / "source/wmp_lab/wmp_lab/tasks/direct/wmp_env.py", "_update_terrain_curriculum")
    physical, slots = 2, 4
    pool = TerrainCurriculumSlots(torch.zeros(physical, dtype=torch.long), slots)
    reference = SimpleNamespace(
        init_done=True, root_states=torch.zeros(8, 13), env_origins=torch.zeros(8, 3),
        terrain=SimpleNamespace(env_length=8.), commands=torch.zeros(8, 4),
        max_episode_length_s=20., terrain_levels=torch.zeros(8, dtype=torch.long),
        max_terrain_level=10, terrain_origins=torch.zeros(10, 1, 3), terrain_types=torch.zeros(8, dtype=torch.long),
    )
    reference.commands[:, 0] = .5
    env = SimpleNamespace(
        _terrain=SimpleNamespace(
            terrain_levels=torch.zeros(physical, dtype=torch.long), env_origins=torch.zeros(physical, 3),
            terrain_origins=reference.terrain_origins, terrain_types=torch.zeros(physical, dtype=torch.long),
            max_terrain_level=10,
        ),
        _robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(physical, 3))),
        episode_length_buf=torch.ones(physical, dtype=torch.long) * 1000,
        _commands=torch.tensor([[.5, 0., 0.]] * physical),
        cfg=SimpleNamespace(
            curriculum_semantics="upstream_v3", episode_length_s=20.,
            terrain=SimpleNamespace(terrain_generator=SimpleNamespace(size=(8., 8.))),
        ),
        _terrain_curriculum_slots=pool, _log_curriculum_metrics=lambda: None,
    )
    ids = torch.arange(physical)
    for episode in range(12):
        active_ids = ids * slots + pool.active
        positions = torch.tensor([4.5 if episode % 3 else 0., 4.5])
        env._robot.data.root_pos_w[:, 0] = positions
        reference.root_states[active_ids, 0] = positions
        original(reference, active_ids)
        actual(env, ids)
        torch.testing.assert_close(pool.levels.flatten(), reference.terrain_levels)
        torch.testing.assert_close(pool.active_levels, env._terrain.terrain_levels)
        if episode == 1:
            restored = TerrainCurriculumSlots(torch.zeros(physical, dtype=torch.long), slots)
            restored.load_state_dict(pool.state_dict())
            torch.testing.assert_close(restored.levels, pool.levels)
            torch.testing.assert_close(restored.active, pool.active)
            env._terrain_curriculum_slots = pool = restored
    assert pool.levels[1].eq(3).all()


def test_formal_entry_rejects_full_network_diagnostic_checkpoint():
    path = Path(__file__).resolve().parents[1] / "scripts/train.py"
    node = next(
        n for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "require_training_schedule"
    )
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    require = scope["require_training_schedule"]
    require(get_learning_cfg(num_envs=1024), smoke=False)
    require(SMOKE_CFG, smoke=True)
    with pytest.raises(ValueError, match="diagnostic checkpoints"):
        require(SMOKE_CFG, smoke=False)
