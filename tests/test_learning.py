# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import torch
from wmp_go2.constants import (
    ACTION_DIM,
    AMP_OBSERVATION_DIM,
    DEPTH_SHAPE,
    FORWARD_HEIGHT_DIM,
    HISTORY_DIM,
    OBSERVATION_DIM,
    OBSERVATION_LAYOUT,
    PROPRIOCEPTION_DIM,
    WORLD_MODEL_ACTION_DIM,
)
from wmp_go2.learning.amp import AMPMotionDataset
from wmp_go2.learning.checkpoint import load_checkpoint, save_checkpoint
from wmp_go2.learning.networks import ActorCriticWMP, AMPDiscriminator, DepthPredictor
from wmp_go2.learning.observations import HistoryBuffer, history_frame, split_observation
from wmp_go2.learning.ppo import AMPPPO
from wmp_go2.learning.world_model import WorldModel


def _assert_optimizer_step(module: torch.nn.Module, loss: torch.Tensor) -> None:
    optimizer = torch.optim.Adam(module.parameters(), lr=1.0e-3)
    before = [parameter.detach().clone() for parameter in module.parameters() if parameter.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in module.parameters())
    optimizer.step()
    after = [parameter.detach() for parameter in module.parameters() if parameter.requires_grad]
    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))


def _expert_dataset(path: Path) -> AMPMotionDataset:
    frames = 8
    np.savez(
        path,
        fps=np.float32(50.0),
        weight=np.float32(1.0),
        root_pos=np.zeros((frames, 3), np.float32),
        root_quat_wxyz=np.tile(np.asarray([[1, 0, 0, 0]], np.float32), (frames, 1)),
        root_lin_vel_b=np.random.default_rng(0).normal(size=(frames, 3)).astype(np.float32),
        root_ang_vel_b=np.random.default_rng(1).normal(size=(frames, 3)).astype(np.float32),
        joint_pos=np.random.default_rng(2).normal(size=(frames, 12)).astype(np.float32),
        joint_vel=np.random.default_rng(3).normal(size=(frames, 12)).astype(np.float32),
    )
    return AMPMotionDataset(path)


def test_observation_and_history_contracts():
    observation = torch.arange(2 * OBSERVATION_DIM, dtype=torch.float32).reshape(2, OBSERVATION_DIM)
    parts = split_observation(observation)
    assert {key: value.shape[-1] for key, value in parts.items()} == {
        "privileged": 53,
        "proprioception": 33,
        "previous_action": 12,
        "height": 187,
    }
    expected = torch.cat(
        (
            observation[:, OBSERVATION_LAYOUT["proprioception"]][:, :6],
            observation[:, OBSERVATION_LAYOUT["proprioception"]][:, 9:],
            observation[:, OBSERVATION_LAYOUT["previous_action"]],
        ),
        dim=-1,
    )
    torch.testing.assert_close(history_frame(observation), expected)
    history = HistoryBuffer(2, "cpu")
    assert history.append(observation).shape == (2, HISTORY_DIM)


def test_policy_amp_depth_and_world_model_optimize(tmp_path: Path):
    torch.manual_seed(1)
    batch = 4
    policy = ActorCriticWMP(
        world_feature_dim=64,
        history_hidden=(64, 32),
        actor_hidden=(64, 32),
        critic_hidden=(64, 32),
        world_hidden=(32,),
    )
    observation = torch.randn(batch, OBSERVATION_DIM)
    history = torch.randn(batch, HISTORY_DIM)
    world_feature = torch.randn(batch, 64)
    action = policy.act(observation, history, world_feature)
    value = policy.evaluate(observation, world_feature)
    assert action.shape == (batch, ACTION_DIM)
    assert value.shape == (batch, 1)
    _assert_optimizer_step(policy, action.square().mean() + value.square().mean())

    discriminator = AMPDiscriminator((64, 32))
    expert_state = torch.randn(batch, AMP_OBSERVATION_DIM)
    expert_next = torch.randn_like(expert_state)
    disc_loss = discriminator.transition_score(expert_state, expert_next).square().mean()
    disc_loss += discriminator.gradient_penalty(expert_state, expert_next)
    _assert_optimizer_step(discriminator, disc_loss)

    predictor = DepthPredictor((64, 32), base_channels=8)
    depth = predictor(torch.randn(batch, FORWARD_HEIGHT_DIM), torch.randn(batch, PROPRIOCEPTION_DIM))
    assert depth.shape == (batch, *DEPTH_SHAPE)
    _assert_optimizer_step(predictor, depth.square().mean())

    for feature_dim in (64, 512):
        world = WorldModel(feature_dim=feature_dim, stochastic_dim=8, hidden_dim=64)
        prop = torch.randn(batch, PROPRIOCEPTION_DIM)
        image = torch.rand(batch, *DEPTH_SHAPE)
        state, prior = world.observe_step(
            prop,
            image,
            torch.randn(batch, WORLD_MODEL_ACTION_DIM),
            torch.ones(batch),
        )
        assert world.policy_feature(state).shape == (batch, feature_dim)
        assert world.policy_feature(prior).shape == (batch, feature_dim)
        if feature_dim == 64:
            sequence = 3
            world_batch = {
                "prop": torch.randn(batch, sequence, PROPRIOCEPTION_DIM),
                "image": torch.rand(batch, sequence, *DEPTH_SHAPE),
                "action": torch.randn(batch, sequence, WORLD_MODEL_ACTION_DIM),
                "reward": torch.randn(batch, sequence),
                "is_first": torch.zeros(batch, sequence),
            }
            world_batch["is_first"][:, 0] = 1.0
            world_loss, metrics = world.loss(world_batch)
            assert all(torch.isfinite(value) for value in metrics.values())
            _assert_optimizer_step(world, world_loss)


def test_amp_ppo_update_is_finite(tmp_path: Path):
    torch.manual_seed(4)
    batch_size = 8
    policy = ActorCriticWMP(
        world_feature_dim=64,
        history_hidden=(64, 32),
        actor_hidden=(64, 32),
        critic_hidden=(64, 32),
        world_hidden=(32,),
    )
    discriminator = AMPDiscriminator((64, 32))
    algorithm = AMPPPO(policy, discriminator, replay_size=64)
    observation = torch.randn(batch_size, OBSERVATION_DIM)
    history = torch.randn(batch_size, HISTORY_DIM)
    feature = torch.randn(batch_size, 64)
    with torch.no_grad():
        action = policy.act(observation, history, feature)
        old_log_prob = policy.get_actions_log_prob(action)
    batch = {
        "observation": observation,
        "critic_observation": observation.clone(),
        "history": history,
        "world_feature": feature,
        "action": action,
        "old_log_prob": old_log_prob,
        "advantage": torch.randn(batch_size),
        "return": torch.randn(batch_size),
        "amp_state": torch.randn(batch_size, AMP_OBSERVATION_DIM),
        "amp_next_state": torch.randn(batch_size, AMP_OBSERVATION_DIM),
    }
    metrics = algorithm.update(batch, _expert_dataset(tmp_path / "expert.npz"))
    assert all(np.isfinite(value) for value in metrics.values())


def test_checkpoint_roundtrip_preserves_fixed_forward(tmp_path: Path):
    torch.manual_seed(8)
    modules = {
        "policy_value": ActorCriticWMP(
            world_feature_dim=64,
            history_hidden=(64, 32),
            actor_hidden=(64, 32),
            critic_hidden=(64, 32),
            world_hidden=(32,),
        ),
        "world_model": WorldModel(feature_dim=64, stochastic_dim=8, hidden_dim=64),
        "depth_predictor": DepthPredictor((64, 32), base_channels=8),
        "amp_discriminator": AMPDiscriminator((64, 32)),
    }
    optimizers = {name: torch.optim.Adam(module.parameters()) for name, module in modules.items()}
    inputs = (torch.randn(3, OBSERVATION_DIM), torch.randn(3, HISTORY_DIM), torch.randn(3, 64))
    with torch.no_grad():
        expected = modules["policy_value"].act_inference(*inputs)
    checkpoint = save_checkpoint(
        tmp_path / "model.pt", config={"world_feature_dim": 64}, iteration=17, modules=modules, optimizers=optimizers
    )
    clones = {
        "policy_value": ActorCriticWMP(
            world_feature_dim=64,
            history_hidden=(64, 32),
            actor_hidden=(64, 32),
            critic_hidden=(64, 32),
            world_hidden=(32,),
        ),
        "world_model": WorldModel(feature_dim=64, stochastic_dim=8, hidden_dim=64),
        "depth_predictor": DepthPredictor((64, 32), base_channels=8),
        "amp_discriminator": AMPDiscriminator((64, 32)),
    }
    payload = load_checkpoint(checkpoint, modules=clones)
    assert payload["iteration"] == 17
    assert payload["replay_buffers_included"] is False
    with torch.no_grad():
        actual = clones["policy_value"].act_inference(*inputs)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
