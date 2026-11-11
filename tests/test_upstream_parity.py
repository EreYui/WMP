# SPDX-License-Identifier: Apache-2.0
"""执行固定上游源码进行差分，验证训练梯度和更新，而非仅检查接口可运行。"""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from wmp_lab.learning.amp import RunningMeanStd
from wmp_lab.learning.networks import ActorCriticWMP, AMPDiscriminator, DepthPredictor
from wmp_lab.learning.ppo import AMPPPO
from wmp_lab.learning.task_rewards import tracking_velocity
from wmp_lab.learning.upstream_dreamer import networks, tools
from wmp_lab.learning.world_model import WorldModel

UPSTREAM = Path(__file__).resolve().parents[2] / "wmp_go1"


def original(relative, names, **namespace):
    tree = ast.parse((UPSTREAM / relative).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef | ast.FunctionDef) and node.name in names]
    scope = dict(torch=torch, nn=nn, np=np, Normal=torch.distributions.Normal, **namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(UPSTREAM / relative), "exec"), scope)
    return scope


def policy_pair():
    scope = original("rsl_rl/modules/actor_critic_wmp.py", {"ActorCriticWMP", "get_activation"})
    ours = ActorCriticWMP(
        world_feature_dim=64, history_hidden=(32,), actor_hidden=(32,), critic_hidden=(32,), world_hidden=(32,)
    )
    reference = scope["ActorCriticWMP"](
        285,
        285,
        12,
        encoder_hidden_dims=[32],
        wm_encoder_hidden_dims=[32],
        actor_hidden_dims=[32],
        critic_hidden_dims=[32],
        height_dim=187,
        privileged_dim=53,
        history_dim=210,
        wm_feature_dim=64,
        latent_dim=35,
        wm_latent_dim=32,
    )

    def rename(name):
        return name.replace("actor_world_encoder", "wm_feature_encoder").replace(
            "critic_world_encoder", "critic_wm_feature_encoder"
        )

    reference.load_state_dict({rename(key): value for key, value in ours.state_dict().items()})
    return ours, reference, rename


def test_actor_distribution_values_and_entropy_gradient_match_source():
    torch.manual_seed(3)
    ours, reference, _ = policy_pair()
    obs, history, feature = torch.randn(5, 285), torch.randn(5, 210), torch.randn(5, 64)
    reference.act(obs, history, feature)
    ours.update_distribution(obs, history, feature)
    torch.testing.assert_close(ours.action_mean, reference.action_mean, rtol=0, atol=0)
    torch.testing.assert_close(ours.action_std, reference.action_std, rtol=0, atol=0)
    torch.testing.assert_close(ours.evaluate(obs, feature), reference.evaluate(obs, feature), rtol=0, atol=0)
    ours.entropy.mean().backward()
    reference.entropy.mean().backward()
    torch.testing.assert_close(ours.std.grad, reference.std.grad, rtol=0, atol=0)


def test_joint_ppo_amp_update_matches_original_optimizer_step():
    torch.manual_seed(4)
    ours, reference, rename = policy_pair()
    disc = AMPDiscriminator((32, 16))
    disc_cls = original("rsl_rl/algorithms/amp_discriminator.py", {"AMPDiscriminator"}, autograd=torch.autograd)[
        "AMPDiscriminator"
    ]
    ref_disc = disc_cls(60, 0.01, [32, 16], "cpu", task_reward_lerp=0.3)
    ref_disc.load_state_dict({key.replace("head", "amp_linear"): val for key, val in disc.state_dict().items()})
    count = 8
    observation, history, feature = torch.randn(count, 285), torch.randn(count, 210), torch.randn(count, 64)
    with torch.no_grad():
        action = ours.act(observation, history, feature)
        value = ours.evaluate(observation, feature).squeeze(-1)
        log_prob = ours.get_actions_log_prob(action)
        old_mean, old_std = ours.action_mean.clone(), ours.action_std.clone()
    batch = dict(
        observation=observation,
        critic_observation=observation.clone(),
        history=history,
        world_feature=feature,
        action=action,
        old_log_prob=log_prob,
        old_value=value,
        old_mean=old_mean,
        old_std=old_std,
        advantage=torch.randn(count),
        amp_state=torch.randn(count, 30),
        amp_next_state=torch.randn(count, 30),
    )
    batch["return"] = torch.randn(count)
    policy_state, policy_next = torch.randn(count, 30) * 20, torch.randn(count, 30) * 20
    expert_state, expert_next = torch.randn(count, 30) * 20, torch.randn(count, 30) * 20

    class Replay:
        def __init__(self, *args):
            pass

        def feed_forward_generator(self, *args):
            yield policy_state, policy_next

        def sample(self, *args):
            return policy_state, policy_next

    class Expert:
        def feed_forward_generator(self, *args):
            yield expert_state, expert_next

        def sample_transitions(self, *args):
            return expert_state, expert_next, torch.zeros(count)

    class Normalizer(RunningMeanStd):
        def normalize_torch(self, value, device):
            return self.normalize(value)

        def update(self, value):
            super().update(torch.as_tensor(value))

    class Storage:
        num_envs, num_transitions_per_env = count, 1

        def mini_batch_generator(self, *args):
            yield (
                observation,
                observation,
                action,
                history,
                feature,
                value[:, None],
                batch["advantage"][:, None],
                batch["return"][:, None],
                log_prob[:, None],
                old_mean,
                old_std,
                (None, None),
                None,
            )

        def clear(self):
            pass

    ref_cls = original(
        "rsl_rl/algorithms/amp_ppo.py",
        {"AMPPPO"},
        optim=torch.optim,
        ActorCritic=nn.Module,
        ReplayBuffer=Replay,
        RolloutStorage=SimpleNamespace(Transition=lambda: SimpleNamespace()),
    )["AMPPPO"]
    normalizer = Normalizer(30)
    ref_alg = ref_cls(
        reference,
        ref_disc,
        Expert(),
        normalizer,
        num_learning_epochs=1,
        num_mini_batches=1,
        entropy_coef=0.01,
        schedule="adaptive",
        min_std=0.02,
    )
    ref_alg.storage = Storage()
    ref_alg.amp_storage = Replay()
    algorithm = AMPPPO(ours, disc, replay_size=32)
    algorithm.amp_replay = Replay()
    result = algorithm.update(batch, Expert(), insert_replay=False)
    ref_metrics = ref_alg.update()
    for name, param in ours.state_dict().items():
        torch.testing.assert_close(param, reference.state_dict()[rename(name)], rtol=1e-5, atol=2e-7)
    for name, param in disc.state_dict().items():
        torch.testing.assert_close(
            param, ref_disc.state_dict()[name.replace("head", "amp_linear")], rtol=1e-5, atol=2e-7
        )
    for key in ("mean", "var", "count"):
        torch.testing.assert_close(getattr(algorithm.amp_normalizer, key), getattr(normalizer, key))
    assert np.isclose(result["discriminator_loss"], ref_metrics[3] + ref_metrics[4], rtol=1e-7)
    assert [group["lr"] for group in algorithm.optimizer.param_groups] == [
        group["lr"] for group in ref_alg.optimizer.param_groups
    ]


def test_world_model_loss_and_all_gradients_match_original_train():
    torch.manual_seed(8)
    world = WorldModel(feature_dim=64, stochastic_dim=8, hidden_dim=64, categories=8)
    ref_cls = original(
        "dreamer/models.py", {"WorldModel"}, tools=tools, networks=networks, to_np=lambda value: value.detach().numpy()
    )["WorldModel"]
    ref = ref_cls.__new__(ref_cls)
    nn.Module.__init__(ref)
    ref._config = SimpleNamespace(
        precision=32, kl_free=1.0, dyn_scale=0.5, rep_scale=0.1, grad_heads=["decoder", "reward"], device="cpu"
    )
    ref._use_amp, ref.device = False, "cpu"
    ref.encoder, ref.dynamics = copy.deepcopy(world.encoder), copy.deepcopy(world.rssm.core)
    ref.heads = nn.ModuleDict({"decoder": copy.deepcopy(world.decoder), "reward": copy.deepcopy(world.reward_head)})
    ref._scales = {"reward": 0.0, "image": 1.0}
    captured = []

    def capture(loss, parameters):
        captured.append(loss.detach())
        loss.backward()
        return {}

    ref._model_opt = capture
    batch = dict(
        prop=torch.randn(2, 4, 33),
        image=torch.rand(2, 4, 64, 64, 1) - 0.5,
        action=torch.randn(2, 4, 60),
        reward=torch.randn(2, 4),
        is_first=torch.zeros(2, 4),
    )
    batch["is_first"][:, 0] = 1
    batch["is_first"][0, 2] = 1
    torch.manual_seed(17)
    loss, _ = world.loss({key: value.clone() for key, value in batch.items()})
    loss.backward()
    torch.manual_seed(17)
    ref._train({key: value.clone() for key, value in batch.items()})
    torch.testing.assert_close(loss.detach(), captured[0], rtol=0, atol=0)

    def rename(name):
        return (
            name.replace("rssm.core", "dynamics")
            .replace("decoder.", "heads.decoder.")
            .replace("reward_head.", "heads.reward.")
        )

    reference = dict(ref.named_parameters())
    for name, parameter in world.named_parameters():
        expected = reference[rename(name)].grad
        assert (parameter.grad is None) == (expected is None)
        if expected is not None:
            torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)


def test_depth_predictor_matches_upstream_layout_and_output():
    cls = original("rsl_rl/modules/depth_predictor.py", {"DepthPredictor", "ImgChLayerNorm"}, tools=tools)[
        "DepthPredictor"
    ]
    torch.manual_seed(3)
    ours = DepthPredictor((64, 32), base_channels=8)
    reference = cls(encoder_hidden_dims=[64, 32], depth=8)
    reference.load_state_dict({key.replace("decoder.", "layers."): value for key, value in ours.state_dict().items()})
    forward, prop = torch.randn(3, 525), torch.randn(3, 33)
    torch.testing.assert_close(ours(forward, prop), reference(forward, prop), rtol=0, atol=0)


def test_obstacle_velocity_reward_matches_original_at_overspeed():
    function = next(
        node
        for node in ast.walk(ast.parse((UPSTREAM / "legged_gym/envs/base/legged_robot.py").read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "_reward_tracking_lin_vel"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "upstream tracking reward", "exec"), namespace)
    velocity = torch.tensor([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.8, 0.5, 0.0], [2.0, 0.0, 0.0]])
    command = torch.tensor([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.3, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    cfg = SimpleNamespace(rewards=SimpleNamespace(lin_vel_clip=0.1, tracking_sigma=0.25))
    env = SimpleNamespace(
        base_lin_vel=velocity,
        commands=command,
        cfg=cfg,
        gap_start_idx=0,
        gap_end_idx=1,
        bream_start_idx=1,
        bream_end_idx=2,
        continuous_gap_start_idx=2,
        continuous_gap_end_idx=3,
        climb_start_idx=3,
        climb_end_idx=4,
    )
    expected = namespace["_reward_tracking_lin_vel"](env)
    actual = tracking_velocity(velocity, command, torch.tensor([True, True, True, True, False]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual[0] > 0.9 and actual[-1] < 0.001
