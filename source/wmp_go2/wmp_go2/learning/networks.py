# SPDX-License-Identifier: Apache-2.0
"""WMP policy, AMP discriminator, and forward-map depth predictor."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Normal

from wmp_go2.constants import (
    ACTION_DIM,
    AMP_OBSERVATION_DIM,
    DEPTH_SHAPE,
    FORWARD_HEIGHT_DIM,
    HISTORY_DIM,
    OBSERVATION_DIM,
    PRIVILEGED_DIM,
    PROPRIOCEPTION_DIM,
)


def _activation(name: str) -> type[nn.Module]:
    choices = {"elu": nn.ELU, "relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}
    try:
        return choices[name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported activation: {name}") from error


def mlp(input_dim: int, hidden: Sequence[int], output_dim: int, activation: str = "elu") -> nn.Sequential:
    act = _activation(activation)
    dims = (input_dim, *hidden, output_dim)
    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(dims, dims[1:], strict=False)):
        layers.append(nn.Linear(in_dim, out_dim))
        if index < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class ActorCriticWMP(nn.Module):
    """History-conditioned WMP actor and privileged critic.

    The command remains the original observation slice ``privileged_dim+6:+9``.
    The history latent includes its last three dimensions as the velocity estimate,
    matching the original loss semantics.
    """

    def __init__(
        self,
        observation_dim: int = OBSERVATION_DIM,
        action_dim: int = ACTION_DIM,
        history_dim: int = HISTORY_DIM,
        world_feature_dim: int = 512,
        history_hidden: Sequence[int] = (256, 128),
        actor_hidden: Sequence[int] = (256, 128, 64),
        critic_hidden: Sequence[int] = (512, 256, 128),
        world_hidden: Sequence[int] = (64, 64),
        history_latent_dim: int = 35,
        world_latent_dim: int = 32,
        init_noise_std: float = 1.0,
    ):
        super().__init__()
        if history_latent_dim < 3:
            raise ValueError("history latent needs three velocity-estimate dimensions")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.history_encoder = mlp(history_dim, history_hidden, history_latent_dim)
        self.actor_world_encoder = mlp(world_feature_dim, world_hidden, world_latent_dim)
        self.critic_world_encoder = mlp(world_feature_dim, world_hidden, world_latent_dim)
        self.actor = mlp(history_latent_dim + 3 + world_latent_dim, actor_hidden, action_dim)
        self.critic = mlp(observation_dim + world_latent_dim, critic_hidden, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(init_noise_std)))
        self.distribution: Normal | None = None

    def _actor_input(
        self, observations: torch.Tensor, history: torch.Tensor, world_feature: torch.Tensor
    ) -> torch.Tensor:
        if observations.shape[-1] != self.observation_dim:
            raise ValueError(f"actor expected {self.observation_dim}-D observation")
        history_latent = self.history_encoder(history)
        command = observations[..., PRIVILEGED_DIM + 6 : PRIVILEGED_DIM + 9]
        world_latent = self.actor_world_encoder(world_feature)
        return torch.cat((history_latent, command, world_latent), dim=-1)

    def update_distribution(
        self, observations: torch.Tensor, history: torch.Tensor, world_feature: torch.Tensor
    ) -> Normal:
        mean = self.actor(self._actor_input(observations, history, world_feature))
        std = self.log_std.exp().expand_as(mean)
        self.distribution = Normal(mean, std)
        return self.distribution

    def act(self, observations: torch.Tensor, history: torch.Tensor, world_feature: torch.Tensor) -> torch.Tensor:
        return self.update_distribution(observations, history, world_feature).sample()

    def act_inference(
        self, observations: torch.Tensor, history: torch.Tensor, world_feature: torch.Tensor
    ) -> torch.Tensor:
        return self.actor(self._actor_input(observations, history, world_feature))

    def evaluate(self, critic_observations: torch.Tensor, world_feature: torch.Tensor) -> torch.Tensor:
        latent = self.critic_world_encoder(world_feature)
        return self.critic(torch.cat((critic_observations, latent), dim=-1))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("call act/update_distribution before requesting log probability")
        return self.distribution.log_prob(actions).sum(-1)

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("distribution has not been initialized")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("distribution has not been initialized")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("distribution has not been initialized")
        return self.distribution.entropy().sum(-1)

    def velocity_estimate(self, history: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(history)[..., -3:]


class AMPDiscriminator(nn.Module):
    def __init__(
        self,
        hidden_dims: Sequence[int] = (1024, 512),
        amp_reward_coef: float = 0.01,
        task_reward_lerp: float = 0.3,
    ):
        super().__init__()
        self.input_dim = AMP_OBSERVATION_DIM * 2
        self.trunk = mlp(self.input_dim, hidden_dims[:-1], hidden_dims[-1], activation="relu")
        self.head = nn.Linear(hidden_dims[-1], 1)
        self.amp_reward_coef = amp_reward_coef
        self.task_reward_lerp = task_reward_lerp

    def forward(self, transition: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(transition))

    def transition_score(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return self(torch.cat((state, next_state), dim=-1))

    def gradient_penalty(
        self, expert_state: torch.Tensor, expert_next_state: torch.Tensor, coefficient: float = 10.0
    ) -> torch.Tensor:
        transition = torch.cat((expert_state, expert_next_state), dim=-1).detach().requires_grad_(True)
        score = self(transition)
        (gradient,) = torch.autograd.grad(score.sum(), transition, create_graph=True)
        return coefficient * gradient.norm(2, dim=-1).square().mean()

    @torch.no_grad()
    def predict_amp_reward(
        self, state: torch.Tensor, next_state: torch.Tensor, task_reward: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score = self.transition_score(state, next_state)
        reward = self.amp_reward_coef * (1.0 - 0.25 * (score - 1.0).square()).clamp_min(0.0)
        # WMP's implementation names this a lerp but adds the task reward.
        if self.task_reward_lerp > 0.0:
            reward = reward + task_reward.unsqueeze(-1)
        return reward.squeeze(-1), score


class ImageChannelLayerNorm(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1.0e-3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.norm(value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DepthPredictor(nn.Module):
    """Predict normalized ``64x64x1`` depth from the 525-D forward map and prop."""

    def __init__(self, hidden_dims: Sequence[int] = (256, 128), base_channels: int = 32):
        super().__init__()
        seed_channels = base_channels * 8
        self.encoder = mlp(FORWARD_HEIGHT_DIM + PROPRIOCEPTION_DIM, hidden_dims, 4 * 4 * seed_channels)
        channels = (seed_channels, base_channels * 4, base_channels * 2, base_channels, 1)
        layers: list[nn.Module] = []
        for index, (in_channels, out_channels) in enumerate(zip(channels, channels[1:], strict=False)):
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1))
            if index < len(channels) - 2:
                layers.extend((ImageChannelLayerNorm(out_channels), nn.ELU()))
        self.decoder = nn.Sequential(*layers)

    def forward(self, forward_height_map: torch.Tensor, proprioception: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(torch.cat((forward_height_map, proprioception), dim=-1))
        decoded = self.decoder(encoded.reshape(encoded.shape[0], -1, 4, 4)).sigmoid()
        output = decoded.permute(0, 2, 3, 1)
        if output.shape[1:] != DEPTH_SHAPE:
            raise RuntimeError(f"depth predictor produced {output.shape}, expected [N, {DEPTH_SHAPE}]")
        return output
