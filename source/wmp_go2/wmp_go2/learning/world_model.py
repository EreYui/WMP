# SPDX-License-Identifier: Apache-2.0
"""Compact Dreamer-style recurrent state-space world model used by WMP."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn
from torch.distributions import Independent, Normal, kl_divergence

from wmp_go2.constants import DEPTH_SHAPE, PROPRIOCEPTION_DIM, WORLD_MODEL_ACTION_DIM
from wmp_go2.learning.networks import mlp


@dataclass
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    def detach(self) -> RSSMState:
        return RSSMState(*(value.detach() for value in (self.deter, self.stoch, self.mean, self.std)))


class DepthEncoder(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(4 * 4 * 64, output_dim),
            nn.SiLU(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-3:] != DEPTH_SHAPE:
            raise ValueError(f"world model expects NHWC depth shape {DEPTH_SHAPE}, got {image.shape[-3:]}")
        leading = image.shape[:-3]
        flat = image.reshape(-1, *DEPTH_SHAPE).permute(0, 3, 1, 2)
        return self.layers(flat).reshape(*leading, -1)


class DepthDecoder(nn.Module):
    def __init__(self, feature_dim: int, base_channels: int = 32):
        super().__init__()
        self.seed = nn.Linear(feature_dim, 4 * 4 * base_channels * 8)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels, 1, 4, 2, 1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        leading = feature.shape[:-1]
        value = self.seed(feature.reshape(-1, feature.shape[-1])).reshape(-1, 256, 4, 4)
        return self.layers(value).sigmoid().permute(0, 2, 3, 1).reshape(*leading, *DEPTH_SHAPE)


class RSSM(nn.Module):
    def __init__(self, action_dim: int, embed_dim: int, deter_dim: int = 512, stoch_dim: int = 32):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.action_encoder = mlp(stoch_dim + action_dim, (deter_dim,), deter_dim, activation="silu")
        self.gru = nn.GRUCell(deter_dim, deter_dim)
        self.prior = mlp(deter_dim, (deter_dim,), stoch_dim * 2, activation="silu")
        self.posterior = mlp(deter_dim + embed_dim, (deter_dim,), stoch_dim * 2, activation="silu")

    def initial(self, batch_size: int, device: torch.device | str) -> RSSMState:
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        mean = torch.zeros(batch_size, self.stoch_dim, device=device)
        std = torch.ones_like(mean)
        return RSSMState(deter, mean, mean, std)

    @staticmethod
    def _stats(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_std = raw.chunk(2, -1)
        return mean, functional.softplus(raw_std + 0.54) + 0.1

    def obs_step(
        self,
        previous: RSSMState | None,
        action: torch.Tensor,
        embed: torch.Tensor,
        is_first: torch.Tensor,
    ) -> tuple[RSSMState, RSSMState]:
        if previous is None:
            previous = self.initial(action.shape[0], action.device)
        reset = (1.0 - is_first.float()).unsqueeze(-1)
        prev_stoch = previous.stoch * reset
        prev_deter = previous.deter * reset
        action = action * reset
        hidden = self.action_encoder(torch.cat((prev_stoch, action), -1))
        deter = self.gru(hidden, prev_deter)
        prior_mean, prior_std = self._stats(self.prior(deter))
        prior = RSSMState(deter, prior_mean + prior_std * torch.randn_like(prior_mean), prior_mean, prior_std)
        post_mean, post_std = self._stats(self.posterior(torch.cat((deter, embed), -1)))
        posterior = RSSMState(deter, post_mean + post_std * torch.randn_like(post_mean), post_mean, post_std)
        return posterior, prior

    @staticmethod
    def distribution(state: RSSMState) -> Independent:
        return Independent(Normal(state.mean, state.std), 1)


class WorldModel(nn.Module):
    """Dreamer observation model with WMP's fixed prop/depth/action inputs."""

    def __init__(self, feature_dim: int = 512, stochastic_dim: int = 32, hidden_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        embed_dim = hidden_dim
        self.prop_encoder = mlp(PROPRIOCEPTION_DIM, (hidden_dim,), embed_dim // 2, activation="silu")
        self.depth_encoder = DepthEncoder(embed_dim // 2)
        self.rssm = RSSM(WORLD_MODEL_ACTION_DIM, embed_dim, feature_dim, stochastic_dim)
        joint_feature_dim = feature_dim + stochastic_dim
        self.prop_decoder = mlp(joint_feature_dim, (hidden_dim,), PROPRIOCEPTION_DIM, activation="silu")
        self.depth_decoder = DepthDecoder(joint_feature_dim)
        self.reward_head = mlp(joint_feature_dim, (hidden_dim, hidden_dim), 1, activation="silu")

    def encode(self, prop: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.prop_encoder(prop), self.depth_encoder(image)), -1)

    @staticmethod
    def feature(state: RSSMState) -> torch.Tensor:
        return torch.cat((state.deter, state.stoch), -1)

    @staticmethod
    def policy_feature(state: RSSMState) -> torch.Tensor:
        return state.deter

    def observe_step(
        self,
        prop: torch.Tensor,
        image: torch.Tensor,
        action_history: torch.Tensor,
        is_first: torch.Tensor,
        previous: RSSMState | None = None,
    ) -> tuple[RSSMState, RSSMState]:
        embed = self.encode(prop, image)
        return self.rssm.obs_step(previous, action_history, embed, is_first)

    def loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute sequence reconstruction, reward, and balanced KL losses."""
        required = {"prop", "image", "action", "reward", "is_first"}
        if required != required.intersection(batch):
            raise KeyError(f"world-model batch missing {sorted(required.difference(batch))}")
        batch_size, sequence_length = batch["prop"].shape[:2]
        previous: RSSMState | None = None
        posterior_states: list[RSSMState] = []
        prior_states: list[RSSMState] = []
        for step in range(sequence_length):
            posterior, prior = self.observe_step(
                batch["prop"][:, step],
                batch["image"][:, step],
                batch["action"][:, step],
                batch["is_first"][:, step],
                previous,
            )
            posterior_states.append(posterior)
            prior_states.append(prior)
            previous = posterior
        feature = torch.stack([self.feature(state) for state in posterior_states], dim=1)
        prop_prediction = self.prop_decoder(feature)
        image_prediction = self.depth_decoder(feature)
        reward_prediction = self.reward_head(feature).squeeze(-1)
        prop_loss = functional.mse_loss(prop_prediction, batch["prop"])
        image_loss = functional.mse_loss(image_prediction, batch["image"])
        reward_loss = functional.mse_loss(reward_prediction, batch["reward"])
        posterior_dist = torch.stack(
            [
                kl_divergence(self.rssm.distribution(post), self.rssm.distribution(prior))
                for post, prior in zip(posterior_states, prior_states, strict=True)
            ],
            dim=1,
        )
        kl_loss = posterior_dist.clamp_min(1.0).mean()
        total = prop_loss + image_loss + reward_loss + 0.5 * kl_loss
        metrics = {
            "world_loss": total.detach(),
            "prop_loss": prop_loss.detach(),
            "image_loss": image_loss.detach(),
            "reward_loss": reward_loss.detach(),
            "kl_loss": kl_loss.detach(),
        }
        if feature.shape != (batch_size, sequence_length, self.feature_dim + self.rssm.stoch_dim):
            raise RuntimeError("RSSM feature shape contract was violated")
        return total, metrics
