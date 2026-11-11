# SPDX-License-Identifier: Apache-2.0
"""离散 RSSM 世界模型：观测重建、symlog 和分离梯度的平衡 KL。"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from wmp_lab.constants import PROPRIOCEPTION_DIM, WORLD_MODEL_ACTION_DIM

from .networks import mlp


class RSSMState(NamedTuple):
    deter: torch.Tensor
    stoch: torch.Tensor
    logits: torch.Tensor


def symlog(value: torch.Tensor) -> torch.Tensor:
    return torch.sign(value) * torch.log1p(value.abs())


class DepthEncoder(nn.Module):
    def __init__(self, output_dim: int, channels: int = 32):
        super().__init__()
        layers = []
        previous = 1
        for channel in (channels, channels * 2, channels * 4, channels * 8):
            layers.extend((nn.Conv2d(previous, channel, 4, 2, 1), nn.SiLU()))
            previous = channel
        layers.extend((nn.Flatten(), nn.Linear(4 * 4 * previous, output_dim), nn.SiLU()))
        self.layers = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers((image - 0.5).permute(0, 3, 1, 2))


class DepthDecoder(nn.Module):
    def __init__(self, feature_dim: int, base_channels: int = 32):
        super().__init__()
        self.channels = base_channels * 8
        self.seed = nn.Linear(feature_dim, 4 * 4 * self.channels)
        layers = []
        previous = self.channels
        for channel in (base_channels * 4, base_channels * 2, base_channels, 1):
            layers.append(nn.ConvTranspose2d(previous, channel, 4, 2, 1))
            if channel != 1:
                layers.append(nn.SiLU())
            previous = channel
        self.layers = nn.Sequential(*layers)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        value = self.seed(feature).reshape(-1, self.channels, 4, 4)
        return self.layers(value).permute(0, 2, 3, 1).tanh() * 0.5


class RSSM(nn.Module):
    def __init__(
        self, action_dim: int, embed_dim: int, deter_dim: int = 512, stoch_dim: int = 32, categories: int = 32
    ):
        super().__init__()
        self.deter_dim, self.stoch_dim, self.categories = deter_dim, stoch_dim, categories
        self.action_encoder = mlp(stoch_dim * categories + action_dim, (deter_dim,), deter_dim, "silu")
        self.gru = nn.GRUCell(deter_dim, deter_dim)
        self.prior = mlp(deter_dim, (deter_dim,), stoch_dim * categories, "silu")
        self.posterior = mlp(deter_dim + embed_dim, (deter_dim,), stoch_dim * categories, "silu")
        self.initial_deter = nn.Parameter(torch.zeros(1, deter_dim))

    def initial(self, batch_size: int, device: torch.device) -> RSSMState:
        deter = self.initial_deter.tanh().expand(batch_size, -1).to(device)
        logits = self.prior(deter).reshape(batch_size, self.stoch_dim, self.categories)
        stoch = self._sample(logits, False)
        return RSSMState(deter, stoch, logits)

    def _sample(self, logits: torch.Tensor, sample: bool) -> torch.Tensor:
        probs = logits.softmax(-1) * 0.99 + 0.01 / self.categories
        flat = probs.reshape(-1, self.categories)
        indices = torch.multinomial(flat, 1).squeeze(-1) if sample else flat.argmax(-1)
        onehot = F.one_hot(indices, self.categories).to(probs.dtype).reshape_as(probs)
        return onehot + probs - probs.detach()

    def forward(
        self,
        deter: torch.Tensor,
        stoch: torch.Tensor,
        action: torch.Tensor,
        embed: torch.Tensor,
        is_first: torch.Tensor,
        sample: bool = True,
    ):
        initial_deter = self.initial_deter.tanh().expand_as(deter)
        initial_logits = self.prior(initial_deter).reshape(-1, self.stoch_dim, self.categories)
        first = is_first.to(torch.bool).unsqueeze(-1)
        deter = torch.where(first, initial_deter, deter)
        stoch = torch.where(first.unsqueeze(-1), self._sample(initial_logits, False), stoch)
        action = torch.where(first, torch.zeros_like(action), action)
        deter = self.gru(self.action_encoder(torch.cat((stoch.flatten(1), action), -1)), deter)
        prior_logits = self.prior(deter).reshape(-1, self.stoch_dim, self.categories)
        logits = self.posterior(torch.cat((deter, embed), -1)).reshape(-1, self.stoch_dim, self.categories)
        return deter, self._sample(logits, sample), prior_logits, logits

    def obs_step(self, previous, action, embed, is_first, sample=True):
        if previous is None:
            previous = self.initial(action.shape[0], action.device)
        deter, stoch, prior_logits, logits = self(previous.deter, previous.stoch, action, embed, is_first, sample)
        return RSSMState(deter, stoch, logits), RSSMState(deter, self._sample(prior_logits, sample), prior_logits)


class WorldModel(nn.Module):
    def __init__(self, feature_dim=512, stochastic_dim=32, hidden_dim=512, categories=32, reward_loss_scale=0.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.reward_loss_scale = reward_loss_scale
        units = 1024 if hidden_dim > 64 else hidden_dim
        layers = (units,) * (5 if hidden_dim > 64 else 2)
        self.prop_encoder = mlp(PROPRIOCEPTION_DIM, layers, hidden_dim // 2, "silu")
        self.depth_encoder = DepthEncoder(hidden_dim // 2, 32 if hidden_dim > 64 else 8)
        self.rssm = RSSM(WORLD_MODEL_ACTION_DIM, hidden_dim, feature_dim, stochastic_dim, categories)
        joint_dim = feature_dim + stochastic_dim * categories
        self.prop_decoder = mlp(joint_dim, layers, PROPRIOCEPTION_DIM, "silu")
        self.depth_decoder = DepthDecoder(joint_dim, 32 if hidden_dim > 64 else 8)
        self.reward_head = mlp(joint_dim, (hidden_dim, hidden_dim), 1, "silu")

    def encode(self, prop, image):
        return torch.cat((self.prop_encoder(symlog(prop)), self.depth_encoder(image)), -1)

    @staticmethod
    def feature(state):
        return torch.cat((state.deter, state.stoch.flatten(1)), -1)

    @staticmethod
    def policy_feature(state):
        return state.deter

    def observe_step(self, prop, image, action_history, is_first, previous=None, sample=True):
        return self.rssm.obs_step(previous, action_history, self.encode(prop, image), is_first, sample)

    def loss(self, batch):
        previous = None
        features, dynamics_kl, representation_kl = [], [], []
        for step in range(batch["prop"].shape[1]):
            post, prior = self.observe_step(
                batch["prop"][:, step],
                batch["image"][:, step],
                batch["action"][:, step],
                batch["is_first"][:, step],
                previous,
            )
            features.append(self.feature(post))
            p = post.logits.softmax(-1) * 0.99 + 0.01 / self.rssm.categories
            q = prior.logits.softmax(-1) * 0.99 + 0.01 / self.rssm.categories
            dynamics_kl.append((p.detach() * (p.detach().log() - q.log())).sum((-1, -2)))
            representation_kl.append((p * (p.log() - q.detach().log())).sum((-1, -2)))
            previous = post
        feature = torch.stack(features, 1)
        flat = feature.flatten(0, 1)
        prop_loss = F.mse_loss(self.prop_decoder(flat).reshape_as(batch["prop"]), symlog(batch["prop"]))
        image_loss = F.mse_loss(self.depth_decoder(flat).reshape_as(batch["image"]), batch["image"])
        reward_loss = F.mse_loss(self.reward_head(flat).reshape_as(batch["reward"]), symlog(batch["reward"]))
        dyn_loss = torch.stack(dynamics_kl).clamp_min(1.0).mean()
        rep_loss = torch.stack(representation_kl).clamp_min(1.0).mean()
        total = prop_loss + image_loss + self.reward_loss_scale * reward_loss + 0.5 * dyn_loss + 0.1 * rep_loss
        metrics = dict(
            world_loss=total,
            prop_loss=prop_loss,
            image_loss=image_loss,
            reward_loss=reward_loss,
            kl_loss=dyn_loss,
            representation_loss=rep_loss,
        )
        return total, {key: value.detach() for key, value in metrics.items()}
