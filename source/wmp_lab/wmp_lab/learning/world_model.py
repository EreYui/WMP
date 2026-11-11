# SPDX-License-Identifier: Apache-2.0
"""上游 Dreamer 的接口适配；网络、分布与损失直接复用固定源码。"""

from typing import NamedTuple

import torch
from torch import nn

from wmp_lab.constants import PROPRIOCEPTION_DIM, WORLD_MODEL_ACTION_DIM

from .inference import inference_chunks
from .upstream_dreamer import networks, tools

symlog = tools.symlog


class RSSMState(NamedTuple):
    deter: torch.Tensor
    stoch: torch.Tensor
    logits: torch.Tensor


class RSSM(nn.Module):
    def __init__(self, action_dim, embed_dim, deter_dim=512, stoch_dim=32, categories=32, hidden_dim=512):
        super().__init__()
        self.deter_dim, self.stoch_dim, self.categories = deter_dim, stoch_dim, categories
        self.core = networks.RSSM(
            stoch=stoch_dim,
            deter=deter_dim,
            hidden=hidden_dim,
            discrete=categories,
            act="SiLU",
            norm=True,
            unimix_ratio=0.01,
            initial="learned",
            num_actions=action_dim,
            embed=embed_dim,
            device="cpu",
        )

    @staticmethod
    def _state(value):
        return RSSMState(value["deter"], value["stoch"], value["logit"])

    @staticmethod
    def _dict(value):
        return {"deter": value.deter, "stoch": value.stoch, "logit": value.logits}

    def initial(self, batch_size, device):
        self.core._device = str(device)
        return self._state(self.core.initial(batch_size))

    def obs_step(self, previous, action, embed, is_first, sample=True):
        self.core._device = str(action.device)
        post, prior = self.core.obs_step(
            None if previous is None else self._dict(previous), action, embed, is_first, sample=sample
        )
        return self._state(post), self._state(prior)


class WorldModel(nn.Module):
    def __init__(self, feature_dim=512, stochastic_dim=32, hidden_dim=512, categories=32, reward_loss_scale=0.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.reward_loss_scale = reward_loss_scale
        self.shapes = {"prop": (PROPRIOCEPTION_DIM,), "image": (64, 64, 1)}
        options = dict(
            mlp_keys=".*",
            cnn_keys="image",
            act="SiLU",
            norm=True,
            cnn_depth=32 if hidden_dim > 64 else 8,
            kernel_size=4,
            minres=4,
            mlp_layers=5 if hidden_dim > 64 else 2,
            mlp_units=1024 if hidden_dim > 64 else hidden_dim,
            use_camera=True,
        )
        self.encoder = networks.MultiEncoder(self.shapes, symlog_inputs=True, **options)
        self.rssm = RSSM(
            WORLD_MODEL_ACTION_DIM, self.encoder.outdim, feature_dim, stochastic_dim, categories, hidden_dim
        )
        joint_dim = feature_dim + stochastic_dim * categories
        self.decoder = networks.MultiDecoder(
            joint_dim,
            self.shapes,
            cnn_sigmoid=False,
            image_dist="mse",
            vector_dist="symlog_mse",
            outscale=1.0,
            **options,
        )
        self.reward_head = networks.MLP(
            joint_dim,
            (255,),
            layers=2,
            units=hidden_dim,
            act="SiLU",
            norm=True,
            dist="symlog_disc",
            outscale=0.0,
            device="cpu",
            name="Reward",
        )

    def encode(self, prop, image):
        if not torch.is_grad_enabled() and prop.ndim == 2:
            return inference_chunks(lambda p, d: self.encoder({"prop": p, "image": d}), prop, image)
        return self.encoder({"prop": prop, "image": image})

    @staticmethod
    def feature(state):
        return torch.cat((state.stoch.flatten(1), state.deter), -1)

    @staticmethod
    def policy_feature(state):
        return state.deter

    def observe_step(self, prop, image, action_history, is_first, previous=None, sample=True):
        return self.rssm.obs_step(previous, action_history, self.encode(prop, image), is_first, sample)

    def loss(self, batch):
        core = self.rssm.core
        core._device = str(batch["prop"].device)
        self.reward_head._device = str(batch["prop"].device)
        embed = self.encode(batch["prop"], batch["image"])
        post, prior = core.observe(embed, batch["action"].clone(), batch["is_first"])
        kl, value, dynamics, representation = core.kl_loss(post, prior, 1.0, 0.5, 0.1)
        feature = core.get_feat(post)
        predictions = self.decoder(feature)
        # 原版分布先沿事件维求和，再对 batch/time 求平均，保持与 KL 的相对权重。
        prop_loss = -predictions["prop"].log_prob(batch["prop"])
        image_loss = -predictions["image"].log_prob(batch["image"])
        reward_loss = -self.reward_head(feature).log_prob(batch["reward"])
        total = (prop_loss + image_loss + self.reward_loss_scale * reward_loss + kl).mean()
        metrics = dict(
            world_loss=total,
            prop_loss=prop_loss.mean(),
            image_loss=image_loss.mean(),
            reward_loss=reward_loss.mean(),
            kl_loss=dynamics.mean(),
            representation_loss=representation.mean(),
            kl_unclipped=value.mean(),
        )
        return total, {key: item.detach() for key, item in metrics.items()}
