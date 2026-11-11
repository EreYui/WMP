# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
# Portions Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES and ETH Zurich, Nikita Rudin.
# Portions Copyright Bytedance Ltd. and/or its affiliates.
"""AMP-PPO update preserving WMP reward and loss semantics."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from .amp import AMPMotionDataset, ReplayBuffer, RunningMeanStd
from .networks import ActorCriticWMP, AMPDiscriminator


class AMPPPO:
    def __init__(
        self,
        actor_critic: ActorCriticWMP,
        discriminator: AMPDiscriminator,
        *,
        lr: float = 1.0e-3,
        clip_param: float = 0.2,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        velocity_loss_coef: float = 1.0,
        max_grad_norm: float = 1.0,
        replay_size: int = 1_000_000,
        desired_kl: float = 0.01,
        micro_batch_size: int | None = None,
    ):
        self.actor_critic = actor_critic
        self.discriminator = discriminator
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.velocity_loss_coef = velocity_loss_coef
        self.max_grad_norm = max_grad_norm
        self.desired_kl = desired_kl
        self.micro_batch_size = micro_batch_size
        self.min_std = 0.02
        self.optimizer = torch.optim.Adam(
            [
                {"params": actor_critic.parameters(), "name": "actor_critic"},
                {"params": discriminator.trunk.parameters(), "weight_decay": 0.001, "name": "amp_trunk"},
                {"params": discriminator.head.parameters(), "weight_decay": 0.1, "name": "amp_head"},
            ],
            lr=lr,
        )
        self.policy_optimizer = self.optimizer
        device = next(actor_critic.parameters()).device
        self.amp_replay = ReplayBuffer(30, replay_size, device)
        self.amp_normalizer = RunningMeanStd(30, device=device)

    def update(
        self,
        batch: dict[str, torch.Tensor],
        expert: AMPMotionDataset,
        *,
        insert_replay: bool = True,
    ) -> dict[str, float]:
        required = {
            "observation",
            "critic_observation",
            "history",
            "world_feature",
            "action",
            "old_log_prob",
            "advantage",
            "return",
            "amp_state",
            "amp_next_state",
        }
        missing = required.difference(batch)
        if missing:
            raise KeyError(f"PPO batch missing {sorted(missing)}")

        size = batch["action"].shape[0]
        if insert_replay:
            self.amp_replay.insert(batch["amp_state"], batch["amp_next_state"])
        replay_state, replay_next = self.amp_replay.sample(size)
        expert_state, expert_next, _ = expert.sample_transitions(size)
        # 整个有效 minibatch 共用冻结的统计与样本；微批次不单独改变归一化器。
        normalized_policy = self.amp_normalizer.normalize(replay_state)
        normalized_policy_next = self.amp_normalizer.normalize(replay_next)
        normalized_expert = self.amp_normalizer.normalize(expert_state)
        normalized_expert_next = self.amp_normalizer.normalize(expert_next)
        micro = self.micro_batch_size or size
        if micro < 1:
            raise ValueError("PPO micro batch size must be positive")
        self.optimizer.zero_grad(set_to_none=True)
        metrics = {}
        for begin in range(0, size, micro):
            end = min(begin + micro, size)
            ids = slice(begin, end)
            part = {key: value[ids] for key, value in batch.items()}
            weight = (end - begin) / size
            distribution = self.actor_critic.update_distribution(
                part["observation"], part["history"], part["world_feature"]
            )
            kl = torch.zeros((), device=part["action"].device)
            if "old_mean" in part and "old_std" in part:
                with torch.no_grad():
                    old_mu, old_std = part["old_mean"], part["old_std"]
                    std, mu = distribution.scale, distribution.loc
                    kl = (
                        torch.log(std / old_std + 1.0e-5)
                        + (old_std.square() + (old_mu - mu).square()) / (2 * std.square()) - 0.5
                    ).sum(-1).mean()
            log_prob = distribution.log_prob(part["action"]).sum(-1)
            ratio = (log_prob - part["old_log_prob"]).exp()
            surrogate = -torch.minimum(
                ratio * part["advantage"],
                ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * part["advantage"],
            ).mean()
            value = self.actor_critic.evaluate(part["critic_observation"], part["world_feature"]).squeeze(-1)
            if "old_value" in part:
                clipped = part["old_value"] + (value - part["old_value"]).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum(
                    (value - part["return"]).square(), (clipped - part["return"]).square()
                ).mean()
            else:
                value_loss = functional.mse_loss(value, part["return"])
            entropy = distribution.entropy().sum(-1).mean()
            velocity_loss = functional.mse_loss(
                self.actor_critic.velocity_estimate(part["history"]), part["critic_observation"][..., 50:53]
            )
            policy_loss = (
                surrogate + self.value_loss_coef * value_loss + self.velocity_loss_coef * velocity_loss
                - self.entropy_coef * entropy
            )
            policy_score = self.discriminator.transition_score(normalized_policy[ids], normalized_policy_next[ids])
            expert_score = self.discriminator.transition_score(normalized_expert[ids], normalized_expert_next[ids])
            expert_loss = functional.mse_loss(expert_score, torch.ones_like(expert_score))
            policy_disc_loss = functional.mse_loss(policy_score, -torch.ones_like(policy_score))
            gradient_penalty = self.discriminator.gradient_penalty(expert_state[ids], expert_next[ids])
            discriminator_loss = 0.5 * (expert_loss + policy_disc_loss) + gradient_penalty
            current = dict(
                policy_loss=policy_loss, kl=kl, surrogate_loss=surrogate, value_loss=value_loss,
                velocity_loss=velocity_loss, entropy=entropy, discriminator_loss=discriminator_loss,
                gradient_penalty=gradient_penalty, expert_score=expert_score.mean(), policy_score=policy_score.mean(),
            )
            if not all(torch.isfinite(value).all() for value in current.values()):
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError("non-finite AMP-PPO micro batch metric")
            ((policy_loss + discriminator_loss) * weight).backward()
            for key, value in current.items():
                metrics[key] = metrics.get(key, 0) + value.detach() * weight

        # KL/LR、裁剪和 Adam 仅在完整 minibatch 的所有梯度累积后执行一次。
        if "old_mean" in batch and "old_std" in batch:
            kl = metrics["kl"]
            lr = self.optimizer.param_groups[0]["lr"]
            if kl > 2 * self.desired_kl:
                lr = max(1e-5, lr / 1.5)
            elif 0 < kl < self.desired_kl / 2:
                lr = min(1e-2, lr * 1.5)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()
        with torch.no_grad():
            self.actor_critic.std.copy_(
                torch.maximum(self.actor_critic.std, torch.as_tensor(self.min_std, device=self.actor_critic.std.device))
            )
        self.amp_normalizer.update(normalized_policy.detach())
        self.amp_normalizer.update(normalized_expert.detach())
        return {key: float(value) for key, value in metrics.items()}
