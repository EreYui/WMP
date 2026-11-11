# SPDX-License-Identifier: Apache-2.0
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
    ):
        self.actor_critic = actor_critic
        self.discriminator = discriminator
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.velocity_loss_coef = velocity_loss_coef
        self.max_grad_norm = max_grad_norm
        self.desired_kl = desired_kl
        self.min_std = 0.02
        self.policy_optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)
        self.discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
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

        distribution = self.actor_critic.update_distribution(
            batch["observation"], batch["history"], batch["world_feature"]
        )
        kl = torch.zeros((), device=batch["action"].device)
        if "old_mean" in batch and "old_std" in batch:
            with torch.no_grad():
                old_mu, old_std = batch["old_mean"], batch["old_std"]
                std, mu = distribution.scale, distribution.loc
                kl = (
                    (torch.log(std / old_std) + (old_std.square() + (old_mu - mu).square()) / (2 * std.square()) - 0.5)
                    .sum(-1)
                    .mean()
                )
                lr = self.policy_optimizer.param_groups[0]["lr"]
                if kl > 2 * self.desired_kl:
                    lr = max(1e-5, lr / 1.5)
                elif 0 < kl < self.desired_kl / 2:
                    lr = min(1e-2, lr * 1.5)
                for group in self.policy_optimizer.param_groups:
                    group["lr"] = lr
        log_prob = distribution.log_prob(batch["action"]).sum(-1)
        ratio = (log_prob - batch["old_log_prob"]).exp()
        advantage = batch["advantage"]
        surrogate = -torch.minimum(
            ratio * advantage,
            ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage,
        ).mean()
        value = self.actor_critic.evaluate(batch["critic_observation"], batch["world_feature"]).squeeze(-1)
        if "old_value" in batch:
            clipped = batch["old_value"] + (value - batch["old_value"]).clamp(-self.clip_param, self.clip_param)
            value_loss = torch.maximum((value - batch["return"]).square(), (clipped - batch["return"]).square()).mean()
        else:
            value_loss = functional.mse_loss(value, batch["return"])
        entropy = distribution.entropy().sum(-1).mean()
        target_velocity = batch["critic_observation"][..., 50:53]
        velocity_loss = functional.mse_loss(self.actor_critic.velocity_estimate(batch["history"]), target_velocity)
        policy_loss = (
            surrogate
            + self.value_loss_coef * value_loss
            + self.velocity_loss_coef * velocity_loss
            - self.entropy_coef * entropy
        )
        self.policy_optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()
        with torch.no_grad():
            floor = torch.as_tensor(self.min_std, device=self.actor_critic.log_std.device).log()
            self.actor_critic.log_std.copy_(torch.maximum(self.actor_critic.log_std, floor))

        if insert_replay:
            self.amp_replay.insert(batch["amp_state"], batch["amp_next_state"])
        replay_state, replay_next = self.amp_replay.sample(batch["amp_state"].shape[0])
        expert_state, expert_next, _ = expert.sample_transitions(batch["amp_state"].shape[0])
        self.amp_normalizer.update(torch.cat((expert_state, replay_state), dim=0))
        normalized_policy = self.amp_normalizer.normalize(replay_state)
        normalized_policy_next = self.amp_normalizer.normalize(replay_next)
        normalized_expert = self.amp_normalizer.normalize(expert_state)
        normalized_expert_next = self.amp_normalizer.normalize(expert_next)
        policy_score = self.discriminator.transition_score(normalized_policy, normalized_policy_next)
        expert_score = self.discriminator.transition_score(normalized_expert, normalized_expert_next)
        expert_loss = functional.mse_loss(expert_score, torch.ones_like(expert_score))
        policy_disc_loss = functional.mse_loss(policy_score, -torch.ones_like(policy_score))
        gradient_penalty = self.discriminator.gradient_penalty(normalized_expert, normalized_expert_next)
        discriminator_loss = 0.5 * (expert_loss + policy_disc_loss) + gradient_penalty
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        discriminator_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
        self.discriminator_optimizer.step()

        metrics = {
            "policy_loss": policy_loss,
            "kl": kl,
            "surrogate_loss": surrogate,
            "value_loss": value_loss,
            "velocity_loss": velocity_loss,
            "entropy": entropy,
            "discriminator_loss": discriminator_loss,
            "gradient_penalty": gradient_penalty,
            "expert_score": expert_score.mean(),
            "policy_score": policy_score.mean(),
        }
        if not all(torch.isfinite(value).all() for value in metrics.values()):
            raise FloatingPointError("non-finite AMP-PPO metric")
        return {key: float(value.detach()) for key, value in metrics.items()}
