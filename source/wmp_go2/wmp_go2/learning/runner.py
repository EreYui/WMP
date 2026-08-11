# SPDX-License-Identifier: Apache-2.0
"""Joint AMP-PPO, depth-predictor, and Dreamer world-model runner."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as functional

from wmp_go2.constants import (
    ACTION_DIM,
    DEPTH_SHAPE,
    WORLD_MODEL_ACTION_HISTORY_STEPS,
)

from .amp import AMPMotionDataset
from .checkpoint import load_checkpoint, save_checkpoint
from .config import LearningCfg
from .networks import ActorCriticWMP, AMPDiscriminator, DepthPredictor
from .observations import HistoryBuffer, proprioception
from .ppo import AMPPPO
from .world_model import RSSMState, WorldModel


class WMPRunner:
    def __init__(self, cfg: LearningCfg, device: torch.device | str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.policy = ActorCriticWMP(
            world_feature_dim=cfg.world_feature_dim,
            history_hidden=cfg.history_hidden,
            actor_hidden=cfg.actor_hidden,
            critic_hidden=cfg.critic_hidden,
        ).to(self.device)
        self.discriminator = AMPDiscriminator(cfg.amp_hidden, cfg.amp_reward_coef, cfg.amp_task_reward_lerp).to(
            self.device
        )
        self.depth_predictor = DepthPredictor(
            hidden_dims=(cfg.world_hidden, max(cfg.world_hidden // 2, 16)),
            base_channels=8 if cfg.world_hidden <= 64 else 32,
        ).to(self.device)
        self.world_model = WorldModel(
            feature_dim=cfg.world_feature_dim,
            stochastic_dim=cfg.world_stochastic_dim,
            hidden_dim=cfg.world_hidden,
        ).to(self.device)
        self.ppo = AMPPPO(
            self.policy,
            self.discriminator,
            lr=cfg.lr,
            clip_param=cfg.clip_param,
            entropy_coef=cfg.entropy_coef,
            replay_size=cfg.replay_size,
        )
        self.depth_optimizer = torch.optim.AdamW(
            self.depth_predictor.parameters(), lr=cfg.depth_lr, weight_decay=1.0e-4
        )
        self.world_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=cfg.world_lr, eps=1.0e-8)
        self.iteration = 0

    @property
    def modules(self) -> dict[str, torch.nn.Module]:
        return {
            "policy_value": self.policy,
            "world_model": self.world_model,
            "depth_predictor": self.depth_predictor,
            "amp_discriminator": self.discriminator,
        }

    @property
    def optimizers(self) -> dict[str, torch.optim.Optimizer]:
        return {
            "policy": self.ppo.policy_optimizer,
            "world_model": self.world_optimizer,
            "depth_predictor": self.depth_optimizer,
            "amp_discriminator": self.ppo.discriminator_optimizer,
        }

    def save(self, path: str | Path) -> Path:
        return save_checkpoint(
            path,
            config=asdict(self.cfg),
            iteration=self.iteration,
            modules=self.modules,
            optimizers=self.optimizers,
            normalizer=self.ppo.amp_normalizer,
        )

    def load(self, path: str | Path, *, load_optimizers: bool = True) -> dict:
        payload = load_checkpoint(
            path,
            modules=self.modules,
            optimizers=self.optimizers if load_optimizers else None,
            normalizer=self.ppo.amp_normalizer,
            map_location=self.device,
        )
        self.iteration = int(payload["iteration"])
        # Replay is deliberately empty and must warm up again after resume.
        return payload

    def _mixed_depth(self, env, prop: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        forward_map = env.get_forward_height_map().to(self.device)
        prediction = self.depth_predictor(forward_map, prop)
        real_depth = env.get_depth().to(self.device)
        if real_depth.shape[1:] == DEPTH_SHAPE[:2]:
            real_depth = real_depth.unsqueeze(-1)
        mask = env.camera_mask.to(self.device)
        mixed = prediction.clone()
        mixed[mask] = real_depth[mask]
        return mixed, prediction, real_depth

    def collect_and_update(self, env, expert: AMPMotionDataset, rollout_steps: int | None = None) -> dict[str, float]:
        """Run a short real-environment rollout and one update of every subsystem."""
        steps = rollout_steps or self.cfg.rollout_steps
        reset_result = env.reset()
        observation_dict = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        observation = observation_dict["policy"].to(self.device)
        num_envs = observation.shape[0]
        history_buffer = HistoryBuffer(num_envs, self.device)
        history = history_buffer.append(observation)
        action_history = torch.zeros(num_envs, WORLD_MODEL_ACTION_HISTORY_STEPS, ACTION_DIM, device=self.device)
        rssm_state: RSSMState | None = None
        is_first = torch.ones(num_envs, device=self.device)
        records: dict[str, list[torch.Tensor]] = {
            key: []
            for key in (
                "observation",
                "history",
                "world_feature",
                "action",
                "old_log_prob",
                "value",
                "reward",
                "done",
                "amp_state",
                "amp_next_state",
                "prop",
                "image",
                "world_action",
                "is_first",
                "depth_prediction",
                "real_depth",
            )
        }
        for _ in range(steps):
            prop = proprioception(observation)
            mixed_depth, depth_prediction, real_depth = self._mixed_depth(env, prop)
            with torch.no_grad():
                rssm_state, _ = self.world_model.observe_step(
                    prop, mixed_depth, action_history.flatten(1), is_first, rssm_state
                )
                world_feature = self.world_model.policy_feature(rssm_state)
                action = self.policy.act(observation, history, world_feature)
                log_prob = self.policy.get_actions_log_prob(action)
                value = self.policy.evaluate(observation, world_feature).squeeze(-1)
                amp_state = env.get_amp_observations().to(self.device)
            result = env.step(action)
            next_observation_dict, task_reward, terminated, truncated, _ = result
            next_observation = next_observation_dict["policy"].to(self.device)
            done = terminated | truncated
            amp_next = env.get_amp_observations().to(self.device)
            with torch.no_grad():
                reward, _ = self.discriminator.predict_amp_reward(amp_state, amp_next, task_reward.to(self.device))
            action_history = torch.cat((action_history[:, 1:], action.unsqueeze(1)), dim=1)
            for key, value_to_store in {
                "observation": observation,
                "history": history,
                "world_feature": world_feature,
                "action": action,
                "old_log_prob": log_prob,
                "value": value,
                "reward": reward,
                "done": done,
                "amp_state": amp_state,
                "amp_next_state": amp_next,
                "prop": prop,
                "image": mixed_depth,
                "world_action": action_history.flatten(1),
                "is_first": is_first,
                "depth_prediction": depth_prediction,
                "real_depth": real_depth,
            }.items():
                records[key].append(value_to_store if key == "depth_prediction" else value_to_store.detach())
            history_buffer.reset(done.nonzero(as_tuple=False).flatten())
            history = history_buffer.append(next_observation)
            is_first = done.float()
            observation = next_observation

        with torch.no_grad():
            final_value = self.policy.evaluate(observation, self.world_model.policy_feature(rssm_state)).squeeze(-1)
        returns: list[torch.Tensor] = []
        running_return = final_value
        for reward, done in reversed(list(zip(records["reward"], records["done"], strict=True))):
            running_return = reward + self.cfg.gamma * running_return * (~done).float()
            returns.append(running_return)
        returns.reverse()
        returns_tensor = torch.stack(returns)
        values_tensor = torch.stack(records["value"])
        advantage = returns_tensor - values_tensor
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1.0e-8)

        def flatten(key: str) -> torch.Tensor:
            return torch.stack(records[key]).flatten(0, 1)

        ppo_batch = {
            "observation": flatten("observation"),
            "critic_observation": flatten("observation"),
            "history": flatten("history"),
            "world_feature": flatten("world_feature"),
            "action": flatten("action"),
            "old_log_prob": flatten("old_log_prob"),
            "advantage": advantage.flatten(),
            "return": returns_tensor.flatten(),
            "amp_state": flatten("amp_state"),
            "amp_next_state": flatten("amp_next_state"),
        }
        metrics = self.ppo.update(ppo_batch, expert)

        camera_mask = env.camera_mask.to(self.device)
        depth_prediction = torch.stack(records["depth_prediction"], dim=1)
        real_depth = torch.stack(records["real_depth"], dim=1)
        depth_loss = functional.mse_loss(depth_prediction[camera_mask], real_depth[camera_mask])
        self.depth_optimizer.zero_grad(set_to_none=True)
        depth_loss.backward()
        self.depth_optimizer.step()

        world_batch = {
            "prop": torch.stack(records["prop"], dim=1),
            "image": torch.stack(records["image"], dim=1),
            "action": torch.stack(records["world_action"], dim=1),
            "reward": torch.stack(records["reward"], dim=1),
            "is_first": torch.stack(records["is_first"], dim=1),
        }
        world_loss, world_metrics = self.world_model.loss(world_batch)
        self.world_optimizer.zero_grad(set_to_none=True)
        world_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1000.0)
        self.world_optimizer.step()
        self.iteration += 1
        metrics["depth_loss"] = float(depth_loss.detach())
        metrics.update({key: float(value) for key, value in world_metrics.items()})
        if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise FloatingPointError("non-finite joint-runner metric")
        return metrics
