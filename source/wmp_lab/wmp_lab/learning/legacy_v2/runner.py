# SPDX-License-Identifier: Apache-2.0
"""连续 AMP-PPO rollout 与 10 Hz 世界模型/深度预测器联合训练。"""

from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn.functional as F

from wmp_lab.constants import ACTION_DIM, WORLD_MODEL_ACTION_HISTORY_STEPS

from .checkpoint import load_checkpoint, save_checkpoint
from .config import LearningCfg
from .networks import ActorCriticWMP, AMPDiscriminator, DepthPredictor
from .observations import HistoryBuffer, proprioception
from .ppo import AMPPPO
from .sequence_replay import SequenceReplay
from .world_model import RSSMState, WorldModel


def generalized_advantage(rewards, values, dones, final_value, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(final_value)
    next_value = final_value
    for step in range(len(rewards) - 1, -1, -1):
        continuing = (~dones[step]).float()
        delta = rewards[step] + gamma * next_value * continuing - values[step]
        carry = delta + gamma * gae_lambda * continuing * carry
        advantages[step] = carry
        next_value = values[step]
    return advantages, advantages + values


class WMPRunner:
    def __init__(self, cfg: LearningCfg, device="cpu"):
        self.cfg, self.device = cfg, torch.device(device)
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
            (256, 128) if cfg.world_hidden > 64 else (64, 32), base_channels=32 if cfg.world_hidden > 64 else 8
        ).to(self.device)
        self.world_model = WorldModel(
            cfg.world_feature_dim,
            cfg.world_stochastic_dim,
            cfg.world_hidden,
            cfg.world_categories,
            cfg.reward_loss_scale,
        ).to(self.device)
        self.ppo = AMPPPO(
            self.policy,
            self.discriminator,
            lr=cfg.lr,
            clip_param=cfg.clip_param,
            entropy_coef=cfg.entropy_coef,
            replay_size=cfg.replay_size,
        )
        self.depth_optimizer = torch.optim.AdamW(self.depth_predictor.parameters(), lr=cfg.depth_lr, weight_decay=1e-4)
        self.world_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=cfg.world_lr, eps=1e-8)
        self.iteration = self.total_steps = 0
        self._env = None
        self._resume_environment = None

    @property
    def modules(self):
        return dict(
            policy_value=self.policy,
            world_model=self.world_model,
            depth_predictor=self.depth_predictor,
            amp_discriminator=self.discriminator,
        )

    @property
    def optimizers(self):
        return dict(
            policy=self.ppo.policy_optimizer,
            world_model=self.world_optimizer,
            depth_predictor=self.depth_optimizer,
            amp_discriminator=self.ppo.discriminator_optimizer,
        )

    def save(self, path):
        environment = (
            self._env.curriculum_state_dict()
            if self._env is not None and hasattr(self._env, "curriculum_state_dict")
            else None
        )
        return save_checkpoint(
            path,
            config=asdict(self.cfg),
            iteration=self.iteration,
            modules=self.modules,
            optimizers=self.optimizers,
            normalizer=self.ppo.amp_normalizer,
            extra={"total_steps": self.total_steps, "environment": environment},
        )

    def load(self, path, *, load_optimizers=True):
        payload = load_checkpoint(
            path,
            modules=self.modules,
            optimizers=self.optimizers if load_optimizers else None,
            normalizer=self.ppo.amp_normalizer,
            map_location=self.device,
        )
        if payload["config"].get("robot", self.cfg.robot) != self.cfg.robot:
            raise ValueError("checkpoint robot differs from selected robot")
        self.iteration = int(payload["iteration"])
        self.total_steps = payload.get("extra", {}).get("total_steps", 0)
        self._resume_environment = payload.get("extra", {}).get("environment")
        self._env = None
        return payload

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = LearningCfg(**payload["config"])
        runner = cls(cfg, device)
        runner.load(path, load_optimizers=False)
        return runner

    @torch.no_grad()
    def _mixed_depth(self, env, prop):
        prediction = self.depth_predictor(env.get_forward_height_map().to(self.device), prop)
        real = env.get_depth().to(self.device)
        if real.ndim == 3:
            real = real.unsqueeze(-1)
        mixed = torch.where(env.camera_mask[:, None, None, None], real, prediction)
        return mixed, prediction, real

    def _initialize_rollout(self, env):
        if self._resume_environment is not None and hasattr(env, "load_curriculum_state_dict"):
            env.load_curriculum_state_dict(self._resume_environment)
            self._resume_environment = None
        self.observation = env.reset()[0]["policy"].to(self.device)
        n = env.num_envs
        self.history_buffer = HistoryBuffer(n, self.device)
        self.history = self.history_buffer.append(self.observation)
        self.action_history = torch.zeros(n, WORLD_MODEL_ACTION_HISTORY_STEPS, ACTION_DIM, device=self.device)
        self.rssm_state = None
        self.is_first = torch.ones(n, device=self.device)
        self.world_feature = torch.zeros(n, self.cfg.world_feature_dim, device=self.device)
        self.world_reward = torch.zeros(n, device=self.device)
        self.world_replay = SequenceReplay(n, self.cfg.world_replay_capacity)
        self.rollout_step = 0
        self._env = env
        if hasattr(env, "_robot"):
            limits = env._robot.data.joint_pos_limits[0, env._joint_ids]
            normalized = torch.tensor((0.05, 0.02, 0.05) * 4, device=self.device)
            self.ppo.min_std = normalized * (limits[:, 1] - limits[:, 0])

    def collect_and_update(self, env, expert, rollout_steps=None):
        if self._env is not env:
            self._initialize_rollout(env)
        if hasattr(env, "update_reward_curriculum"):
            env.update_reward_curriculum(self.iteration)
        records = {
            key: []
            for key in (
                "observation",
                "history",
                "world_feature",
                "action",
                "old_log_prob",
                "old_value",
                "old_mean",
                "old_std",
                "reward",
                "done",
                "amp_state",
                "amp_next_state",
            )
        }
        for _ in range(rollout_steps or self.cfg.rollout_steps):
            with torch.no_grad():
                if self.rollout_step % self.cfg.world_update_interval == 0:
                    prop = proprioception(self.observation)
                    depth, _, real = self._mixed_depth(env, prop)
                    self.rssm_state, _ = self.world_model.observe_step(
                        prop, depth, self.action_history.flatten(1), self.is_first, self.rssm_state
                    )
                    self.world_feature = self.world_model.policy_feature(self.rssm_state)
                    valid_camera = getattr(env, "depth_supervision_mask", env.camera_mask)
                    self.world_replay.insert(
                        dict(
                            prop=prop,
                            image=depth,
                            action=self.action_history.flatten(1),
                            reward=self.world_reward,
                            is_first=self.is_first,
                            forward=env.get_forward_height_map(),
                            real=real,
                            camera_valid=valid_camera,
                        )
                    )
                    self.is_first.zero_()
                    self.world_reward.zero_()
                action = self.policy.act(self.observation, self.history, self.world_feature)
                log_prob = self.policy.get_actions_log_prob(action)
                value = self.policy.evaluate(self.observation, self.world_feature).squeeze(-1)
                amp = env.get_amp_observations().clone().to(self.device)
                next_obs, task_reward, terminated, truncated, info = env.step(action)
                done = terminated | truncated
                amp_next = env.get_amp_observations().clone().to(self.device)
                if "terminal_amp_states" in info:
                    amp_next[done] = info["terminal_amp_states"][done]
                normalized_amp = self.ppo.amp_normalizer.normalize(amp)
                normalized_next = self.ppo.amp_normalizer.normalize(amp_next)
                reward, _ = self.discriminator.predict_amp_reward(normalized_amp, normalized_next, task_reward)
                # 自动 reset 前的 terminal observation 才能用于 timeout bootstrap。
                if "terminal_observation" in info and truncated.any():
                    terminal_value = self.policy.evaluate(info["terminal_observation"], self.world_feature).squeeze(-1)
                    reward += self.cfg.gamma * terminal_value * (truncated & ~terminated).float()
            for key, value_to_store in dict(
                observation=self.observation,
                history=self.history,
                world_feature=self.world_feature,
                action=action,
                old_log_prob=log_prob,
                old_value=value,
                old_mean=self.policy.action_mean,
                old_std=self.policy.action_std,
                reward=reward,
                done=done,
                amp_state=amp,
                amp_next_state=amp_next,
            ).items():
                records[key].append(value_to_store.detach().clone())
            self.world_reward += task_reward
            self.action_history = torch.cat((self.action_history[:, 1:], action.unsqueeze(1)), 1)
            reset_ids = done.nonzero(as_tuple=False).flatten()
            self.action_history[reset_ids] = 0
            self.history_buffer.reset(reset_ids)
            self.observation = next_obs["policy"].to(self.device)
            self.history = self.history_buffer.append(self.observation)
            self.is_first[reset_ids] = 1
            self.world_feature[reset_ids] = 0
            self.world_reward[reset_ids] = 0
            self.rollout_step += 1
            self.total_steps += env.num_envs
        with torch.no_grad():
            final_value = self.policy.evaluate(self.observation, self.world_feature).squeeze(-1)
        values = torch.stack(records["old_value"])
        advantage, returns = generalized_advantage(
            torch.stack(records["reward"]),
            values,
            torch.stack(records["done"]),
            final_value,
            self.cfg.gamma,
            self.cfg.gae_lambda,
        )
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        batch = {
            key: torch.stack(value).flatten(0, 1) for key, value in records.items() if key not in {"reward", "done"}
        }
        batch.update(
            critic_observation=batch["observation"], advantage=advantage.flatten(), **{"return": returns.flatten()}
        )
        self.ppo.amp_replay.insert(batch["amp_state"], batch["amp_next_state"])
        metrics = {}
        count = 0
        for _ in range(self.cfg.learning_epochs):
            for ids in torch.randperm(len(advantage.flatten()), device=self.device).chunk(self.cfg.mini_batches):
                if not len(ids):
                    continue
                result = self.ppo.update({key: value[ids] for key, value in batch.items()}, expert, insert_replay=False)
                for key, value in result.items():
                    metrics[key] = metrics.get(key, 0.0) + value
                count += 1
        metrics = {key: value / count for key, value in metrics.items()}
        if self.world_replay.size * env.num_envs >= self.cfg.world_train_start:
            supervision = getattr(env, "depth_supervision_mask", env.camera_mask).nonzero(as_tuple=False).flatten()
            if len(supervision):
                for _ in range(self.cfg.depth_train_steps):
                    depth_batch = self.world_replay.sample(self.cfg.depth_batch_size, 1, self.device, supervision)
                    prediction = self.depth_predictor(
                        depth_batch["forward"].flatten(0, 1), depth_batch["prop"].flatten(0, 1)
                    )
                    depth_loss = F.mse_loss(prediction, depth_batch["real"].flatten(0, 1))
                    self.depth_optimizer.zero_grad(set_to_none=True)
                    (depth_loss * 100).backward()
                    torch.nn.utils.clip_grad_norm_(self.depth_predictor.parameters(), 1.0)
                    self.depth_optimizer.step()
                    metrics["depth_loss"] = float(depth_loss.detach())
            for _ in range(self.cfg.world_train_steps):
                world_batch = self.world_replay.sample(
                    self.cfg.world_batch_size, self.cfg.world_sequence_length, self.device
                )
                world_loss, world_metrics = self.world_model.loss(world_batch)
                self.world_optimizer.zero_grad(set_to_none=True)
                world_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1000.0)
                self.world_optimizer.step()
                metrics.update({key: float(value) for key, value in world_metrics.items()})
        self.rssm_state = RSSMState(*(value.detach() for value in self.rssm_state))
        self.iteration += 1
        metrics.update(env.get_curriculum_metrics() if hasattr(env, "get_curriculum_metrics") else {})
        metrics = {key: float(value) for key, value in metrics.items()}
        if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise FloatingPointError("non-finite joint-runner metric")
        return metrics
