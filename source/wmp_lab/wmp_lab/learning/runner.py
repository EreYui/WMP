# SPDX-License-Identifier: Apache-2.0
"""连续 AMP-PPO rollout 与 10 Hz 世界模型/深度预测器联合训练。"""

from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn.functional as F

from wmp_lab.constants import ACTION_DIM, JOINT_ORDER, WORLD_MODEL_ACTION_HISTORY_STEPS
from wmp_lab.robots import (
    ENVIRONMENT_REVISION,
    LEGACY_ENVIRONMENT_REVISION,
    checkpoint_control_revision,
    default_control_revision,
)

from .checkpoint import load_checkpoint, save_checkpoint
from .config import ALGORITHM_REVISION, REFERENCE_BATCH_SIZE, LearningCfg
from .contracts import environment_contract
from .networks import ActorCriticWMP, AMPDiscriminator, DepthPredictor
from .observations import HistoryBuffer, proprioception
from .ppo import AMPPPO
from .sequence_replay import EpisodeReplay, SequenceReplay
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


def segmented_advantage(rewards, values, dones, final_values, gamma, gae_lambda):
    """每段独立计算 GAE，禁止跨采集段延长 24 步信用分配窗口。"""
    chunks = len(final_values)
    if len(rewards) % chunks:
        raise ValueError("rollout must contain complete advantage segments")
    steps = len(rewards) // chunks
    advantage, returns = [], []
    for chunk, final_value in enumerate(final_values):
        block = slice(chunk * steps, (chunk + 1) * steps)
        a, r = generalized_advantage(rewards[block], values[block], dones[block], final_value, gamma, gae_lambda)
        advantage.append(a)
        returns.append(r)
    return torch.cat(advantage), torch.cat(returns)


def reference_batch_order(value, chunks, steps):
    """将串行采集的 [段,时间,环境] 合并为参考的 [时间,逻辑环境]。"""
    return value.reshape(chunks, steps, *value.shape[1:]).transpose(0, 1).flatten(0, 2)


class WMPRunner:
    def __init__(self, cfg: LearningCfg, device="cpu"):
        cfg.validate_schedule()
        self.cfg, self.device = cfg, torch.device(device)
        self.control_revision = default_control_revision(cfg.robot)
        self.environment_revision = ENVIRONMENT_REVISION
        self.environment_contract = None
        self.world_model = WorldModel(
            cfg.world_feature_dim,
            cfg.world_stochastic_dim,
            cfg.world_hidden,
            cfg.world_categories,
            cfg.reward_loss_scale,
        ).to(self.device)
        self.depth_predictor = DepthPredictor(
            (256, 128) if cfg.world_hidden > 64 else (64, 32), base_channels=32 if cfg.world_hidden > 64 else 8
        ).to(self.device)
        self.policy = ActorCriticWMP(
            world_feature_dim=cfg.world_feature_dim,
            history_hidden=cfg.history_hidden,
            actor_hidden=cfg.actor_hidden,
            critic_hidden=cfg.critic_hidden,
        ).to(self.device)
        self.discriminator = AMPDiscriminator(cfg.amp_hidden, cfg.amp_reward_coef, cfg.amp_task_reward_lerp).to(
            self.device
        )
        self.ppo = AMPPPO(
            self.policy,
            self.discriminator,
            lr=cfg.lr,
            clip_param=cfg.clip_param,
            entropy_coef=cfg.entropy_coef,
            replay_size=cfg.replay_size,
            micro_batch_size=cfg.ppo_micro_batch_size,
        )
        self.depth_optimizer = torch.optim.Adam(self.depth_predictor.parameters(), lr=cfg.depth_lr, weight_decay=1e-4)
        self.world_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=cfg.world_lr, eps=1e-8)
        self.iteration = self.total_steps = 0
        self._env = None
        self._resume_environment = None
        self.amp_dataset = self._amp_sampling_state = self._expert = None
        self._amp_contract_required = False

    @property
    def curriculum_iteration(self):
        return self.total_steps / REFERENCE_BATCH_SIZE if self.cfg.aligned_schedule else self.iteration

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
            policy_amp=self.ppo.optimizer,
            world_model=self.world_optimizer,
            depth_predictor=self.depth_optimizer,
        )

    def save(self, path):
        environment = (
            self._env.curriculum_state_dict()
            if self._env is not None and hasattr(self._env, "curriculum_state_dict")
            else self._resume_environment
        )
        return save_checkpoint(
            path,
            config=asdict(self.cfg),
            iteration=self.iteration,
            modules=self.modules,
            optimizers=self.optimizers,
            normalizer=self.ppo.amp_normalizer,
            extra={
                "total_steps": self.total_steps,
                "environment": environment,
                "control_revision": self.control_revision,
                "environment_revision": self.environment_revision,
                "environment_contract": self.environment_contract,
                "amp_dataset": self.amp_dataset,
                "amp_sampling_state": self._expert.sampling_state()
                if self._expert is not None
                else self._amp_sampling_state,
            },
        )

    def load(self, path, *, load_optimizers=True):
        if load_optimizers:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if saved["config"].get("algorithm_revision") != self.cfg.algorithm_revision:
                raise ValueError("checkpoint update schedule differs; use playback or start a fresh alignment run")
            if LearningCfg(**saved["config"]) != self.cfg:
                raise ValueError("checkpoint learning configuration differs; start a separate experiment")
            del saved
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
        self.control_revision = checkpoint_control_revision(payload)
        self.environment_revision = payload.get("extra", {}).get("environment_revision", LEGACY_ENVIRONMENT_REVISION)
        self.environment_contract = payload.get("extra", {}).get("environment_contract")
        self.amp_dataset = payload.get("extra", {}).get("amp_dataset")
        self._amp_contract_required = self.iteration > 0 and self.amp_dataset is None
        self._amp_sampling_state = payload.get("extra", {}).get("amp_sampling_state")
        self._expert = None
        self.total_steps = payload.get("extra", {}).get("total_steps", 0)
        self._resume_environment = payload.get("extra", {}).get("environment")
        self._env = None
        return payload

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("version") == 2:
            from .legacy_v2.runner import WMPRunner as LegacyRunner

            return LegacyRunner.from_checkpoint(path, device)
        cfg = LearningCfg(**payload["config"])
        runner = cls(cfg, device)
        runner.load(path, load_optimizers=False)
        return runner

    @torch.no_grad()
    def _mixed_depth(self, env, prop):
        from .inference import inference_chunks

        prediction = inference_chunks(self.depth_predictor, env.get_forward_height_map().to(self.device), prop)
        real = env.get_depth().to(self.device)
        if real.ndim == 3:
            real = real.unsqueeze(-1)
        mixed = torch.where(env.camera_mask[:, None, None, None], real, prediction)
        return mixed, prediction, real

    def _initialize_rollout(self, env):
        if self.cfg.aligned_schedule:
            if env.num_envs != self.cfg.num_envs:
                raise ValueError("physical environment count differs from aligned learning configuration")
            if hasattr(env, "configure_training_schedule"):
                env.configure_training_schedule(self.cfg.reference_num_envs)
        if hasattr(env, "cfg") and env.cfg.control_revision != self.control_revision:
            raise ValueError("checkpoint and environment control revisions differ; start a fresh training run")
        if hasattr(env, "cfg") and hasattr(env.cfg, "environment_revision"):
            if env.cfg.environment_revision != self.environment_revision:
                raise ValueError("checkpoint and environment revisions differ; start a fresh training run")
            actual_contract = environment_contract(env.cfg)
            if self.environment_contract and actual_contract["fingerprint"] != self.environment_contract["fingerprint"]:
                raise ValueError(
                    "environment/source/asset contract differs from checkpoint; start a separate experiment"
                )
            self.environment_contract = actual_contract
        if self._resume_environment is not None and hasattr(env, "load_curriculum_state_dict"):
            env.load_curriculum_state_dict(self._resume_environment)
            self._resume_environment = None
        self.observation = env.reset()[0]["policy"].to(self.device)
        if getattr(getattr(env, "cfg", None), "randomize_initial_episode_length", False):
            env.episode_length_buf[:] = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
        n = env.num_envs
        self.history_buffer = HistoryBuffer(n, self.device)
        self.history = self.history_buffer.append(self.observation)
        self.action_history = torch.zeros(n, WORLD_MODEL_ACTION_HISTORY_STEPS, ACTION_DIM, device=self.device)
        self.rssm_state = None
        self.is_first = torch.ones(n, device=self.device)
        self.world_feature = torch.zeros(n, self.cfg.world_feature_dim, device=self.device)
        self.world_reward = torch.zeros(n, device=self.device)
        if self.cfg.replay_mode == "episodes":
            capacity = int(env.max_episode_length / self.cfg.world_update_interval) + 3
            self.world_replay = EpisodeReplay(n, capacity, env.camera_mask, self.cfg.completed_episodes_per_env)
        elif self.cfg.replay_mode == "ring":
            self.world_replay = SequenceReplay(n, self.cfg.world_replay_capacity, env.camera_mask)
        else:
            raise ValueError(f"unknown replay mode: {self.cfg.replay_mode}")
        self.rollout_step = getattr(env, "_global_step", 0) if self.environment_revision == ENVIRONMENT_REVISION else 0
        self._env = env
        self._cached_depth = None
        if hasattr(env, "_robot"):
            all_limits = (
                env._robot.data.soft_joint_pos_limits
                if self.environment_revision == ENVIRONMENT_REVISION
                else env._robot.data.joint_pos_limits
            )
            limits = all_limits[0, env._joint_ids]
            normalized = torch.tensor((0.05, 0.02, 0.05) * 4, device=self.device)
            self.ppo.min_std = normalized * (limits[:, 1] - limits[:, 0])

    def bind_expert(self, expert):
        if self._amp_contract_required:
            raise ValueError("checkpoint is missing its AMP contract; use playback or start a fresh experiment")
        if not hasattr(expert, "contract"):
            return
        if self.amp_dataset and self.amp_dataset["fingerprint"] != expert.contract["fingerprint"]:
            raise ValueError("AMP dataset contract differs from checkpoint; start a separate experiment")
        if self._expert is not expert:
            if self._amp_sampling_state is not None:
                expert.load_sampling_state(self._amp_sampling_state)
                self._amp_sampling_state = None
            self._expert, self.amp_dataset = expert, expert.contract

    def collect_and_update(self, env, expert, rollout_steps=None):
        if self.cfg.algorithm_revision != ALGORITHM_REVISION:
            raise ValueError("historical update schedules support playback only; start a fresh alignment run")
        if self.cfg.aligned_schedule and rollout_steps is not None:
            raise ValueError("aligned training requires complete configured collection chunks")
        self.bind_expert(expert)
        if self._env is not env:
            self._initialize_rollout(env)
        curriculum_clock = self.curriculum_iteration
        if hasattr(env, "update_reward_curriculum"):
            env.update_reward_curriculum(curriculum_clock)
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
        task_return = torch.zeros(env.num_envs, device=self.device)
        segment_steps = rollout_steps or self.cfg.rollout_steps
        chunks = self.cfg.collection_chunks
        collection_steps = segment_steps * chunks
        final_values = []
        for step in range(collection_steps):
            with torch.no_grad():
                if self.rollout_step % self.cfg.world_update_interval == 0:
                    prop = proprioception(self.observation)
                    if self._cached_depth is None:
                        depth, _, real = self._mixed_depth(env, prop)
                    else:
                        depth, real = self._cached_depth
                    self.rssm_state, _ = self.world_model.observe_step(
                        prop, depth, self.action_history.flatten(1), self.is_first, self.rssm_state
                    )
                    self.world_feature = self.world_model.policy_feature(self.rssm_state)
                    valid_camera = getattr(env, "depth_supervision_mask", env.camera_mask)
                    self.world_replay.insert(
                        dict(
                            prop=prop,
                            action=self.action_history.flatten(1),
                            reward=self.world_reward,
                            is_first=self.is_first,
                            forward=env.get_forward_height_map(),
                            real=real,
                            camera_valid=valid_camera,
                            camera_real=env.camera_mask,
                        )
                    )
                    self.is_first.zero_()
                    self.world_reward.zero_()
                action = self.policy.act(self.observation, self.history, self.world_feature)
                log_prob = self.policy.get_actions_log_prob(action)
                value = self.policy.evaluate(self.observation, self.world_feature).squeeze(-1)
                amp = env.get_amp_observations().clone().to(self.device)
                next_obs, task_reward, terminated, truncated, info = env.step(action)
                task_return += task_reward
                done = terminated | truncated
                amp_next = env.get_amp_observations().clone().to(self.device)
                if "terminal_amp_states" in info:
                    amp_next[done] = info["terminal_amp_states"][done]
                normalized_amp = self.ppo.amp_normalizer.normalize(amp)
                normalized_next = self.ppo.amp_normalizer.normalize(amp_next)
                style_gate = info.get("locomotion_reward_gate")
                if style_gate is None:
                    reward, _ = self.discriminator.predict_amp_reward(normalized_amp, normalized_next, task_reward)
                else:
                    reward, _ = self.discriminator.predict_amp_reward(
                        normalized_amp, normalized_next, task_reward, style_gate=style_gate
                    )
                # 自动 reset 前的 terminal observation 才能用于 timeout bootstrap。
                if self.environment_revision == ENVIRONMENT_REVISION:
                    reward += self.cfg.gamma * value * truncated.float()
                elif "terminal_observation" in info and truncated.any():
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
            if isinstance(self.world_replay, EpisodeReplay):
                self.world_replay.finish_episodes(reset_ids)
            self.action_history[reset_ids] = 0
            self.history_buffer.reset(reset_ids)
            self.observation = next_obs["policy"].to(self.device)
            self.history = self.history_buffer.append(self.observation)
            self.is_first[reset_ids] = 1
            if self.environment_revision != ENVIRONMENT_REVISION:
                self.world_feature[reset_ids] = 0
                self.world_reward[reset_ids] = 0
            self.rollout_step += 1
            self.total_steps += env.num_envs
            with torch.no_grad():
                if self.rollout_step % self.cfg.world_update_interval == 0:
                    depth, _, real = self._mixed_depth(env, proprioception(self.observation))
                    # 上游在 env.step 后缓存，跨 PPO/感知更新边界仍消费旧参数生成的图像。
                    self._cached_depth = depth, real.clone()
                if (step + 1) % segment_steps == 0:
                    final_values.append(self.policy.evaluate(self.observation, self.world_feature).squeeze(-1))
        values = torch.stack(records["old_value"])
        advantage, returns = segmented_advantage(
            torch.stack(records["reward"]),
            values,
            torch.stack(records["done"]),
            final_values,
            self.cfg.gamma,
            self.cfg.gae_lambda,
        )
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        batch = {
            key: reference_batch_order(torch.stack(value), chunks, segment_steps)
            for key, value in records.items() if key not in {"reward", "done"}
        }
        batch.update(
            critic_observation=batch["observation"],
            advantage=reference_batch_order(advantage, chunks, segment_steps),
            **{"return": reference_batch_order(returns, chunks, segment_steps)}
        )
        self.ppo.amp_replay.insert(batch["amp_state"], batch["amp_next_state"])
        metrics = {}
        count = 0
        # 上游每轮 rollout 只洗牌一次，所有 PPO epoch 复用同一 minibatch 划分。
        mini_size = len(advantage.flatten()) // self.cfg.mini_batches
        if mini_size < 1:
            raise ValueError("rollout must contain at least one sample per minibatch")
        indices = torch.randperm(self.cfg.mini_batches * mini_size, device=self.device)
        for _ in range(self.cfg.learning_epochs):
            for ids in indices.split(mini_size):
                result = self.ppo.update({key: value[ids] for key, value in batch.items()}, expert, insert_replay=False)
                for key, value in result.items():
                    metrics[key] = metrics.get(key, 0.0) + value
                count += 1
        metrics = {key: value / count for key, value in metrics.items()}
        metrics["ppo_optimizer_steps"] = count
        metrics["depth_optimizer_steps"] = 0
        metrics["world_optimizer_steps"] = 0
        metrics["replay_completed_frames"] = self.world_replay.total_samples
        metrics["collection_steps"] = collection_steps
        metrics["batch_samples"] = collection_steps * env.num_envs
        metrics["ppo_minibatch_samples"] = mini_size
        metrics["gae_horizon"] = segment_steps
        metrics["curriculum_reference_iteration"] = curriculum_clock
        metrics["reference_iterations_collected"] = self.curriculum_iteration
        metrics["ppo_learning_rate"] = self.ppo.optimizer.param_groups[0]["lr"]
        if self.world_replay.total_samples > self.cfg.world_train_start:
            supervision = getattr(env, "depth_supervision_mask", env.camera_mask).nonzero(as_tuple=False).flatten()
            if isinstance(self.world_replay, EpisodeReplay):
                supervision = supervision[self.world_replay.has_completed(supervision)]
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
                    metrics["depth_optimizer_steps"] += 1
                    metrics["depth_loss"] = float(depth_loss.detach())
            for _ in range(self.cfg.world_train_steps):
                world_batch = self.world_replay.sample(
                    self.cfg.world_batch_size, self.cfg.world_sequence_length, self.device
                )
                if world_batch["prop"].shape[1] <= 1:
                    continue
                metrics["world_sequence_length"] = world_batch["prop"].shape[1]
                with torch.no_grad():
                    predicted = self.depth_predictor(
                        world_batch["forward"].flatten(0, 1), world_batch["prop"].flatten(0, 1)
                    ).reshape_as(world_batch["real"])
                    world_batch["image"] = torch.where(
                        world_batch["camera_real"][..., None, None, None], world_batch["real"], predicted
                    )
                world_loss, world_metrics = self.world_model.loss(world_batch)
                self.world_optimizer.zero_grad(set_to_none=True)
                world_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1000.0)
                self.world_optimizer.step()
                metrics["world_optimizer_steps"] += 1
                metrics.update({key: float(value) for key, value in world_metrics.items()})
        if self.rssm_state is not None:
            self.rssm_state = RSSMState(*(value.detach() for value in self.rssm_state))
        self.iteration += 1
        if hasattr(env, "_reward_curriculum_scale"):
            metrics["reward_curriculum_scale"] = env._reward_curriculum_scale
        if hasattr(env, "command_curriculum"):
            for index, name in enumerate(("omni", "rough_flat", "stairs_discrete", "forward", "slope_cliff")):
                metrics[f"command_limit/{name}_x"] = float(env.command_curriculum.upper[index, 0])
        metrics.update(env.get_curriculum_metrics() if hasattr(env, "get_curriculum_metrics") else {})
        metrics = {key: float(value) for key, value in metrics.items()}
        with torch.no_grad():
            for index, name in enumerate(JOINT_ORDER):
                metrics[f"action_std/{name}"] = float(self.policy.std[index])
                metrics[f"action_mean_abs/{name}"] = float(batch["old_mean"][:, index].abs().mean())
                metrics[f"action_saturation/{name}"] = float((batch["action"][:, index].abs() > 6).float().mean())
            metrics["rollout_return_mean"] = float(torch.stack(records["reward"]).sum(0).mean() / chunks)
            metrics["task_return_mean"] = float(task_return.mean() / chunks)
            metrics["done_fraction"] = float(torch.stack(records["done"]).float().mean())
            if hasattr(env, "_robot") and hasattr(env, "_terrain"):
                delta = env._robot.data.root_pos_w[:, :2] - env._terrain.env_origins[:, :2]
                metrics["course_progress_mean"] = float(delta[:, 0].mean())
                metrics["course_progress_max"] = float(delta[:, 0].max())
        if not all(torch.isfinite(torch.tensor(value)) for value in metrics.values()):
            raise FloatingPointError("non-finite joint-runner metric")
        return metrics
