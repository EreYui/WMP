# SPDX-License-Identifier: Apache-2.0
"""有状态 TorchScript 部署策略与跨双频率的数值一致性验证。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from wmp_lab.robots import (
    ENVIRONMENT_REVISION,
    GO2_JOINT_CONTROL_REVISION,
    LEGACY_CONTROL_REVISION,
    LEGACY_ENVIRONMENT_REVISION,
    robot_metadata,
)


class ScriptDepthEncoder(nn.Module):
    def __init__(self, source):
        super().__init__()
        layers = []
        for layer in source.layers:
            if isinstance(layer, nn.Conv2d):
                # 固定64x64输入的四级SamePad均等价于padding=1，避免脚本依赖第三方动态形状代码。
                target = nn.Conv2d(layer.in_channels, layer.out_channels, 4, 2, 1, bias=layer.bias is not None)
                target.load_state_dict(copy.deepcopy(layer.state_dict()))
                layers.append(target)
            else:
                layers.append(copy.deepcopy(layer).cpu())
        self.layers = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers((image - 0.5).permute(0, 3, 1, 2)).flatten(1)


class ScriptRSSM(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.action_encoder = copy.deepcopy(source._img_in_layers).cpu()
        self.cell = copy.deepcopy(source._cell.layers).cpu()
        self.prior = copy.deepcopy(source._img_out_layers).cpu()
        self.prior_logits = copy.deepcopy(source._imgs_stat_layer).cpu()
        self.posterior = copy.deepcopy(source._obs_out_layers).cpu()
        self.posterior_logits = copy.deepcopy(source._obs_stat_layer).cpu()
        self.register_buffer("initial_deter", source.W.detach().cpu().clone())
        self.deter_dim, self.stoch_dim, self.categories = source._deter, source._stoch, source._discrete

    def _mode(self, logits: torch.Tensor) -> torch.Tensor:
        return F.one_hot(logits.argmax(-1), self.categories).to(logits.dtype)

    def forward(self, deter, stoch, action, embed, is_first, sample: bool = False):
        initial = self.initial_deter.tanh().expand_as(deter)
        initial_logits = self.prior_logits(self.prior(initial)).reshape(-1, self.stoch_dim, self.categories)
        first = is_first.to(torch.bool).unsqueeze(-1)
        deter = torch.where(first, initial, deter)
        stoch = torch.where(first.unsqueeze(-1), self._mode(initial_logits), stoch)
        action = torch.where(first, torch.zeros_like(action), action)
        encoded = self.action_encoder(torch.cat((stoch.flatten(1), action), -1))
        reset, candidate, update = self.cell(torch.cat((encoded, deter), -1)).chunk(3, -1)
        candidate = torch.tanh(torch.sigmoid(reset) * candidate)
        update = torch.sigmoid(update - 1.0)
        deter = update * candidate + (1.0 - update) * deter
        prior = self.prior_logits(self.prior(deter)).reshape(-1, self.stoch_dim, self.categories)
        logits = self.posterior_logits(self.posterior(torch.cat((deter, embed), -1)))
        logits = logits.reshape(-1, self.stoch_dim, self.categories)
        if sample:
            probabilities = torch.softmax(logits, -1) * 0.99 + 0.01 / self.categories
            indices = torch.multinomial(probabilities.reshape(-1, self.categories), 1).reshape(-1, self.stoch_dim)
            stochastic = F.one_hot(indices, self.categories).to(logits.dtype)
        else:
            stochastic = self._mode(logits)
        return deter, stochastic, prior, logits


class UpstreamDeploymentPolicy(nn.Module):
    def __init__(self, runner, batch_size=1, sample_rssm=None):
        super().__init__()
        self.reference_mode = (
            getattr(runner, "environment_revision", LEGACY_ENVIRONMENT_REVISION) == ENVIRONMENT_REVISION
        )
        self.sample_rssm = self.reference_mode if sample_rssm is None else bool(sample_rssm)
        world = runner.world_model
        self.prop_encoder = copy.deepcopy(world.encoder._mlp.layers).cpu()
        self.depth_encoder = ScriptDepthEncoder(world.encoder._cnn)
        self.rssm = ScriptRSSM(world.rssm.core)
        self.history_encoder = copy.deepcopy(runner.policy.history_encoder).cpu()
        self.world_encoder = copy.deepcopy(runner.policy.actor_world_encoder).cpu()
        self.actor = copy.deepcopy(runner.policy.actor).cpu()
        self.interval = runner.cfg.world_update_interval
        self.update_phase = self.interval - 1 if self.reference_mode else 0
        initial = world.rssm.initial(batch_size, runner.device)
        self.register_buffer("deter", initial.deter.detach().cpu().clone())
        self.register_buffer("stoch", initial.stoch.detach().cpu().clone())
        self.register_buffer("world_feature", torch.zeros_like(self.deter))
        self.register_buffer("history", torch.zeros(batch_size, 5, 42))
        self.register_buffer("actions", torch.zeros(batch_size, 5, 12))
        self.register_buffer("first", torch.ones(batch_size))
        self.register_buffer("counter", torch.zeros((), dtype=torch.long))
        self.eval()

    @torch.jit.export
    def reset(self):
        self.history.zero_()
        self.actions.zero_()
        self.world_feature.zero_()
        self.first.fill_(1)
        self.counter.zero_()

    @torch.jit.export
    def reset_envs(self, done: torch.Tensor):
        self.history[done] = 0
        self.actions[done] = 0
        if not self.reference_mode:
            self.world_feature[done] = 0
        self.first[done] = 1

    def forward(self, observation: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        if depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(-1)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(-1)
        elif depth.dim() == 4 and depth.shape[1] == 1:
            depth = depth.permute(0, 2, 3, 1)
        torch._assert(observation.dim() == 2 and observation.shape[1] == 45, "expected [B,45] observation")
        torch._assert(observation.shape[0] == self.history.shape[0], "batch differs from exported policy")
        torch._assert(
            depth.dim() == 4 and depth.shape[1] == 64 and depth.shape[2] == 64 and depth.shape[3] == 1,
            "expected [B,64,64,1] normalized depth",
        )
        frame = torch.cat((observation[:, :6], observation[:, 9:]), -1)
        self.history.copy_(torch.cat((self.history[:, 1:], frame.unsqueeze(1)), 1))
        if int(self.counter.item()) % self.interval == self.update_phase:
            prop = observation[:, :33]
            prop = torch.sign(prop) * torch.log1p(prop.abs())
            embed = torch.cat((self.depth_encoder(depth), self.prop_encoder(prop)), -1)
            deter, stoch, _, _ = self.rssm(
                self.deter, self.stoch, self.actions.flatten(1), embed, self.first, self.sample_rssm
            )
            self.deter.copy_(deter.detach())
            self.stoch.copy_(stoch.detach())
            self.world_feature.copy_(deter.detach())
            self.first.zero_()
        actor_input = torch.cat(
            (
                self.history_encoder(self.history.flatten(1)),
                observation[:, 6:9],
                self.world_encoder(self.world_feature),
            ),
            -1,
        )
        action = self.actor(actor_input).detach()
        self.actions.copy_(torch.cat((self.actions[:, 1:], action.unsqueeze(1)), 1))
        self.counter.add_(1)
        return action


def DeploymentPolicy(runner, batch_size=1, sample_rssm=None):
    if not hasattr(runner.cfg, "algorithm_revision"):
        from .legacy_v2.deployment import DeploymentPolicy as LegacyPolicy

        return LegacyPolicy(runner, batch_size)
    return UpstreamDeploymentPolicy(runner, batch_size, sample_rssm)


def deployment_environment_metadata(runner):
    """导出保存时的来源与求解器契约；历史模型缺失的内容保持未知。"""
    contract = getattr(runner, "environment_contract", None) or {}
    config = contract.get("config", {})
    sim = config.get("sim", {})
    return dict(
        environment_contract_fingerprint=contract.get("fingerprint"),
        source_sha256=copy.deepcopy(contract.get("source", {})),
        asset_sha256=copy.deepcopy(contract.get("assets", {})),
        simulation=dict(
            physics_dt=sim.get("dt"), decimation=config.get("decimation"),
            physx=copy.deepcopy(sim.get("physx", {})),
        ),
    )


def deployment_actuation_metadata(profile, control_revision):
    """仅为 v2 标明实验性传动语义，保持历史版本原有字段含义。"""
    if control_revision != GO2_JOINT_CONTROL_REVISION:
        return {}
    return dict(actuation_semantics=dict(
        position_target_space="joint_radians",
        pd_torque_space="joint",
        torque_speed_curve_space="motor",
        joint_gear_ratios=list(profile["joint_gear_ratios"]),
        motor_requested_torque="joint_pd_torque / gear_ratio",
        motor_velocity="joint_velocity * gear_ratio",
        joint_applied_torque="motor_applied_torque * gear_ratio",
        motor_strength_order="after joint output mapping",
        adaptation_status="experimental asset-derived calf transmission",
    ))


def export_policy(runner, destination: str | Path, batch_size=1):
    if not hasattr(runner.cfg, "algorithm_revision"):
        from .legacy_v2.deployment import export_policy as legacy_export

        result = legacy_export(runner, destination, batch_size)
        metadata_path = Path(destination).with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        metadata["robot"] = robot_metadata(runner.cfg.robot, LEGACY_CONTROL_REVISION, LEGACY_ENVIRONMENT_REVISION)
        metadata["control_revision"] = LEGACY_CONTROL_REVISION
        metadata.update(deployment_environment_metadata(runner))
        metadata_path.write_text(json.dumps(metadata, indent=2))
        return result
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    policy = torch.jit.script(DeploymentPolicy(runner, batch_size))
    intended_sampling = policy.sample_rssm
    policy.sample_rssm = False
    # 对比独立参考推理路径，覆盖更新边界、单环境 reset 和整体 reset。
    state = None
    history = torch.zeros(batch_size, 5, 42, device=runner.device)
    actions = torch.zeros(batch_size, 5, 12, device=runner.device)
    first = torch.ones(batch_size, device=runner.device)
    feature = torch.zeros(batch_size, runner.cfg.world_feature_dim, device=runner.device)
    generator = torch.Generator().manual_seed(17)
    max_error = 0.0
    with torch.inference_mode():
        for step in range(3 * runner.cfg.world_update_interval + 2):
            obs = torch.randn(batch_size, 45, generator=generator)
            depth = torch.rand(batch_size, 64, 64, 1, generator=generator) - 0.5
            obs_device, depth_device = obs.to(runner.device), depth.to(runner.device)
            history = torch.cat((history[:, 1:], torch.cat((obs_device[:, :6], obs_device[:, 9:]), -1)[:, None]), 1)
            if step % runner.cfg.world_update_interval == policy.update_phase:
                state, _ = runner.world_model.observe_step(
                    obs_device[:, :33], depth_device, actions.flatten(1), first, state, sample=False
                )
                feature = state.deter
                first.zero_()
            reference = runner.policy.actor(
                torch.cat(
                    (
                        runner.policy.history_encoder(history.flatten(1)),
                        obs_device[:, 6:9],
                        runner.policy.actor_world_encoder(feature),
                    ),
                    -1,
                )
            )
            actual = policy(obs, depth).to(runner.device)
            max_error = max(max_error, float((reference - actual).abs().max()))
            actions = torch.cat((actions[:, 1:], reference[:, None]), 1)
            if step == runner.cfg.world_update_interval + 1:
                done = torch.zeros(batch_size, dtype=torch.bool)
                done[0] = True
                policy.reset_envs(done)
                history[0] = 0
                actions[0] = 0
                if not policy.reference_mode:
                    feature[0] = 0
                first[0] = 1
        if max_error > 1e-4:
            raise RuntimeError(f"deployment parity failed: {max_error}")
        policy.reset()
        policy.sample_rssm = intended_sampling
        policy.save(str(destination))
        reloaded = torch.jit.load(str(destination))
        obs, depth = torch.zeros(batch_size, 45), torch.zeros(batch_size, 64, 64, 1)
        torch.testing.assert_close(policy(obs, depth), reloaded(obs, depth), rtol=0, atol=0)
    metadata = {
        "format_version": 4,
        "robot": robot_metadata(runner.cfg.robot, runner.control_revision, runner.environment_revision),
        "environment_revision": runner.environment_revision,
        "control_revision": runner.control_revision,
        "observation_dim": 45,
        "depth_shape": [64, 64, 1],
        "depth_range_m": [0, 2],
        "normalized_depth_range": [-0.5, 0.5],
        "policy_hz": 50,
        "world_hz": 10,
        "batch_size": batch_size,
        "rssm_sampling": "sample" if intended_sampling else "mode",
        "world_update_phase": policy.update_phase,
        "camera_capture_after_call_phase": 3 if policy.reference_mode else 0,
        "initial_normalized_depth": 0.0 if policy.reference_mode else 0.5,
        "camera_delay_frames": 1,
        "max_abs_error": max_error,
        "checkpoint_iteration": runner.iteration,
    }
    metadata.update(deployment_environment_metadata(runner))
    metadata.update(deployment_actuation_metadata(metadata["robot"], runner.control_revision))
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return destination, max_error
