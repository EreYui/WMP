# SPDX-License-Identifier: Apache-2.0
"""有状态 TorchScript 部署策略与跨双频率的数值一致性验证。"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from wmp_lab.robots import robot_profile


class DeploymentPolicy(nn.Module):
    def __init__(self, runner, batch_size=1):
        super().__init__()
        world = runner.world_model
        self.prop_encoder = copy.deepcopy(world.prop_encoder).cpu()
        self.depth_encoder = copy.deepcopy(world.depth_encoder).cpu()
        self.rssm = copy.deepcopy(world.rssm).cpu()
        self.history_encoder = copy.deepcopy(runner.policy.history_encoder).cpu()
        self.world_encoder = copy.deepcopy(runner.policy.actor_world_encoder).cpu()
        self.actor = copy.deepcopy(runner.policy.actor).cpu()
        self.interval = runner.cfg.world_update_interval
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
        if int(self.counter.item()) % self.interval == 0:
            prop = observation[:, :33]
            prop = torch.sign(prop) * torch.log1p(prop.abs())
            embed = torch.cat((self.prop_encoder(prop), self.depth_encoder(depth)), -1)
            deter, stoch, _, _ = self.rssm(self.deter, self.stoch, self.actions.flatten(1), embed, self.first, False)
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


def export_policy(runner, destination: str | Path, batch_size=1):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    policy = torch.jit.script(DeploymentPolicy(runner, batch_size))
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
            if step % runner.cfg.world_update_interval == 0:
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
                feature[0] = 0
                first[0] = 1
        if max_error > 1e-4:
            raise RuntimeError(f"deployment parity failed: {max_error}")
        policy.reset()
        policy.save(str(destination))
        reloaded = torch.jit.load(str(destination))
        obs, depth = torch.zeros(batch_size, 45), torch.zeros(batch_size, 64, 64, 1)
        torch.testing.assert_close(policy(obs, depth), reloaded(obs, depth), rtol=0, atol=0)
    metadata = {
        "format_version": 2,
        "robot": asdict(robot_profile(runner.cfg.robot)),
        "observation_dim": 45,
        "depth_shape": [64, 64, 1],
        "depth_range_m": [0, 2],
        "normalized_depth_range": [-0.5, 0.5],
        "policy_hz": 50,
        "world_hz": 10,
        "batch_size": batch_size,
        "rssm_sampling": "mode",
        "max_abs_error": max_error,
        "checkpoint_iteration": runner.iteration,
    }
    destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return destination, max_error
