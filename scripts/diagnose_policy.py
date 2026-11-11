#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""暂停训练后在四类地形回放，并记录动作饱和、课程进度与奖励分解。

用法：WMP_DIAG_MODE=mean ISAACLAB_SMOKE_SCRIPT=$PWD/scripts/diagnose_policy.py
      ISAACLAB_SMOKE_STEPS=1000 /home/re/IsaacLab/tools/run_codex_isaaclab_smoke.sh。
模式 mean/sample/capped；WMP_DIAG_VIDEO=1 输出 gap、climb 回放。
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from types import MethodType

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
video_enabled = os.environ.get("WMP_DIAG_VIDEO") == "1"
args.enable_cameras = video_enabled
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import wmp_lab  # noqa: E402,F401
from wmp_lab.learning.observations import HistoryBuffer, proprioception  # noqa: E402
from wmp_lab.learning.runner import WMPRunner  # noqa: E402
from wmp_lab.learning.progress_rewards import checkpoint_reward_revision  # noqa: E402
from wmp_lab.robots import LEGACY_CONTROL_REVISION, LEGACY_ENVIRONMENT_REVISION  # noqa: E402
from wmp_lab.tasks.direct.wmp_env_cfg import make_env_cfg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
checkpoint = Path(os.environ.get("WMP_DIAG_CHECKPOINT", str(ROOT / "logs/go2/20261014_113404/model_9500.pt")))
mode = os.environ.get("WMP_DIAG_MODE", "mean")
depth_mode = os.environ.get("WMP_DIAG_DEPTH", "real")
speed = float(os.environ.get("WMP_DIAG_SPEED", "0.45"))
output = Path(os.environ.get("WMP_DIAG_OUTPUT", str(checkpoint.parent / "diagnostics" / f"{mode}_{depth_mode}")))
num_envs = int(os.environ.get("WMP_DIAG_ENVS", "16"))
records = []


def policy_std(runner):
    return runner.policy.std if hasattr(runner.policy, "std") else runner.policy.log_std.exp()


def instrument(env):
    original = env._get_rewards

    def wrapped(self):
        reward = original()
        robot = self._robot.data
        reconstructed = sum(self._reward_terms.values()) * self.step_dt
        torch.testing.assert_close(reconstructed.clamp_min(0), reward, atol=1e-6, rtol=1e-5)
        row = dict(
            pos=(robot.root_pos_w - self._terrain.env_origins).clone(),
            start_progress=(robot.root_pos_w[:, 0] - self._episode_start_pos[:, 0]).clone(),
            velocity=robot.root_lin_vel_b.clone(),
            gravity=robot.projected_gravity_b.clone(),
            actions=self._actions.clone(),
            q=robot.joint_pos[:, self._joint_ids].clone(),
            dq=robot.joint_vel[:, self._joint_ids].clone(),
            applied_torque=robot.applied_torque[:, self._joint_ids].clone(),
            command=self._commands.clone(),
            task_reward=reward.clone(),
            episode_steps=self.episode_length_buf.clone(),
            unclipped_reward=reconstructed.clone(),
        )
        row.update({f"reward_{key}": value.clone() for key, value in self._reward_terms.items()})
        self._diagnostic_record = row
        return reward

    env._get_rewards = MethodType(wrapped, env)


def main():
    if mode not in {"mean", "sample", "capped"} or depth_mode not in {"real", "mixed"}:
        raise ValueError("unsupported diagnostic mode/depth")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    runner = WMPRunner.from_checkpoint(checkpoint, args.device)
    original_std = policy_std(runner).detach().cpu().tolist()
    if mode == "capped":
        with torch.no_grad():
            runner.policy.std.clamp_(max=1) if hasattr(runner.policy, "std") else runner.policy.log_std.clamp_(max=0)
    cfg = make_env_cfg(
        "go2",
        terrain="all",
        num_envs=num_envs,
        evaluation=True,
        reward_revision=checkpoint_reward_revision({"extra": {
            "environment_contract": getattr(runner, "environment_contract", None),
            "environment": getattr(runner, "_resume_environment", None),
        }}),
        control_revision=getattr(runner, "control_revision", LEGACY_CONTROL_REVISION),
        environment_revision=getattr(runner, "environment_revision", LEGACY_ENVIRONMENT_REVISION),
        camera_implementation=(getattr(runner, "environment_contract", None) or {})
        .get("config", {})
        .get("camera_implementation", "isaaclab_sparse"),
    )
    if not hasattr(runner.cfg, "algorithm_revision"):
        cfg.reward_semantics = cfg.curriculum_semantics = "legacy_v2"
    cfg.sim.device = args.device
    cfg.seed = 7
    cfg.terrain.terrain_generator.num_rows = 16
    cfg.terrain.terrain_generator.num_cols = 4
    names = ("wmp_gap", "wmp_climb", "rough_flat", "stairs_up")
    cfg.terrain.terrain_generator.sub_terrains = {
        name: cfg.terrain.terrain_generator.sub_terrains[name] for name in names
    }
    for terrain in cfg.terrain.terrain_generator.sub_terrains.values():
        terrain.proportion = 0.25
    if os.environ.get("WMP_DIAG_NO_OBSTACLES") == "1":
        from isaaclab.terrains import MeshPlaneTerrainCfg

        for name in ("wmp_gap", "wmp_climb"):
            cfg.terrain.terrain_generator.sub_terrains[name] = MeshPlaneTerrainCfg(proportion=0.25)
    cfg.logical_camera_envs = num_envs if depth_mode == "real" else max(4, num_envs // 4)
    cfg.command_x_range = cfg.jump_command_x_range = cfg.tilt_command_x_range = (speed, speed)
    cfg.command_y_range = cfg.command_yaw_range = (0, 0)
    cfg.terrain.max_init_terrain_level = 0
    cfg.viewer.resolution = (960, 540)
    env = gym.make(
        "Isaac-WMP-Go2-Rough-Direct-v0", cfg=cfg, render_mode="rgb_array" if video_enabled else None
    ).unwrapped
    env.update_reward_curriculum(getattr(runner, "curriculum_iteration", runner.iteration))
    instrument(env)
    observation = env.reset()[0]["policy"]
    history = HistoryBuffer(num_envs, env.device)
    history.append(observation)
    action_history = torch.zeros(num_envs, 5, 12, device=env.device)
    first = torch.ones(num_envs, device=env.device)
    feature = torch.zeros(num_envs, runner.cfg.world_feature_dim, device=env.device)
    state = None
    writers = {}
    representatives = {name: env.terrain_names.index(name) for name in ("gap", "climb")}
    if video_enabled:
        import imageio.v2 as imageio

        writers = {
            name: imageio.get_writer(output / f"{name}.mp4", fps=10, codec="libx264") for name in representatives
        }
    try:
        with torch.inference_mode():
            for step in range(args.steps):
                if step % runner.cfg.world_update_interval == 0:
                    prop = proprioception(observation)
                    depth = env.get_depth()
                    if depth_mode == "mixed":
                        predicted = runner.depth_predictor(env.get_forward_height_map(), prop)
                        depth = torch.where(env.camera_mask[:, None, None, None], depth, predicted)
                    state, _ = runner.world_model.observe_step(
                        prop, depth, action_history.flatten(1), first, state, sample=mode != "mean"
                    )
                    feature = runner.world_model.policy_feature(state)
                    first.zero_()
                mean = runner.policy.act_inference(observation, history.flatten(), feature)
                action = mean if mode == "mean" else mean + policy_std(runner) * torch.randn_like(mean)
                observation_dict, _, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                row = {key: value.cpu().numpy() for key, value in env._diagnostic_record.items()}
                row.update(
                    terminated=terminated.cpu().numpy(),
                    truncated=truncated.cpu().numpy(),
                    raw_action=action.cpu().numpy(),
                    action_mean=mean.cpu().numpy(),
                )
                records.append(row)
                action_history = torch.cat((action_history[:, 1:], action[:, None]), 1)
                action_history[done] = 0
                feature[done] = 0
                first[done] = 1
                history.reset(done.nonzero(as_tuple=False).flatten())
                observation = observation_dict["policy"]
                history.append(observation)
                if writers and step % 5 == 0:
                    import imageio.v2 as imageio

                    for name, writer in writers.items():
                        position = env._robot.data.root_pos_w[representatives[name]].cpu().numpy()
                        env.sim.set_camera_view(position + np.array((-1.8, -3.0, 1.3)), position + (0.6, 0, 0))
                        frame = env.render()
                        writer.append_data(frame)
                        if step in (0, 100, 300, 600):
                            imageio.imwrite(output / f"{name}_{step:04d}.png", frame)
        arrays = {key: np.stack([row[key] for row in records]) for key in records[0]}
        np.savez_compressed(output / "trace.npz", **arrays)
        promotion_distance = 4.0 if cfg.curriculum_semantics == "upstream_v3" else 3.0
        summaries = {}
        for name in sorted(set(env.terrain_names)):
            ids = [i for i, terrain in enumerate(env.terrain_names) if terrain == name]
            pos = arrays["pos"][:, ids]
            summaries[name] = dict(
                environments=ids,
                max_progress_m=float(pos[..., 0].max()),
                last_progress_mean_m=float(pos[-1, :, 0].mean()),
                mean_vx=float(arrays["velocity"][:, ids, 0].mean()),
                mean_task_reward=float(arrays["task_reward"][:, ids].mean()),
                zero_task_fraction=float((arrays["task_reward"][:, ids] <= 0).mean()),
                terminations=int(arrays["terminated"][:, ids].sum()),
                timeouts=int(arrays["truncated"][:, ids].sum()),
                reached_curriculum_distance=int((pos[..., 0].max(axis=0) > promotion_distance).sum()),
                action_saturation_per_joint=(np.abs(arrays["raw_action"][:, ids]) > 6).mean(axis=(0, 1)).tolist(),
                mean_reward_components={
                    key: float(value[:, ids].mean()) for key, value in arrays.items() if key.startswith("reward_")
                },
            )
        result = dict(
            checkpoint=str(checkpoint),
            checkpoint_iteration=runner.iteration,
            mode=mode,
            depth=depth_mode,
            speed=speed,
            steps=args.steps,
            original_std=original_std,
            promotion_distance=promotion_distance,
            reward_semantics=cfg.reward_semantics,
            control_revision=cfg.control_revision,
            evaluated_std=policy_std(runner).cpu().tolist(),
            joint_order=list(wmp_lab.JOINT_ORDER),
            terrain_names=env.terrain_names,
            joint_velocity_limits=env._robot.data.joint_vel_limits[0, env._joint_ids].cpu().tolist(),
            joint_effort_limits=env._robot.data.joint_effort_limits[0, env._joint_ids].cpu().tolist(),
            joint_position_limits=env._robot.data.joint_pos_limits[0, env._joint_ids].cpu().tolist(),
            real_camera_mask=env.camera_mask.cpu().tolist(),
            summaries=summaries,
            obstacles_removed=os.environ.get("WMP_DIAG_NO_OBSTACLES") == "1",
        )
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(summaries, indent=2), flush=True)
        print(f"[PASS] replay saved {output}", flush=True)
    finally:
        for writer in writers.values():
            writer.close()
        env.close()


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
