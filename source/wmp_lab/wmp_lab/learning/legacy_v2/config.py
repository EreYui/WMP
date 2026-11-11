# SPDX-License-Identifier: Apache-2.0
"""Training configurations for paper-scale and low-memory correctness runs."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LearningCfg:
    num_envs: int = 4096
    rollout_steps: int = 24
    learning_epochs: int = 5
    mini_batches: int = 4
    actor_hidden: tuple[int, ...] = (256, 128, 64)
    critic_hidden: tuple[int, ...] = (512, 256, 128)
    history_hidden: tuple[int, ...] = (256, 128)
    world_hidden: int = 512
    world_feature_dim: int = 512
    world_stochastic_dim: int = 32
    world_categories: int = 32
    world_update_interval: int = 5
    world_train_steps: int = 10
    depth_train_steps: int = 50
    depth_batch_size: int = 1024
    world_replay_capacity: int = 300_000
    world_train_start: int = 10_000
    reward_loss_scale: float = 0.0
    robot: str = "go2"
    amp_hidden: tuple[int, ...] = (1024, 512)
    world_batch_size: int = 16
    world_sequence_length: int = 64
    replay_size: int = 1_000_000
    lr: float = 1.0e-3
    world_lr: float = 1.0e-4
    depth_lr: float = 3.0e-4
    gamma: float = 0.998
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    amp_reward_coef: float = 0.01
    amp_task_reward_lerp: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)


PAPER_CFG = LearningCfg()
SMOKE_CFG = LearningCfg(
    num_envs=4,
    rollout_steps=4,
    learning_epochs=1,
    mini_batches=1,
    actor_hidden=(64, 32),
    critic_hidden=(64, 32),
    history_hidden=(64, 32),
    world_hidden=64,
    world_feature_dim=64,
    world_stochastic_dim=8,
    world_categories=8,
    world_train_steps=1,
    depth_train_steps=1,
    depth_batch_size=4,
    world_train_start=1,
    world_replay_capacity=128,
    amp_hidden=(64, 32),
    world_batch_size=4,
    world_sequence_length=4,
    replay_size=256,
)


def get_learning_cfg(smoke: bool = False) -> LearningCfg:
    return SMOKE_CFG if smoke else PAPER_CFG
