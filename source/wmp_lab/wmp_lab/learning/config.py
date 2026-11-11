# SPDX-License-Identifier: Apache-2.0
"""Training configurations for paper-scale and low-memory correctness runs."""

from dataclasses import asdict, dataclass, replace

ALGORITHM_REVISION = "gpufree_4096_v1"
REFERENCE_ENVS = 4096
REFERENCE_ROLLOUT_STEPS = 24
REFERENCE_BATCH_SIZE = REFERENCE_ENVS * REFERENCE_ROLLOUT_STEPS


@dataclass(frozen=True)
class LearningCfg:
    algorithm_revision: str = ALGORITHM_REVISION
    num_envs: int = REFERENCE_ENVS
    rollout_steps: int = REFERENCE_ROLLOUT_STEPS
    reference_num_envs: int | None = REFERENCE_ENVS
    collection_chunks: int = 1
    completed_episodes_per_env: int = 1
    ppo_micro_batch_size: int = 6144
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
    replay_mode: str = "episodes"
    world_train_start: int = 10_000
    reward_loss_scale: float = 0.0
    robot: str = "go1"
    amp_hidden: tuple[int, ...] = (1024, 512)
    world_batch_size: int = 16
    world_sequence_length: int = 64
    replay_size: int = 1_000_000
    lr: float = 1.0e-3
    world_lr: float = 1.0e-4
    depth_lr: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    amp_reward_coef: float = 0.01
    amp_task_reward_lerp: float = 0.3

    @property
    def aligned_schedule(self) -> bool:
        return self.algorithm_revision == ALGORITHM_REVISION and self.reference_num_envs is not None

    @property
    def collection_steps(self) -> int:
        return self.rollout_steps * self.collection_chunks

    @property
    def batch_samples(self) -> int:
        return self.num_envs * self.collection_steps

    def validate_schedule(self):
        if not self.aligned_schedule:
            return
        expected = dict(
            reference_num_envs=REFERENCE_ENVS, rollout_steps=REFERENCE_ROLLOUT_STEPS,
            learning_epochs=5, mini_batches=4, world_train_steps=10, depth_train_steps=50,
            world_batch_size=16, world_sequence_length=64, depth_batch_size=1024, world_train_start=10000,
        )
        if any(getattr(self, key) != value for key, value in expected.items()):
            raise ValueError("aligned training requires the gpufree 4096/24 update configuration")
        if self.num_envs * self.collection_chunks != REFERENCE_ENVS:
            raise ValueError("aligned collection must contain exactly 4096 logical environments")
        if self.completed_episodes_per_env != self.collection_chunks:
            raise ValueError("completed episode slots must match collection chunks")
        if self.ppo_micro_batch_size < 1:
            raise ValueError("PPO micro batch size must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


PAPER_CFG = LearningCfg()
SMOKE_CFG = LearningCfg(
    reference_num_envs=None,
    replay_mode="ring",
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


def get_learning_cfg(smoke: bool = False, *, num_envs: int | None = None) -> LearningCfg:
    """用完整参考批次对齐采样、PPO 和感知更新；小环境以多段采集补齐。"""
    base = SMOKE_CFG if smoke else PAPER_CFG
    if num_envs is None:
        return base
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    if smoke:
        return replace(base, num_envs=num_envs)
    if num_envs > REFERENCE_ENVS or REFERENCE_ENVS % num_envs:
        raise ValueError("num_envs must divide 4096 exactly; use 1024, 2048 or 4096")
    chunks = REFERENCE_ENVS // num_envs
    cfg = replace(base, num_envs=num_envs, collection_chunks=chunks, completed_episodes_per_env=chunks)
    cfg.validate_schedule()
    return cfg
