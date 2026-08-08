"""Paper-faithful clipped Proximal Policy Optimization components."""

from rl_core.algorithms.ppo.model import AtariActorCritic
from rl_core.algorithms.ppo.rollout import RolloutBatch, RolloutStorage, generalized_advantage_estimation
from rl_core.algorithms.ppo.trainer import PPOConfig, PPOTrainer, PPOUpdateMetrics, clipped_surrogate_loss

__all__ = [
    "AtariActorCritic",
    "PPOConfig",
    "PPOTrainer",
    "PPOUpdateMetrics",
    "RolloutBatch",
    "RolloutStorage",
    "clipped_surrogate_loss",
    "generalized_advantage_estimation",
]
