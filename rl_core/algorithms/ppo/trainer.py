"""Clipped-surrogate PPO optimization, kept independent of environment code."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rl_core.algorithms.ppo.rollout import RolloutBatch


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters for the 2017 clipped-PPO update."""

    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5
    epochs: int = 4
    minibatches: int = 4

    def __post_init__(self) -> None:
        """Reject invalid update schedules before collecting expensive data."""
        if self.learning_rate <= 0 or not 0 < self.clip_range < 1:
            raise ValueError("learning_rate must be positive and clip_range must be in (0, 1)")
        if self.epochs <= 0 or self.minibatches <= 0:
            raise ValueError("epochs and minibatches must be positive")


@dataclass(frozen=True)
class PPOUpdateMetrics:
    """Averaged diagnostics from a complete PPO update."""

    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    optimizer_steps: int


def clipped_surrogate_loss(
    new_log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return PPO's clipped policy loss, ratios, and clipping mask.

    The loss is the negative of the paper's clipped surrogate objective.
    """
    if new_log_probabilities.shape != old_log_probabilities.shape or advantages.shape != old_log_probabilities.shape:
        raise ValueError("PPO log probabilities and advantages must have matching shapes")
    ratios = (new_log_probabilities - old_log_probabilities).exp()
    unclipped = ratios * advantages
    clipped = ratios.clamp(1.0 - clip_range, 1.0 + clip_range) * advantages
    loss = -torch.minimum(unclipped, clipped).mean()
    clipped_mask = (ratios - 1.0).abs() > clip_range
    return loss, ratios, clipped_mask


class PPOTrainer:
    """Owns optimizer state for several minibatch epochs over one rollout."""

    def __init__(self, model: nn.Module, config: PPOConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.optimizer_steps = 0

    def update(self, batch: RolloutBatch) -> PPOUpdateMetrics:
        """Optimize the clipped objective across shuffled minibatches."""
        if batch.size < self.config.minibatches:
            raise ValueError("rollout batch is smaller than the number of minibatches")
        advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std(unbiased=False) + 1e-8)
        indices = torch.randperm(batch.size, device=self.device)
        chunks = torch.tensor_split(indices, self.config.minibatches)
        losses: list[tuple[float, float, float, float, float]] = []
        for _ in range(self.config.epochs):
            indices = indices[torch.randperm(batch.size, device=self.device)]
            chunks = torch.tensor_split(indices, self.config.minibatches)
            for minibatch in chunks:
                distribution, values = self.model(batch.observations[minibatch])
                log_probabilities = distribution.log_prob(batch.actions[minibatch])
                policy_loss, _ratios, clipped = clipped_surrogate_loss(
                    log_probabilities,
                    batch.old_log_probabilities[minibatch],
                    advantages[minibatch],
                    self.config.clip_range,
                )
                value_loss = 0.5 * (batch.returns[minibatch] - values).square().mean()
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss + self.config.value_coefficient * value_loss - self.config.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_gradient_norm)
                self.optimizer.step()
                self.optimizer_steps += 1
                approximate_kl = (batch.old_log_probabilities[minibatch] - log_probabilities).mean()
                losses.append(
                    (
                        float(policy_loss.detach()),
                        float(value_loss.detach()),
                        float(entropy.detach()),
                        float(approximate_kl.detach()),
                        float(clipped.float().mean().detach()),
                    )
                )
        means = torch.tensor(losses).mean(dim=0).tolist()
        return PPOUpdateMetrics(*[float(value) for value in means], optimizer_steps=self.optimizer_steps)
