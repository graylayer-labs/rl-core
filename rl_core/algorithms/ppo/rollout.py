"""On-policy rollout storage and generalized advantage estimation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RolloutBatch:
    """A flattened batch collected from complete vector-environment rollouts."""

    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probabilities: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor

    @property
    def size(self) -> int:
        """Return the number of state-action samples in this batch."""
        return int(self.actions.numel())


def generalized_advantage_estimation(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    bootstrap_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE-Lambda advantages and bootstrapped value targets.

    Inputs are shaped ``[time, environments]`` and ``dones`` is one for a
    transition that must not bootstrap.  The function is deliberately kept
    separate from collection so its termination semantics are easy to test.
    """
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be probabilities")
    if rewards.shape != values.shape or dones.shape != values.shape:
        raise ValueError("rewards, values, and dones must have matching shapes")
    if bootstrap_value.shape != values.shape[1:]:
        raise ValueError("bootstrap_value must have one value per environment")

    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros_like(bootstrap_value)
    next_value = bootstrap_value
    for timestep in range(rewards.shape[0] - 1, -1, -1):
        nonterminal = 1.0 - dones[timestep]
        delta = rewards[timestep] + gamma * next_value * nonterminal - values[timestep]
        next_advantage = delta + gamma * gae_lambda * nonterminal * next_advantage
        advantages[timestep] = next_advantage
        next_value = values[timestep]
    return advantages, advantages + values


class RolloutStorage:
    """Fixed-length storage for one synchronous PPO collection phase."""

    def __init__(self, steps: int, environments: int, observation_shape: tuple[int, ...], device: torch.device) -> None:
        if steps <= 0 or environments <= 0:
            raise ValueError("steps and environments must be positive")
        self.steps = steps
        self.environments = environments
        self.device = device
        self.observations = torch.empty((steps, environments, *observation_shape), dtype=torch.uint8, device=device)
        self.actions = torch.empty((steps, environments), dtype=torch.long, device=device)
        self.log_probabilities = torch.empty((steps, environments), dtype=torch.float32, device=device)
        self.rewards = torch.empty((steps, environments), dtype=torch.float32, device=device)
        self.dones = torch.empty((steps, environments), dtype=torch.float32, device=device)
        self.values = torch.empty((steps, environments), dtype=torch.float32, device=device)
        self._position = 0

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        log_probability: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Append a vectorized transition."""
        if self._position >= self.steps:
            raise RuntimeError("rollout storage is full")
        index = self._position
        self.observations[index].copy_(observation)
        self.actions[index].copy_(action)
        self.log_probabilities[index].copy_(log_probability)
        self.rewards[index].copy_(reward)
        self.dones[index].copy_(done)
        self.values[index].copy_(value)
        self._position += 1

    def batch(self, bootstrap_value: torch.Tensor, gamma: float, gae_lambda: float) -> RolloutBatch:
        """Finalize the rollout into the flattened optimization batch."""
        if self._position != self.steps:
            raise RuntimeError("cannot optimize an incomplete rollout")
        advantages, returns = generalized_advantage_estimation(
            self.rewards, self.values, self.dones, bootstrap_value, gamma, gae_lambda
        )
        return RolloutBatch(
            observations=self.observations.flatten(0, 1),
            actions=self.actions.flatten(),
            old_log_probabilities=self.log_probabilities.flatten(),
            advantages=advantages.flatten(),
            returns=returns.flatten(),
            old_values=self.values.flatten(),
        )
