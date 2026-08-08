"""Random Network Distillation novelty and reward normalization."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from rl_core.utils.running_stats import RunningMeanStd


def _flatten_observations(observations: torch.Tensor) -> torch.Tensor:
    if observations.ndim < 2:
        raise ValueError("observations must include a batch dimension")
    return observations.reshape(observations.shape[0], -1).float()


def _network(input_size: int, hidden_sizes: Sequence[int], output_size: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_size
    for hidden in hidden_sizes:
        layers.extend((nn.Linear(width, hidden), nn.ReLU()))
        width = hidden
    layers.append(nn.Linear(width, output_size))
    return nn.Sequential(*layers)


class RNDModel(nn.Module):
    """Fixed random target plus a trainable predictor.

    This is intentionally observation-model agnostic: a paper runner can feed
    an Atari convolutional encoder while cheap readiness checks use flattened
    vectors.  The target's parameters are frozen by construction.
    """

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        feature_dim: int = 128,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        if not observation_shape or any(size <= 0 for size in observation_shape):
            raise ValueError("observation_shape must contain positive dimensions")
        if feature_dim <= 0 or not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("network dimensions must be positive")
        input_size = int(np.prod(observation_shape))
        self.observation_shape = observation_shape
        self.target = _network(input_size, hidden_sizes, feature_dim)
        self.predictor = _network(input_size, hidden_sizes, feature_dim)
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    def prediction_error(self, observations: torch.Tensor) -> torch.Tensor:
        """Return per-observation squared target-prediction error."""
        inputs = _flatten_observations(observations)
        with torch.no_grad():
            target = self.target(inputs)
        prediction = self.predictor(inputs)
        return torch.mean(torch.square(prediction - target), dim=-1)

    def loss(self, observations: torch.Tensor) -> torch.Tensor:
        """Mean RND prediction error suitable for predictor optimization."""
        return self.prediction_error(observations).mean()

    def train_step(self, observations: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        """Update only the predictor and return the scalar loss."""
        optimizer.zero_grad(set_to_none=True)
        loss = self.loss(observations)
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())


class RNDRewardNormalizer:
    """Normalize RND rewards by the variance of discounted novelty returns."""

    def __init__(self, gamma: float = 0.99, epsilon: float = 1e-8) -> None:
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be between zero and one")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.gamma = gamma
        self.epsilon = epsilon
        self.return_stats = RunningMeanStd()
        self._discounted_return = 0.0

    def reset(self) -> None:
        """Start a new episode while retaining cross-episode normalization."""
        self._discounted_return = 0.0

    def normalize(self, novelty: float, done: bool = False, update: bool = True) -> float:
        """Return variance-normalized novelty and optionally update statistics."""
        if novelty < 0 or not np.isfinite(novelty):
            raise ValueError("novelty must be finite and non-negative")
        self._discounted_return = self.gamma * self._discounted_return + float(novelty)
        if update:
            self.return_stats.update(self._discounted_return)
        reward = float(novelty / np.sqrt(self.return_stats.var + self.epsilon))
        if done:
            self.reset()
        return reward
