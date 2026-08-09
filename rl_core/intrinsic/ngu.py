"""Episodic and controllable-representation components used by NGU."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class EpisodicKNNNovelty:
    """NGU-style pseudo-count novelty over one episode's embeddings."""

    def __init__(
        self,
        k: int = 10,
        kernel_epsilon: float = 1e-4,
        cluster_distance: float = 8e-3,
        c: float = 1e-3,
    ) -> None:
        if k <= 0 or kernel_epsilon <= 0 or cluster_distance <= 0 or c < 0:
            raise ValueError("invalid episodic novelty parameters")
        self.k = k
        self.kernel_epsilon = kernel_epsilon
        self.cluster_distance = cluster_distance
        self.c = c
        self._memory: list[np.ndarray] = []

    @property
    def size(self) -> int:
        """Return the number of embeddings remembered this episode."""
        return len(self._memory)

    def reset(self) -> None:
        """Clear the per-episode memory."""
        self._memory.clear()

    def reward(self, embedding: np.ndarray, update: bool = True) -> float:
        """Compute pseudo-count novelty, then optionally add this embedding."""
        point = np.asarray(embedding, dtype=np.float64).reshape(-1)
        if point.size == 0 or not np.all(np.isfinite(point)):
            raise ValueError("embedding must be non-empty and finite")
        if not self._memory:
            reward = 1.0 / self.c if self.c else 1.0
        else:
            memory = np.stack(self._memory)
            distances = np.linalg.norm(memory - point, axis=1)
            nearest = np.partition(distances, min(self.k, len(distances)) - 1)[: self.k]
            kernels = self.kernel_epsilon / (nearest + self.kernel_epsilon)
            kernels = kernels * (nearest <= self.cluster_distance)
            pseudo_count = float(np.sum(kernels))
            reward = 1.0 / (math.sqrt(pseudo_count) + self.c)
        if update:
            self._memory.append(point.copy())
        return float(reward)


class InverseDynamicsEmbedding(nn.Module):
    """A learned embedding constrained to retain action-controllable state."""

    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_dim: int,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if not observation_shape or action_dim <= 1 or embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("invalid inverse-dynamics dimensions")
        input_size = int(np.prod(observation_shape))
        self.observation_shape = observation_shape
        self.encoder = nn.Sequential(nn.Linear(input_size, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embedding_dim))
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim)
        )

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode a batch of observations into controllable features."""
        if observations.ndim < 2:
            raise ValueError("observations must include a batch dimension")
        return self.encoder(observations.reshape(observations.shape[0], -1).float())

    def forward(self, observations: torch.Tensor, next_observations: torch.Tensor) -> torch.Tensor:
        """Predict actions from encoded state transitions."""
        return self.predictor(torch.cat((self.encode(observations), self.encode(next_observations)), dim=-1))

    def loss(self, observations: torch.Tensor, next_observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return inverse-dynamics cross entropy for a transition batch."""
        return nn.functional.cross_entropy(self(observations, next_observations), actions.long())

    def train_step(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        actions: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Optimize the embedding and action predictor once."""
        optimizer.zero_grad(set_to_none=True)
        loss = self.loss(observations, next_observations, actions)
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())
