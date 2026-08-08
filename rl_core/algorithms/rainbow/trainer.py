"""Mathematically explicit Rainbow DQN update rule."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor

from rl_core.algorithms.rainbow.network import RainbowNetwork


@dataclass(frozen=True)
class RainbowConfig:
    """Rainbow settings; each extension can be disabled independently."""

    action_dim: int
    input_channels: int = 4
    gamma: float = 0.99
    learning_rate: float = 6.25e-5
    adam_epsilon: float = 1.5e-4
    target_update_interval: int = 2_000
    double_q: bool = True
    dueling: bool = True
    prioritized_replay: bool = True
    n_step: int = 3
    distributional: bool = True
    noisy: bool = True
    atom_count: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    priority_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        """Validate settings that remain meaningful when components are disabled."""
        if self.action_dim < 1 or self.input_channels < 1 or self.n_step < 1:
            raise ValueError("action_dim, input_channels, and n_step must be positive")
        if not 0.0 <= self.gamma <= 1.0 or self.learning_rate <= 0.0 or self.adam_epsilon <= 0.0:
            raise ValueError("invalid optimizer or discount settings")
        if self.target_update_interval < 1 or self.priority_epsilon <= 0.0:
            raise ValueError("target_update_interval and priority_epsilon must be positive")
        if self.distributional and (self.atom_count < 2 or self.v_max <= self.v_min):
            raise ValueError("C51 requires at least two atoms and an ordered support")


class Rainbow:
    """Rainbow learner with Double DQN, C51, PER, n-step, and noisy-net hooks."""

    def __init__(self, config: RainbowConfig, device: torch.device | str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        network_arguments = {
            "action_dim": config.action_dim,
            "input_channels": config.input_channels,
            "dueling": config.dueling,
            "distributional": config.distributional,
            "noisy": config.noisy,
            "atom_count": config.atom_count,
            "v_min": config.v_min,
            "v_max": config.v_max,
        }
        self.online_network = RainbowNetwork(**network_arguments).to(self.device)
        self.target_network = copy.deepcopy(self.online_network).to(self.device)
        self.target_network.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=config.learning_rate, eps=config.adam_epsilon
        )
        self.optimizer_updates = 0

    def select_action(
        self, observation: np.ndarray | Tensor, epsilon: float, rng: np.random.Generator, *, evaluation: bool = False
    ) -> int:
        """Choose a noisy-greedy or epsilon-greedy action from one stacked state."""
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if rng.random() < epsilon:
            return int(rng.integers(self.config.action_dim))
        observation_tensor = torch.as_tensor(observation, device=self.device)
        if observation_tensor.ndim == 3:
            observation_tensor = observation_tensor.unsqueeze(0)
        if self.config.noisy and not evaluation:
            self.online_network.reset_noise()
        with torch.no_grad():
            return int(self.online_network.q_values(observation_tensor, use_noise=not evaluation).argmax(dim=1).item())

    def train_step(self, batch: dict[str, Tensor]) -> dict[str, Tensor | float]:
        """Apply one weighted Rainbow update and return per-sample priorities."""
        observations = batch["obs"].to(self.device)
        actions = batch["action"].to(self.device).long().reshape(-1, 1)
        rewards = batch["reward"].to(self.device).float().reshape(-1, 1)
        next_observations = batch["next_obs"].to(self.device)
        dones = batch["done"].to(self.device).float().reshape(-1, 1)
        discounts = batch.get("discount")
        if discounts is None:
            discounts = torch.full_like(rewards, self.config.gamma**self.config.n_step)
        else:
            discounts = discounts.to(self.device).float().reshape(-1, 1)
        weights = batch.get("weights")
        if weights is None or not self.config.prioritized_replay:
            weights = torch.ones_like(rewards)
        else:
            weights = weights.to(self.device).float().reshape(-1, 1)
        if self.config.noisy:
            self.online_network.reset_noise()
            self.target_network.reset_noise()

        if self.config.distributional:
            per_item_loss = self._distributional_loss(
                observations, actions, rewards, next_observations, dones, discounts
            )
        else:
            per_item_loss = self._scalar_loss(observations, actions, rewards, next_observations, dones, discounts)
        loss = (weights.squeeze(1) * per_item_loss).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.optimizer_updates += 1
        target_updated = self.optimizer_updates % self.config.target_update_interval == 0
        if target_updated:
            self.target_network.load_state_dict(self.online_network.state_dict())
        return {
            "loss/q": float(loss.item()),
            "priority_errors": per_item_loss.detach() + self.config.priority_epsilon,
            "optimizer_updates": float(self.optimizer_updates),
            "target_updated": float(target_updated),
        }

    def _scalar_loss(
        self,
        observations: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_observations: Tensor,
        dones: Tensor,
        discounts: Tensor,
    ) -> Tensor:
        values = self.online_network.q_values(observations).gather(1, actions)
        with torch.no_grad():
            target_values = self.target_network.q_values(next_observations)
            if self.config.double_q:
                next_actions = self.online_network.q_values(next_observations).argmax(dim=1, keepdim=True)
                next_values = target_values.gather(1, next_actions)
            else:
                next_values = target_values.max(dim=1, keepdim=True).values
            targets = rewards + (1.0 - dones) * discounts * next_values
        return functional.smooth_l1_loss(values, targets, reduction="none").squeeze(1)

    def _distributional_loss(
        self,
        observations: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_observations: Tensor,
        dones: Tensor,
        discounts: Tensor,
    ) -> Tensor:
        current_distribution = (
            self.online_network.distribution(observations)
            .gather(1, actions.unsqueeze(-1).expand(-1, 1, self.config.atom_count))
            .squeeze(1)
        )
        with torch.no_grad():
            target_distribution = self.target_network.distribution(next_observations)
            if self.config.double_q:
                next_actions = self.online_network.q_values(next_observations).argmax(dim=1)
            else:
                next_actions = self.target_network.q_values(next_observations).argmax(dim=1)
            next_distribution = target_distribution[torch.arange(len(actions), device=self.device), next_actions]
            projection = self.project_distribution(next_distribution, rewards, dones, discounts)
        return -(projection * current_distribution.clamp_min(1e-8).log()).sum(dim=1)

    def project_distribution(
        self, next_distribution: Tensor, rewards: Tensor, dones: Tensor, discounts: Tensor
    ) -> Tensor:
        """Project a Bellman-updated categorical distribution onto the fixed C51 support."""
        support = cast(Tensor, self.online_network.support)
        delta = (support[-1] - support[0]) / (self.config.atom_count - 1)
        target_support = rewards + (1.0 - dones) * discounts * support.unsqueeze(0)
        target_support = target_support.clamp(float(support[0]), float(support[-1]))
        positions = (target_support - support[0]) / delta
        lower = positions.floor().long().clamp(0, self.config.atom_count - 1)
        upper = positions.ceil().long().clamp(0, self.config.atom_count - 1)
        projected = torch.zeros_like(next_distribution)
        lower_weight = upper.float() - positions
        lower_weight += (lower == upper).float()
        upper_weight = positions - lower.float()
        projected.scatter_add_(1, lower, next_distribution * lower_weight)
        projected.scatter_add_(1, upper, next_distribution * upper_weight)
        return projected

    def state_dicts(self) -> dict[str, Any]:
        """Return all state necessary to continue the update schedule."""
        return {
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_updates": self.optimizer_updates,
        }

    def load_state_dicts(self, state_dicts: dict[str, Any]) -> None:
        """Restore a checkpoint written by :meth:`state_dicts`."""
        self.online_network.load_state_dict(state_dicts["online_network"])
        self.target_network.load_state_dict(state_dicts["target_network"])
        self.optimizer.load_state_dict(state_dicts["optimizer"])
        self.optimizer_updates = int(state_dicts["optimizer_updates"])
