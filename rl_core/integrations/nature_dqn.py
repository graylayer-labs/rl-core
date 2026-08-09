"""A compact Nature-DQN integration for Atari smoke and qualifying runs.

This module intentionally owns only the algorithm update.  Environment stepping,
run artifacts, and budget accounting remain the responsibility of the
experiment runner so that their provenance can be recorded separately.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from rl_core.optim import DeepMindRMSprop


class NatureQNetwork(nn.Module):
    """The convolutional Q-network specified in Mnih et al. (2015)."""

    def __init__(self, action_dim: int, input_channels: int = 4) -> None:
        super().__init__()
        if action_dim < 1 or input_channels < 1:
            raise ValueError("action_dim and input_channels must be positive")
        self.convolutions = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(3136, 512), nn.ReLU(), nn.Linear(512, action_dim))

    def forward(self, observations: Tensor) -> Tensor:
        """Return Q-values for ``uint8`` or normalized stacked Atari frames."""
        if observations.ndim != 4:
            raise ValueError("observations must have shape (batch, channels, height, width)")
        if observations.dtype == torch.uint8:
            observations = observations.float().div_(255.0)
        return self.head(self.convolutions(observations))


@dataclass(frozen=True)
class NatureDQNConfig:
    """Algorithm settings fixed by the 2015 DQN protocol."""

    action_dim: int
    input_channels: int = 4
    gamma: float = 0.99
    learning_rate: float = 0.00025
    rmsprop_alpha: float = 0.95
    rmsprop_epsilon: float = 0.01
    rmsprop_momentum: float = 0.95
    target_update_interval: int = 10_000

    def __post_init__(self) -> None:
        """Validate independently meaningful algorithm settings."""
        if self.action_dim < 1 or self.input_channels < 1:
            raise ValueError("action_dim and input_channels must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.target_update_interval < 1:
            raise ValueError("target_update_interval must be positive")


class NatureDQN:
    """Nature DQN update rule with explicit optimizer-update target scheduling.

    The target network is copied exactly after every configured number of
    **optimizer updates**, rather than environment frames.  This makes the
    schedule unambiguous when a runner uses different update frequencies.
    No gradient clipping is applied: the Huber loss is the paper-era stabilizer.
    """

    def __init__(self, config: NatureDQNConfig, device: torch.device | str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.online_network = NatureQNetwork(config.action_dim, config.input_channels).to(self.device)
        self.target_network = copy.deepcopy(self.online_network).to(self.device)
        self.target_network.requires_grad_(False)
        self.target_network.eval()
        self.optimizer = DeepMindRMSprop(
            self.online_network.parameters(),
            lr=config.learning_rate,
            alpha=config.rmsprop_alpha,
            eps=config.rmsprop_epsilon,
            momentum=config.rmsprop_momentum,
        )
        self.optimizer_updates = 0

    def select_action(self, observation: np.ndarray | Tensor, epsilon: float, rng: np.random.Generator) -> int:
        """Select an ε-greedy Atari action using the caller-owned RNG."""
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        if rng.random() < epsilon:
            return int(rng.integers(self.config.action_dim))
        observation_tensor = torch.as_tensor(observation, device=self.device)
        if observation_tensor.ndim == 3:
            observation_tensor = observation_tensor.unsqueeze(0)
        with torch.no_grad():
            return int(self.online_network(observation_tensor).argmax(dim=1).item())

    def train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """Apply one Huber-loss Q-learning update and maybe sync the target net."""
        observations = batch["obs"].to(self.device)
        actions = batch["action"].to(self.device).long()
        rewards = batch["reward"].to(self.device).float()
        next_observations = batch["next_obs"].to(self.device)
        dones = batch["done"].to(self.device).float()
        if actions.ndim == 1:
            actions = actions.unsqueeze(1)
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(1)
        if dones.ndim == 1:
            dones = dones.unsqueeze(1)

        with torch.no_grad():
            next_values = self.target_network(next_observations).amax(dim=1, keepdim=True)
            targets = rewards + (1.0 - dones) * self.config.gamma * next_values
        values = self.online_network(observations).gather(1, actions)
        loss = functional.smooth_l1_loss(values, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.optimizer_updates += 1
        target_updated = self.optimizer_updates % self.config.target_update_interval == 0
        if target_updated:
            self.target_network.load_state_dict(self.online_network.state_dict())
        return {
            "loss/q": float(loss.item()),
            "q/mean": float(values.mean().item()),
            "q/target_mean": float(targets.mean().item()),
            "optimizer_updates": float(self.optimizer_updates),
            "target_updated": float(target_updated),
        }

    def state_dicts(self) -> dict[str, Any]:
        """Return all state needed to resume the algorithm update schedule."""
        return {
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "optimizer_updates": self.optimizer_updates,
        }

    def load_state_dicts(self, state_dicts: dict[str, Any]) -> None:
        """Restore a checkpoint created by :meth:`state_dicts`."""
        self.online_network.load_state_dict(state_dicts["online_network"])
        self.target_network.load_state_dict(state_dicts["target_network"])
        self.optimizer.load_state_dict(state_dicts["optimizer"])
        self.optimizer_updates = int(state_dicts["optimizer_updates"])
