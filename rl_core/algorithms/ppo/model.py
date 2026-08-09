"""Actor-critic networks used by the Atari PPO reproduction."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class AtariActorCritic(nn.Module):
    """The shared convolutional policy/value network used for Atari PPO."""

    def __init__(self, action_count: int, input_channels: int = 4) -> None:
        super().__init__()
        if action_count <= 0:
            raise ValueError("action_count must be positive")
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
        )
        self.policy = nn.Linear(512, action_count)
        self.value = nn.Linear(512, 1)
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)

    def forward(self, observations: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        """Return the action distribution and scalar value estimates."""
        encoded = self.encoder(observations.float() / 255.0)
        return Categorical(logits=self.policy(encoded)), self.value(encoded).squeeze(-1)

    @torch.no_grad()
    def act(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions for collection and return action log probabilities."""
        distribution, values = self(observations)
        actions = distribution.sample()
        return actions, distribution.log_prob(actions), values
