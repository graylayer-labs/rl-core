"""Atari network components used by the Rainbow agent."""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class NoisyLinear(nn.Module):
    """Factorised Gaussian noisy linear layer from Fortunato et al. (2018)."""

    def __init__(self, in_features: int, out_features: int, sigma_zero: float = 0.5) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1 or sigma_zero <= 0.0:
            raise ValueError("layer dimensions and sigma_zero must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self._sigma_zero = sigma_zero
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        """Initialize mean and standard-deviation parameters as specified."""
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        sigma = self._sigma_zero / math.sqrt(self.in_features)
        self.weight_sigma.data.fill_(sigma)
        self.bias_sigma.data.fill_(sigma)

    def reset_noise(self) -> None:
        """Draw a fresh factorised noise sample."""
        input_noise = self._scaled_noise(self.in_features, self.weight_mu.device)
        output_noise = self._scaled_noise(self.out_features, self.weight_mu.device)
        cast(Tensor, self.weight_epsilon).copy_(output_noise.outer(input_noise))
        cast(Tensor, self.bias_epsilon).copy_(output_noise)

    def forward(self, inputs: Tensor, use_noise: bool = True) -> Tensor:
        """Apply the noisy affine transform, optionally using its mean only."""
        if use_noise:
            weight = self.weight_mu + self.weight_sigma * cast(Tensor, self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma * cast(Tensor, self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return functional.linear(inputs, weight, bias)

    @staticmethod
    def _scaled_noise(size: int, device: torch.device) -> Tensor:
        noise = torch.randn(size, device=device)
        return noise.sign() * noise.abs().sqrt()


class RainbowNetwork(nn.Module):
    """Convolutional Q-network with independently enabled Rainbow heads."""

    def __init__(
        self,
        action_dim: int,
        *,
        input_channels: int = 4,
        dueling: bool = True,
        distributional: bool = True,
        noisy: bool = True,
        atom_count: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ) -> None:
        super().__init__()
        if action_dim < 1 or input_channels < 1 or atom_count < 1 or v_max <= v_min:
            raise ValueError("invalid Rainbow network dimensions or support")
        self.action_dim = action_dim
        self.dueling = dueling
        self.distributional = distributional
        self.noisy = noisy
        self.atom_count = atom_count if distributional else 1
        self.register_buffer("support", torch.linspace(v_min, v_max, self.atom_count))
        self.convolutions = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        linear: type[NoisyLinear] | type[nn.Linear] = NoisyLinear if noisy else nn.Linear
        self.feature = linear(3136, 512)
        if dueling:
            self.value = linear(512, self.atom_count)
            self.advantage = linear(512, action_dim * self.atom_count)
        else:
            self.value = None
            self.advantage = linear(512, action_dim * self.atom_count)

    def reset_noise(self) -> None:
        """Resample every noisy layer without affecting deterministic layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def distribution(self, observations: Tensor, *, use_noise: bool | None = None) -> Tensor:
        """Return action-value distributions, or one-atom values when C51 is off."""
        use_noise = self.noisy if use_noise is None else use_noise
        inputs = observations.float().div(255.0) if observations.dtype == torch.uint8 else observations
        features = functional.relu(self._apply_linear(self.feature, self.convolutions(inputs).flatten(1), use_noise))
        advantage = self._apply_linear(self.advantage, features, use_noise).view(-1, self.action_dim, self.atom_count)
        if self.value is not None:
            value = self._apply_linear(self.value, features, use_noise).view(-1, 1, self.atom_count)
            logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            logits = advantage
        return functional.softmax(logits, dim=-1) if self.distributional else logits

    def q_values(self, observations: Tensor, *, use_noise: bool | None = None) -> Tensor:
        """Return expected action values for either scalar or C51 output heads."""
        outputs = self.distribution(observations, use_noise=use_noise)
        if self.distributional:
            return (outputs * cast(Tensor, self.support)).sum(dim=-1)
        return outputs.squeeze(-1)

    def forward(self, observations: Tensor) -> Tensor:
        """Return expected per-action values for ordinary module use."""
        return self.q_values(observations)

    @staticmethod
    def _apply_linear(layer: nn.Module, inputs: Tensor, use_noise: bool) -> Tensor:
        if isinstance(layer, NoisyLinear):
            return layer(inputs, use_noise)
        return layer(inputs)
