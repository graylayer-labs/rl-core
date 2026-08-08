"""A2C boundary used to make PPO's multi-epoch clipping explicit."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class A2CObjective:
    """One-pass on-policy actor-critic objective, intentionally not PPO."""

    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor


def a2c_objective(
    log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
) -> A2CObjective:
    """Build the unclipped, single-policy A2C objective components.

    A2C has no behaviour-policy likelihood ratio, clipping range, or repeated
    epochs.  Keeping this separate prevents silently presenting A2C as PPO.
    """
    if not (log_probabilities.shape == advantages.shape == values.shape == returns.shape):
        raise ValueError("A2C tensors must have matching shapes")
    return A2CObjective(
        policy_loss=-(log_probabilities * advantages.detach()).mean(),
        value_loss=0.5 * (returns - values).square().mean(),
        entropy=entropy.mean(),
    )
