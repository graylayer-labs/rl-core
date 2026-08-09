"""Centered RMSProp variant used by the original Nature DQN implementation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


class DeepMindRMSprop(Optimizer):
    """RMSProp with a running gradient mean in the denominator.

    This follows the centered update used by DeepMind DQN:
    ``p -= lr * g / sqrt(E[g²] - E[g]² + eps)``.  Unlike PyTorch's RMSprop,
    ``eps`` lives inside the square root, which is materially significant for
    the paper's ``eps=0.01`` setting.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 0.00025,
        alpha: float = 0.95,
        eps: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError("lr must be non-negative")
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0, 1)")
        if eps < 0.0:
            raise ValueError("eps must be non-negative")
        if momentum < 0.0:
            raise ValueError("momentum must be non-negative")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        defaults = {
            "lr": lr,
            "alpha": alpha,
            "eps": eps,
            "momentum": momentum,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(  # ty: ignore[invalid-method-override]
        self, closure: Callable[[], float] | None = None
    ) -> float | None:
        """Perform one centered RMSProp update."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("DeepMindRMSprop does not support sparse gradients")
                if group["weight_decay"]:
                    gradient = gradient.add(parameter, alpha=group["weight_decay"])
                state = self.state[parameter]
                if not state:
                    state["square_avg"] = torch.zeros_like(parameter)
                    state["grad_avg"] = torch.zeros_like(parameter)
                    if group["momentum"]:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                square_avg = state["square_avg"]
                grad_avg = state["grad_avg"]
                alpha = group["alpha"]
                square_avg.mul_(alpha).addcmul_(gradient, gradient, value=1.0 - alpha)
                grad_avg.mul_(alpha).add_(gradient, alpha=1.0 - alpha)
                denominator = (square_avg - grad_avg.square()).add_(group["eps"]).sqrt_()
                update = gradient / denominator
                if group["momentum"]:
                    momentum_buffer = state["momentum_buffer"]
                    momentum_buffer.mul_(group["momentum"]).add_(update)
                    update = momentum_buffer
                parameter.add_(update, alpha=-group["lr"])
        return loss
