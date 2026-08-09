"""Separate A2C baseline primitives; this is not an alias for PPO."""

from rl_core.algorithms.a2c.trainer import A2CObjective, a2c_objective

__all__ = ["A2CObjective", "a2c_objective"]
