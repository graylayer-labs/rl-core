"""Seed aggregation for future RND Montezuma evaluations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RNDAggregate:
    """Raw per-seed evaluation summary; no reproduction decision is implied."""

    seed_mean_return: float
    seed_standard_deviation: float
    seeds: int


def aggregate_rnd_returns(seed_returns: Sequence[float], required_seeds: int = 5) -> RNDAggregate:
    """Aggregate one final evaluation mean from each declared seed."""
    if len(seed_returns) != required_seeds:
        raise ValueError(f"requires {required_seeds} seed returns, got {len(seed_returns)}")
    values = np.asarray(seed_returns, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("seed returns must be finite")
    return RNDAggregate(float(values.mean()), float(values.std()), len(values))
