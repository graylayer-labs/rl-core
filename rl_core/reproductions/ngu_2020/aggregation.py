"""Coverage aggregation for preregistered NGU lifelong-novelty variants."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NGUCoverageAggregate:
    """Coverage result for one declared lifelong-novelty variant."""

    variant: str
    mean_unique_cells: float
    standard_deviation: float
    seeds: int


def aggregate_coverage(
    seed_coverages: Mapping[str, Sequence[float]], required_seeds: int = 5
) -> tuple[NGUCoverageAggregate, ...]:
    """Aggregate random and RND coverage separately, preserving variants."""
    expected = {"random", "rnd"}
    if set(seed_coverages) != expected:
        raise ValueError("coverage must contain exactly random and rnd variants")
    aggregates: list[NGUCoverageAggregate] = []
    for variant in sorted(expected):
        values = np.asarray(seed_coverages[variant], dtype=np.float64)
        if len(values) != required_seeds:
            raise ValueError(f"{variant} requires {required_seeds} seed coverages, got {len(values)}")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError(f"{variant} coverages must be finite and non-negative")
        aggregates.append(NGUCoverageAggregate(variant, float(values.mean()), float(values.std()), len(values)))
    return tuple(aggregates)
