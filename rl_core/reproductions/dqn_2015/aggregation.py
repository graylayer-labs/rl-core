"""Cross-seed aggregation for the frozen five-game DQN claim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from rl_core.reproductions.spec import ReproductionSpec


def human_normalized_score(score: float, random_score: float, human_score: float) -> float:
    """Normalize a raw game score to random=0 and human=100."""
    if human_score == random_score:
        raise ValueError("human and random reference scores must differ")
    return 100.0 * (score - random_score) / (human_score - random_score)


@dataclass(frozen=True)
class GameAggregate:
    """Five-seed result for one declared game."""

    environment_id: str
    seed_mean_score: float
    human_normalized_score: float
    paper_human_normalized_score: float
    passed: bool


@dataclass(frozen=True)
class ReproductionDecision:
    """Majority-of-five outcome for protocol revision 1."""

    games: tuple[GameAggregate, ...]
    passed_games: int
    reproduced: bool


def aggregate_reproduction(
    spec: ReproductionSpec,
    seed_scores: Mapping[str, Sequence[float]],
) -> ReproductionDecision:
    """Aggregate seed-level episode means without pooling episodes."""
    aggregates: list[GameAggregate] = []
    for game in spec.protocol.games:
        scores = seed_scores.get(game.id)
        if scores is None:
            raise ValueError(f"missing scores for {game.id}")
        if len(scores) != len(spec.seeds):
            raise ValueError(f"{game.id} requires {len(spec.seeds)} seed scores, got {len(scores)}")
        mean_score = float(np.mean(scores))
        normalized = human_normalized_score(mean_score, game.random_score, game.human_score)
        paper_normalized = human_normalized_score(game.paper_score, game.random_score, game.human_score)
        aggregates.append(
            GameAggregate(
                environment_id=game.id,
                seed_mean_score=mean_score,
                human_normalized_score=normalized,
                paper_human_normalized_score=paper_normalized,
                passed=normalized >= paper_normalized,
            )
        )
    passed_games = sum(game.passed for game in aggregates)
    required_majority = len(aggregates) // 2 + 1
    return ReproductionDecision(tuple(aggregates), passed_games, passed_games >= required_majority)
