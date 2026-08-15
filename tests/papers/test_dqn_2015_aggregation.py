from __future__ import annotations

import pytest

from rl_core.reproductions import load_reproduction_spec
from rl_core.reproductions.dqn_2015.aggregation import aggregate_reproduction, human_normalized_score


def test_human_normalized_score_reference_points() -> None:
    assert human_normalized_score(1.7, 1.7, 31.8) == pytest.approx(0.0)
    assert human_normalized_score(31.8, 1.7, 31.8) == pytest.approx(100.0)


def test_aggregation_requires_every_declared_seed() -> None:
    spec = load_reproduction_spec("papers/dqn-2015/reproduction.yaml")
    with pytest.raises(ValueError, match="requires 5"):
        aggregate_reproduction(spec, {game.id: [game.paper_score] for game in spec.protocol.games})


def test_paper_scores_recover_the_preregistered_majority() -> None:
    spec = load_reproduction_spec("papers/dqn-2015/reproduction.yaml")
    scores = {game.id: [game.paper_score] * len(spec.seeds) for game in spec.protocol.games}
    decision = aggregate_reproduction(spec, scores)
    assert decision.reproduced
    assert decision.passed_games == 5


def test_aggregation_uses_seed_means_and_majority_rule() -> None:
    spec = load_reproduction_spec("papers/dqn-2015/reproduction.yaml")
    scores = {
        game.id: [game.paper_score if index < 3 else game.random_score] * len(spec.seeds)
        for index, game in enumerate(spec.protocol.games)
    }
    decision = aggregate_reproduction(spec, scores)
    assert decision.reproduced
    assert decision.passed_games == 3
