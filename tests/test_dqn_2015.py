from __future__ import annotations

import pytest

from rl_core.reproductions.dqn_2015.config import exploration_epsilon, load_dqn_run_config
from rl_core.reproductions.dqn_2015.evaluation import EvaluationSummary


def test_smoke_config_is_small_and_permanently_separate() -> None:
    smoke = load_dqn_run_config("ALE/Pong-v5", 7, "smoke")
    pilot = load_dqn_run_config("ALE/Pong-v5", 7, "pilot")
    qualifying = load_dqn_run_config("ALE/Pong-v5", 7, "qualifying")

    assert smoke.agent_steps == 512
    assert smoke.evaluation_episodes == 2
    assert pilot.agent_steps == 1_000_000
    assert pilot.raw_ale_frames == 4_000_000
    assert pilot.evaluation_episodes == 10
    assert qualifying.agent_steps == 50_000_000
    assert qualifying.raw_ale_frames == 200_000_000
    assert qualifying.evaluation_episodes == 30
    assert qualifying.evaluation_max_agent_steps == 4_500


def test_run_config_rejects_undeclared_seed_and_environment() -> None:
    with pytest.raises(ValueError, match="seed"):
        load_dqn_run_config("ALE/Pong-v5", 8, "smoke")
    with pytest.raises(ValueError, match="environment"):
        load_dqn_run_config("ALE/Alien-v5", 7, "smoke")


def test_exploration_schedule_matches_paper_points() -> None:
    assert exploration_epsilon(0) == pytest.approx(1.0)
    assert exploration_epsilon(500_000) == pytest.approx(0.55)
    assert exploration_epsilon(1_000_000) == pytest.approx(0.1)
    assert exploration_epsilon(2_000_000) == pytest.approx(0.1)


def test_evaluation_summary_preserves_raw_evidence() -> None:
    summary = EvaluationSummary(returns=(1.0, 3.0), lengths=(10, 20))
    assert summary.mean_return == pytest.approx(2.0)
    assert summary.return_std == pytest.approx(1.0)
