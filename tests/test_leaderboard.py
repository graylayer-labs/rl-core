from __future__ import annotations

from pathlib import Path

import pytest

from rl_core.experiments import ResultArtifact, RunManifest, artifact_digest, utc_now
from rl_core.results import LeaderboardError, leaderboard_rows


def _artifacts(tmp_path: Path, track: str) -> Path:
    manifest = RunManifest(
        run_id="run",
        reproduction_key="ppo-2017",
        protocol_revision="1",
        track=track,
        preset="qualifying",
        seed=7,
        preset_counters={"environment_steps": 10},
        config_digest="config",
        configuration={
            "benchmark": "atari-dense",
            "benchmark_version": "v1",
            "resolved_run": {"environment_id": "ALE/Pong-v5"},
        },
        created_at=utc_now(),
        provenance={"git": {"commit": "abc"}},
    )
    result = ResultArtifact(
        run_id="run",
        manifest_digest=artifact_digest(manifest.to_dict()),
        track=track,
        preset="qualifying",
        status="completed",
        actual_counters={"environment_steps": 10},
        metrics={"mean_episode_return": 3.0, "return_standard_deviation": 1.0, "wall_clock_seconds": 2.0},
        created_at=utc_now(),
        provenance={},
    )
    manifest.write(tmp_path / "manifest.json")
    result.write(tmp_path / "result.json")
    return tmp_path / "result.json"


def test_leaderboard_accepts_completed_benchmark_artifact(tmp_path: Path) -> None:
    rows = leaderboard_rows([_artifacts(tmp_path, "benchmark")])
    assert rows[0]["environment"] == "ALE/Pong-v5"
    assert rows[0]["mean_return"] == pytest.approx(3.0)


def test_leaderboard_rejects_paper_artifact(tmp_path: Path) -> None:
    with pytest.raises(LeaderboardError, match="paper-reproduction"):
        leaderboard_rows([_artifacts(tmp_path, "paper_reproduction")])
