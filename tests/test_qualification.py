from __future__ import annotations

import pytest

from rl_core.experiments.artifacts import ResultArtifact, RunManifest, artifact_digest, utc_now
from rl_core.experiments.qualification import qualify_result


def _manifest(track: str = "paper_reproduction", preset: str = "qualifying") -> RunManifest:
    return RunManifest(
        run_id="run",
        reproduction_key="dqn-2015",
        protocol_revision="1",
        track=track,
        preset=preset,
        seed=1,
        preset_counters={"environment_frames": 100, "evaluation_episodes": 10},
        config_digest="config",
        configuration={"environment": "Pong"},
        created_at=utc_now(),
        provenance={},
    )


def _result(manifest: RunManifest, **overrides: object) -> ResultArtifact:
    data: dict[str, object] = {
        "schema_version": 1,
        "run_id": manifest.run_id,
        "manifest_digest": artifact_digest(manifest.to_dict()),
        "track": manifest.track,
        "preset": manifest.preset,
        "status": "completed",
        "actual_counters": {"environment_frames": 100, "evaluation_episodes": 10},
        "metrics": {},
        "created_at": utc_now(),
        "provenance": {},
    }
    data.update(overrides)
    return ResultArtifact.from_dict(data)


def test_qualifying_result_meeting_preset_is_qualified() -> None:
    manifest = _manifest()
    decision = qualify_result(manifest, _result(manifest))
    assert decision.qualified
    assert decision.reasons == ()


def test_qualification_checks_track_status_linkage_and_counters() -> None:
    manifest = _manifest()
    result = _result(
        manifest,
        track="benchmark",
        preset="smoke",
        status="interrupted",
        manifest_digest="wrong",
        actual_counters={"environment_frames": 99},
    )
    decision = qualify_result(manifest, result)
    assert not decision.qualified
    assert "result track is not paper_reproduction" in decision.reasons
    assert "result preset is not qualifying" in decision.reasons
    assert "result manifest_digest does not match manifest" in decision.reasons
    assert "result status is interrupted, not completed" in decision.reasons
    assert "counter environment_frames is 99, below required 100" in decision.reasons
    assert "missing actual counter: evaluation_episodes" in decision.reasons


def test_smoke_preset_can_never_qualify() -> None:
    manifest = _manifest(preset="smoke")
    decision = qualify_result(manifest, _result(manifest))
    assert not decision.qualified
    assert decision.reasons == ("manifest preset is not qualifying", "result preset is not qualifying")


def test_extra_requirement_raises_the_counter_minimum() -> None:
    manifest = _manifest()
    decision = qualify_result(manifest, _result(manifest), {"evaluation_episodes": 11})
    assert not decision.qualified
    assert decision.reasons == ("counter evaluation_episodes is 10, below required 11",)


def test_invalid_qualification_counter_is_rejected() -> None:
    manifest = _manifest()
    with pytest.raises(ValueError, match="non-negative"):
        qualify_result(manifest, _result(manifest), {"evaluation_episodes": -1})
