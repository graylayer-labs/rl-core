from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl_core.experiments.artifacts import (
    ArtifactValidationError,
    ResultArtifact,
    RunManifest,
    artifact_digest,
    utc_now,
    write_immutable_json,
)


def _manifest() -> RunManifest:
    return RunManifest(
        run_id="pong__seed7",
        reproduction_key="dqn-2015",
        protocol_revision="1",
        track="paper_reproduction",
        preset="qualifying",
        seed=7,
        preset_counters={"environment_frames": 50_000_000, "evaluation_episodes": 100},
        config_digest="a" * 64,
        configuration={"environment": "Pong"},
        created_at=utc_now(),
        provenance={"git": {"commit": "abc"}},
    )


def _result(manifest: RunManifest) -> ResultArtifact:
    return ResultArtifact(
        run_id=manifest.run_id,
        manifest_digest=artifact_digest(manifest.to_dict()),
        track="paper_reproduction",
        preset="qualifying",
        status="completed",
        actual_counters={"environment_frames": 50_000_000, "evaluation_episodes": 100},
        metrics={"human_normalized_score": 1.2},
        created_at=utc_now(),
        provenance={},
    )


def test_manifest_round_trip_is_strict() -> None:
    manifest = _manifest()
    assert RunManifest.from_dict(manifest.to_dict()) == manifest
    data = manifest.to_dict()
    data["unexpected"] = 1
    with pytest.raises(ArtifactValidationError, match="unknown"):
        RunManifest.from_dict(data)


def test_result_rejects_non_finite_metrics() -> None:
    manifest = _manifest()
    data = _result(manifest).to_dict()
    data["metrics"] = {"score": float("nan")}
    with pytest.raises(ArtifactValidationError, match="finite"):
        ResultArtifact.from_dict(data)


def test_manifest_write_is_atomic_and_idempotent(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "manifest.json"
    manifest.write(path)
    manifest.write(path)
    assert json.loads(path.read_text()) == manifest.to_dict()
    assert not list(tmp_path.glob(".*.tmp"))


def test_immutable_write_rejects_different_content(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    write_immutable_json(path, {"answer": 42})
    with pytest.raises(FileExistsError, match="Refusing"):
        write_immutable_json(path, {"answer": 43})


def test_digest_does_not_depend_on_mapping_order() -> None:
    assert artifact_digest({"a": 1, "b": 2}) == artifact_digest({"b": 2, "a": 1})
