"""Generate benchmark leaderboards exclusively from validated artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rl_core.experiments import ResultArtifact, RunManifest, artifact_digest

FIELDS = (
    "benchmark",
    "benchmark_version",
    "environment",
    "algorithm",
    "implementation_commit",
    "seed",
    "mean_return",
    "return_std",
    "environment_steps",
    "wall_clock_seconds",
    "result_path",
)


class LeaderboardError(ValueError):
    """Raised when an artifact cannot safely enter a benchmark leaderboard."""


def _load_pair(result_path: Path) -> tuple[RunManifest, ResultArtifact]:
    manifest_path = result_path.with_name("manifest.json")
    if not manifest_path.is_file():
        raise LeaderboardError(f"missing manifest beside {result_path}")
    manifest = RunManifest.from_dict(json.loads(manifest_path.read_text()))
    result = ResultArtifact.from_dict(json.loads(result_path.read_text()))
    if result.manifest_digest != artifact_digest(manifest.to_dict()):
        raise LeaderboardError(f"manifest digest mismatch for {result_path}")
    if manifest.track != "benchmark" or result.track != "benchmark":
        raise LeaderboardError(f"paper-reproduction artifact cannot enter leaderboard: {result_path}")
    if result.status != "completed":
        raise LeaderboardError(f"incomplete result cannot enter leaderboard: {result_path}")
    return manifest, result


def leaderboard_rows(result_paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    """Build stable per-seed rows from completed benchmark artifacts."""
    rows: list[dict[str, Any]] = []
    for raw_path in result_paths:
        result_path = Path(raw_path)
        manifest, result = _load_pair(result_path)
        configuration = manifest.configuration
        resolved = configuration.get("resolved_run")
        if not isinstance(resolved, dict):
            raise LeaderboardError(f"missing resolved_run configuration: {result_path}")
        benchmark = configuration.get("benchmark")
        benchmark_version = configuration.get("benchmark_version")
        environment = resolved.get("environment_id")
        if not all(isinstance(value, str) and value for value in (benchmark, benchmark_version, environment)):
            raise LeaderboardError(f"missing benchmark identity: {result_path}")
        rows.append(
            {
                "benchmark": benchmark,
                "benchmark_version": benchmark_version,
                "environment": environment,
                "algorithm": manifest.reproduction_key,
                "implementation_commit": manifest.provenance.get("git", {}).get("commit") or "",
                "seed": manifest.seed,
                "mean_return": result.metrics.get("mean_episode_return", ""),
                "return_std": result.metrics.get("return_standard_deviation", ""),
                "environment_steps": result.actual_counters.get("environment_steps", ""),
                "wall_clock_seconds": result.metrics.get("wall_clock_seconds", ""),
                "result_path": result_path.as_posix(),
            }
        )
    return sorted(rows, key=lambda row: (row["benchmark"], row["environment"], row["algorithm"], row["seed"]))


def write_leaderboard(rows: Iterable[dict[str, Any]], path: Path | str) -> Path:
    """Write a deterministic CSV leaderboard."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return destination
