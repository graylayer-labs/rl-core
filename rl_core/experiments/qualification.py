"""Deterministic checks deciding whether a run can support a reproduction claim."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from rl_core.experiments.artifacts import ResultArtifact, RunManifest, artifact_digest


@dataclass(frozen=True)
class QualificationDecision:
    """Result of checking a run's declared and actual protocol requirements."""

    qualified: bool
    reasons: tuple[str, ...]


def qualify_result(
    manifest: RunManifest,
    result: ResultArtifact,
    required_counters: Mapping[str, int] | None = None,
) -> QualificationDecision:
    """Check immutable linkage, qualifying preset, completion, and counters.

    ``required_counters`` adds (or raises) minimums over the manifest preset;
    it is useful for a paper-level qualifying gate across several runs.
    """
    reasons: list[str] = []
    if manifest.track != "paper_reproduction":
        reasons.append("manifest track is not paper_reproduction")
    if result.track != "paper_reproduction":
        reasons.append("result track is not paper_reproduction")
    if result.track != manifest.track:
        reasons.append("result track does not match manifest")
    if manifest.preset != "qualifying":
        reasons.append("manifest preset is not qualifying")
    if result.preset != "qualifying":
        reasons.append("result preset is not qualifying")
    if result.preset != manifest.preset:
        reasons.append("result preset does not match manifest")
    if result.run_id != manifest.run_id:
        reasons.append("result run_id does not match manifest")
    if result.manifest_digest != artifact_digest(manifest.to_dict()):
        reasons.append("result manifest_digest does not match manifest")
    if result.status != "completed":
        reasons.append(f"result status is {result.status}, not completed")

    required = dict(manifest.preset_counters)
    for name, minimum in (required_counters or {}).items():
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 0:
            raise ValueError(f"required counter {name!r} must be a non-negative integer")
        required[name] = max(required.get(name, 0), minimum)
    for name, minimum in sorted(required.items()):
        actual = result.actual_counters.get(name)
        if actual is None:
            reasons.append(f"missing actual counter: {name}")
        elif actual < minimum:
            reasons.append(f"counter {name} is {actual}, below required {minimum}")
    return QualificationDecision(qualified=not reasons, reasons=tuple(reasons))
