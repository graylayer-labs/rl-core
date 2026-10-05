"""Immutable JSON artifacts produced by experiment runs.

The artifact format is deliberately small and independent of any particular
trainer.  A run writes a :class:`RunManifest` before execution and a
:class:`ResultArtifact` only after it has finished.  Both formats reject
unknown fields so a reader never silently ignores a protocol change.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

ARTIFACT_SCHEMA_VERSION = 1
_TRACKS = frozenset({"benchmark", "paper_reproduction"})
_PRESETS = frozenset({"smoke", "exploration_10", "exploration_50", "pilot", "qualifying"})


class ArtifactValidationError(ValueError):
    """Raised when an artifact does not conform exactly to its schema."""


def _require_exact_keys(data: Mapping[str, Any], keys: set[str], name: str) -> None:
    actual = set(data)
    if actual != keys:
        missing = sorted(keys - actual)
        unknown = sorted(actual - keys)
        raise ArtifactValidationError(f"Invalid {name} keys; missing={missing}, unknown={unknown}")


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"{name} must be a non-empty string")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactValidationError(f"{name} must be a non-negative integer")
    return value


def _timestamp(value: Any, name: str) -> str:
    text = _non_empty_string(value, name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactValidationError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ArtifactValidationError(f"{name} must include a timezone")
    return text


def _json_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{name} must be a JSON object")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"{name} must contain JSON-safe values") from exc
    return value


def _counters(value: Any, name: str) -> dict[str, int]:
    raw = _json_object(value, name)
    if not raw:
        raise ArtifactValidationError(f"{name} must not be empty")
    return {
        _non_empty_string(key, f"{name} key"): _non_negative_int(item, f"{name}.{key}") for key, item in raw.items()
    }


def _metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ArtifactValidationError("metrics must be a JSON object")
    metrics: dict[str, float] = {}
    for key, item in value.items():
        _non_empty_string(key, "metrics key")
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ArtifactValidationError(f"metrics.{key} must be a finite number")
        metrics[key] = float(item)
    return metrics


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Encode JSON deterministically for artifact storage and hashing."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def artifact_digest(value: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of an artifact's canonical JSON."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def write_immutable_json(path: Path | str, value: Mapping[str, Any]) -> Path:
    """Atomically create *path*, refusing to replace a different artifact.

    Repeating the same write is safe and idempotent.  Hard-link creation gives
    an exclusive-create operation without a check-then-replace race.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(value)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            try:
                existing = json.loads(destination.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ArtifactValidationError(f"Existing artifact is unreadable: {destination}") from exc
            if canonical_json_bytes(existing) != content:
                raise FileExistsError(f"Refusing to overwrite immutable artifact: {destination}") from None
        return destination
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class RunManifest:
    """Declared, immutable run intent written before training begins."""

    run_id: str
    reproduction_key: str
    protocol_revision: str
    track: str
    preset: str
    seed: int
    preset_counters: dict[str, int]
    config_digest: str
    configuration: dict[str, Any]
    created_at: str
    provenance: dict[str, Any]
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the manifest at construction time."""
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactValidationError("Unsupported manifest schema_version")
        for name in ("run_id", "reproduction_key", "protocol_revision", "config_digest"):
            _non_empty_string(getattr(self, name), name)
        if self.track not in _TRACKS:
            raise ArtifactValidationError(f"track must be one of {sorted(_TRACKS)}")
        if self.preset not in _PRESETS:
            raise ArtifactValidationError(f"preset must be one of {sorted(_PRESETS)}")
        _non_negative_int(self.seed, "seed")
        _counters(self.preset_counters, "preset_counters")
        _json_object(self.configuration, "configuration")
        _timestamp(self.created_at, "created_at")
        _json_object(self.provenance, "provenance")

    def to_dict(self) -> dict[str, Any]:
        """Return the exact JSON representation of this manifest."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "reproduction_key": self.reproduction_key,
            "protocol_revision": self.protocol_revision,
            "track": self.track,
            "preset": self.preset,
            "seed": self.seed,
            "preset_counters": self.preset_counters,
            "config_digest": self.config_digest,
            "configuration": self.configuration,
            "created_at": self.created_at,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build a manifest after rejecting missing or unknown fields."""
        _require_exact_keys(data, set(cls.__dataclass_fields__), "manifest")
        return cls(**dict(data))

    def write(self, path: Path | str) -> Path:
        """Write this manifest once, atomically and immutably."""
        return write_immutable_json(path, self.to_dict())


@dataclass(frozen=True)
class ResultArtifact:
    """Immutable outcome of one completed or interrupted run."""

    run_id: str
    manifest_digest: str
    track: str
    preset: str
    status: str
    actual_counters: dict[str, int]
    metrics: dict[str, float]
    created_at: str
    provenance: dict[str, Any]
    schema_version: int = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the result at construction time."""
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ArtifactValidationError("Unsupported result schema_version")
        _non_empty_string(self.run_id, "run_id")
        _non_empty_string(self.manifest_digest, "manifest_digest")
        if self.track not in _TRACKS:
            raise ArtifactValidationError(f"track must be one of {sorted(_TRACKS)}")
        if self.preset not in _PRESETS:
            raise ArtifactValidationError(f"preset must be one of {sorted(_PRESETS)}")
        if self.status not in {"completed", "failed", "interrupted"}:
            raise ArtifactValidationError("status must be completed, failed, or interrupted")
        _counters(self.actual_counters, "actual_counters")
        _metrics(self.metrics)
        _timestamp(self.created_at, "created_at")
        _json_object(self.provenance, "provenance")

    def to_dict(self) -> dict[str, Any]:
        """Return the exact JSON representation of this result."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "manifest_digest": self.manifest_digest,
            "track": self.track,
            "preset": self.preset,
            "status": self.status,
            "actual_counters": self.actual_counters,
            "metrics": self.metrics,
            "created_at": self.created_at,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build a result after rejecting missing or unknown fields."""
        _require_exact_keys(data, set(cls.__dataclass_fields__), "result")
        return cls(**dict(data))

    def write(self, path: Path | str) -> Path:
        """Write this result once, atomically and immutably."""
        return write_immutable_json(path, self.to_dict())


def utc_now() -> str:
    """Return a timezone-aware timestamp suitable for newly-created artifacts."""
    return datetime.now(UTC).isoformat()
