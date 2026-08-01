"""Load and validate paper-reproduction specifications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

VALID_STATUSES = frozenset({"planned", "running", "partial", "reproduced", "not_reproduced"})
VALID_IMPLEMENTATION_KINDS = frozenset({"external", "local", "hybrid"})


class SpecValidationError(ValueError):
    """Raised when a reproduction specification is incomplete or inconsistent."""


@dataclass(frozen=True)
class Paper:
    """Bibliographic details for the paper under test."""

    title: str
    authors: tuple[str, ...]
    year: int
    url: str


@dataclass(frozen=True)
class Budget:
    """Fixed training budget for qualifying reproduction runs."""

    unit: str
    value: int


@dataclass(frozen=True)
class ReproductionSpec:
    """Validated, machine-readable protocol for one paper claim."""

    schema_version: int
    key: str
    paper: Paper
    algorithm: str
    implementation: str
    implementation_kind: str
    claim: str
    original_testbed: str
    status: str
    seeds: tuple[int, ...]
    budget: Budget
    primary_metric: str
    success_criterion: str
    deviations: tuple[str, ...]


def load_reproduction_spec(path: Path | str) -> ReproductionSpec:
    """Load and validate a reproduction specification from YAML."""
    spec_path = Path(path)
    raw = yaml.safe_load(spec_path.read_text())
    if not isinstance(raw, dict):
        raise SpecValidationError(f"{spec_path}: expected a YAML mapping")

    try:
        paper_raw = _mapping(raw, "paper")
        budget_raw = _mapping(raw, "budget")
        status = _string(raw, "status")
        implementation_kind = _string(raw, "implementation_kind")
        seeds = _integer_list(raw, "seeds")
        deviations = _string_list(raw, "deviations")

        if status not in VALID_STATUSES:
            allowed = ", ".join(sorted(VALID_STATUSES))
            raise SpecValidationError(f"{spec_path}: status must be one of: {allowed}")
        if implementation_kind not in VALID_IMPLEMENTATION_KINDS:
            allowed = ", ".join(sorted(VALID_IMPLEMENTATION_KINDS))
            raise SpecValidationError(f"{spec_path}: implementation_kind must be one of: {allowed}")
        if len(set(seeds)) != len(seeds):
            raise SpecValidationError(f"{spec_path}: seeds must be unique")

        return ReproductionSpec(
            schema_version=_integer(raw, "schema_version"),
            key=_string(raw, "key"),
            paper=Paper(
                title=_string(paper_raw, "title"),
                authors=_string_list(paper_raw, "authors"),
                year=_integer(paper_raw, "year"),
                url=_string(paper_raw, "url"),
            ),
            algorithm=_string(raw, "algorithm"),
            implementation=_string(raw, "implementation"),
            implementation_kind=implementation_kind,
            claim=_string(raw, "claim"),
            original_testbed=_string(raw, "original_testbed"),
            status=status,
            seeds=seeds,
            budget=Budget(
                unit=_string(budget_raw, "unit"),
                value=_positive_integer(budget_raw, "value"),
            ),
            primary_metric=_string(raw, "primary_metric"),
            success_criterion=_string(raw, "success_criterion"),
            deviations=deviations,
        )
    except SpecValidationError:
        raise
    except (KeyError, TypeError) as exc:
        raise SpecValidationError(f"{spec_path}: invalid specification: {exc}") from exc


def discover_reproduction_specs(papers_dir: Path | str = "papers") -> list[Path]:
    """Return reproduction specifications in stable paper-key order."""
    return sorted(Path(papers_dir).glob("*/reproduction.yaml"))


def _mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise SpecValidationError(f"{key!r} must be a mapping")
    return value


def _string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SpecValidationError(f"{key!r} must be a non-empty string")
    return value.strip()


def _integer(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise SpecValidationError(f"{key!r} must be an integer")
    return value


def _positive_integer(data: dict[str, Any], key: str) -> int:
    value = _integer(data, key)
    if value <= 0:
        raise SpecValidationError(f"{key!r} must be positive")
    return value


def _string_list(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise SpecValidationError(f"{key!r} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise SpecValidationError(f"{key!r} must contain non-empty strings")
    return tuple(item.strip() for item in value)


def _integer_list(data: dict[str, Any], key: str) -> tuple[int, ...]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise SpecValidationError(f"{key!r} must be a non-empty list")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise SpecValidationError(f"{key!r} must contain integers")
    return tuple(value)
