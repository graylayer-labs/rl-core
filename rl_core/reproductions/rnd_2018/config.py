"""Frozen configuration for the RND 2018 readiness track."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rl_core.reproductions import load_reproduction_spec

PAPER_DIR = Path("papers/rnd-2018")


@dataclass(frozen=True)
class RNDRunConfig:
    """Resolved Montezuma protocol plus readiness-run counters."""

    environment_id: str
    seed: int
    preset: str
    protocol_revision: int
    environment_frames: int
    synthetic_observations: int
    intrinsic_discount: float
    source: dict[str, Any]
    source_digest: str


def load_rnd_run_config(environment_id: str, seed: int, preset: str, paper_dir: Path = PAPER_DIR) -> RNDRunConfig:
    """Resolve a declared RND run while retaining the paper-scale budget."""
    if preset not in {"smoke", "pilot", "qualifying"}:
        raise ValueError("preset must be 'smoke', 'pilot', or 'qualifying'")
    spec = load_reproduction_spec(paper_dir / "reproduction.yaml")
    if seed not in spec.seeds:
        raise ValueError(f"seed {seed} is not declared by the protocol")
    game = next((item for item in spec.protocol.games if item.id == environment_id), None)
    if game is None:
        raise ValueError(f"environment {environment_id!r} is not declared by the protocol")
    raw_text = (paper_dir / game.config).read_text()
    raw = yaml.safe_load(raw_text)
    frame_budgets = {"smoke": 256, "pilot": 100_000, "qualifying": spec.budget.value}
    synthetic_budgets = {"smoke": 16, "pilot": 256, "qualifying": 2_048}
    return RNDRunConfig(
        environment_id=environment_id,
        seed=seed,
        preset=preset,
        protocol_revision=spec.protocol.version,
        environment_frames=frame_budgets[preset],
        synthetic_observations=synthetic_budgets[preset],
        intrinsic_discount=float(raw["training"]["intrinsic_discount"]),
        source=raw,
        source_digest=hashlib.sha256(raw_text.encode()).hexdigest(),
    )
