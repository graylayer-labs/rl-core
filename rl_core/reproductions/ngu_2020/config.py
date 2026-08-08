"""Configuration loading for the local NGU 2020 readiness protocol."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rl_core.reproductions import load_reproduction_spec

PAPER_DIR = Path("papers/ngu-2020")


@dataclass(frozen=True)
class NGURunConfig:
    """Resolved local-maze protocol and declared lifelong variant."""

    environment_id: str
    seed: int
    preset: str
    lifelong_variant: str
    protocol_revision: int
    episodes: int
    max_steps: int
    source: dict[str, Any]
    source_digest: str


def load_ngu_run_config(
    environment_id: str,
    seed: int,
    preset: str,
    lifelong_variant: str = "rnd",
    paper_dir: Path = PAPER_DIR,
) -> NGURunConfig:
    """Resolve one preregistered maze run and lifelong-novelty variant."""
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
    variants = raw["training"]["lifelong_variants"]
    if lifelong_variant not in variants:
        raise ValueError(f"lifelong_variant must be one of {variants}")
    episodes = {"smoke": 1, "pilot": 10, "qualifying": 100}[preset]
    return NGURunConfig(
        environment_id=environment_id,
        seed=seed,
        preset=preset,
        lifelong_variant=lifelong_variant,
        protocol_revision=spec.protocol.version,
        episodes=episodes,
        max_steps=int(raw["environment"]["max_steps"]),
        source=raw,
        source_digest=hashlib.sha256(raw_text.encode()).hexdigest(),
    )
