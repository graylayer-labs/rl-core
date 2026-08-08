"""Configuration loading for the frozen Rainbow 2018 protocol."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rl_core.reproductions import load_reproduction_spec

PAPER_DIR = Path("papers/rainbow-2018")


@dataclass(frozen=True)
class RainbowRunConfig:
    """Resolved settings and component switches for one Rainbow run."""

    environment_id: str
    seed: int
    preset: str
    protocol_revision: int
    agent_steps: int
    raw_ale_frames: int
    replay_capacity: int
    learning_starts: int
    train_frequency: int
    batch_size: int
    target_update_optimizer_steps: int
    evaluation_episodes: int
    evaluation_max_agent_steps: int
    evaluation_epsilon: float
    gamma: float
    n_step: int
    priority_alpha: float
    priority_beta_initial: float
    priority_beta_final: float
    priority_beta_steps: int
    component_switches: dict[str, bool]
    source: dict[str, Any]
    source_digest: str


def load_rainbow_run_config(
    environment_id: str, seed: int, preset: str, paper_dir: Path = PAPER_DIR
) -> RainbowRunConfig:
    """Load a declared game config while preserving qualifying settings in YAML."""
    if preset not in {"smoke", "pilot", "qualifying"}:
        raise ValueError("preset must be 'smoke', 'pilot', or 'qualifying'")
    spec = load_reproduction_spec(paper_dir / "reproduction.yaml")
    if seed not in spec.seeds:
        raise ValueError(f"seed {seed} is not declared by the protocol")
    game = next((candidate for candidate in spec.protocol.games if candidate.id == environment_id), None)
    if game is None:
        raise ValueError(f"environment {environment_id!r} is not declared by the protocol")
    raw_text = (paper_dir / game.config).read_text()
    raw = yaml.safe_load(raw_text)
    training = raw["training"]
    evaluation = raw["evaluation"]
    if preset == "qualifying":
        agent_steps = int(raw["training_agent_steps"])
        replay_capacity = int(training["replay_buffer_size"])
        learning_starts = int(training["learning_starts_agent_steps"])
        batch_size = int(training["batch_size"])
        evaluation_episodes = 30
        evaluation_max_agent_steps = int(evaluation["max_raw_ale_frames"]) // spec.protocol.frame_skip
    elif preset == "pilot":
        agent_steps = 1_000_000
        replay_capacity = 250_000
        learning_starts = int(training["learning_starts_agent_steps"])
        batch_size = int(training["batch_size"])
        evaluation_episodes = 10
        evaluation_max_agent_steps = int(evaluation["max_raw_ale_frames"]) // spec.protocol.frame_skip
    else:
        agent_steps = 512
        replay_capacity = 1_024
        learning_starts = 64
        batch_size = 16
        evaluation_episodes = 2
        evaluation_max_agent_steps = 100
    return RainbowRunConfig(
        environment_id=environment_id,
        seed=seed,
        preset=preset,
        protocol_revision=spec.protocol.version,
        agent_steps=agent_steps,
        raw_ale_frames=agent_steps * spec.protocol.frame_skip,
        replay_capacity=replay_capacity,
        learning_starts=learning_starts,
        train_frequency=int(training["train_frequency_agent_steps"]),
        batch_size=batch_size,
        target_update_optimizer_steps=int(training["target_network_update_agent_steps"])
        // int(training["train_frequency_agent_steps"]),
        evaluation_episodes=evaluation_episodes,
        evaluation_max_agent_steps=evaluation_max_agent_steps,
        evaluation_epsilon=float(evaluation["epsilon"]),
        gamma=float(training["gamma"]),
        n_step=int(training["n_step"]),
        priority_alpha=float(training["priority_alpha"]),
        priority_beta_initial=float(training["priority_beta_initial"]),
        priority_beta_final=float(training["priority_beta_final"]),
        priority_beta_steps=int(training["priority_beta_steps_agent_steps"]),
        component_switches={key: bool(value) for key, value in raw["components"].items()},
        source=raw,
        source_digest=hashlib.sha256(raw_text.encode()).hexdigest(),
    )


def priority_beta(agent_step: int, config: RainbowRunConfig) -> float:
    """Linearly anneal PER importance sampling from β₀ to β=1."""
    fraction = min(max(agent_step, 0) / config.priority_beta_steps, 1.0)
    return config.priority_beta_initial + fraction * (config.priority_beta_final - config.priority_beta_initial)
