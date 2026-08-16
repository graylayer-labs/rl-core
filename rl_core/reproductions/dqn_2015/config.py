"""Configuration loading for the frozen DQN 2015 protocol."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rl_core.reproductions import load_reproduction_spec

PAPER_DIR = Path("papers/dqn-2015")


@dataclass(frozen=True)
class DQNRunConfig:
    """Resolved settings for one DQN paper run."""

    environment_id: str
    seed: int
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
    source: dict[str, Any]
    source_digest: str


def load_dqn_run_config(
    environment_id: str,
    seed: int,
    agent_steps: int,
    paper_dir: Path = PAPER_DIR,
) -> DQNRunConfig:
    """Load a declared game config with specified agent step budget."""
    spec = load_reproduction_spec(paper_dir / "reproduction.yaml")
    if seed not in spec.seeds:
        raise ValueError(f"seed {seed} is not declared by the protocol")
    game = next((item for item in spec.protocol.games if item.id == environment_id), None)
    if game is None:
        raise ValueError(f"environment {environment_id!r} is not declared by the protocol")
    config_path = paper_dir / game.config
    raw_text = config_path.read_text()
    raw = yaml.safe_load(raw_text)
    training = raw["training"]
    evaluation = raw["evaluation"]

    replay_capacity = int(training["replay_buffer_size"])
    learning_starts = int(training["learning_starts_agent_steps"])
    batch_size = int(training["batch_size"])
    evaluation_episodes = 30
    evaluation_max_agent_steps = int(evaluation["max_raw_ale_frames"]) // spec.protocol.frame_skip

    return DQNRunConfig(
        environment_id=environment_id,
        seed=seed,
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
        source=raw,
        source_digest=hashlib.sha256(raw_text.encode()).hexdigest(),
    )


def exploration_epsilon(agent_step: int) -> float:
    """Return the paper's linear 1.0-to-0.1 exploration schedule."""
    fraction = min(max(agent_step, 0) / 1_000_000, 1.0)
    return 1.0 + fraction * (0.1 - 1.0)
