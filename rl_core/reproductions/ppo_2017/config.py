"""Configuration loading for the frozen PPO 2017 protocol."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rl_core.algorithms.ppo import PPOConfig
from rl_core.reproductions import load_reproduction_spec

PAPER_DIR = Path("papers/ppo-2017")


@dataclass(frozen=True)
class PPORunConfig:
    """Resolved settings for one PPO paper run."""

    environment_id: str
    seed: int
    preset: str
    protocol_revision: int
    environment_steps: int
    raw_ale_frames: int
    num_envs: int
    rollout_steps: int
    evaluation_episodes: int
    evaluation_max_agent_steps: int
    source: dict[str, Any]
    source_digest: str
    ppo: PPOConfig


def load_ppo_run_config(
    environment_id: str,
    seed: int,
    preset: str,
    paper_dir: Path = PAPER_DIR,
) -> PPORunConfig:
    """Load one frozen game configuration and a declared run budget."""
    if preset not in {"smoke", "pilot", "qualifying"}:
        raise ValueError("preset must be 'smoke', 'pilot', or 'qualifying'")
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
    if preset == "qualifying":
        num_envs = int(training["num_envs"])
        rollout_steps = int(training["rollout_steps"])
        environment_steps = int(raw["training_environment_steps"])
        evaluation_episodes = int(evaluation["episodes"])
    elif preset == "pilot":
        num_envs = int(training["num_envs"])
        rollout_steps = int(training["rollout_steps"])
        environment_steps = 102_400
        evaluation_episodes = 10
    else:
        num_envs = 2
        rollout_steps = 8
        environment_steps = num_envs * rollout_steps * 2
        evaluation_episodes = 2
    rollout_size = num_envs * rollout_steps
    if environment_steps % rollout_size:
        raise ValueError("environment_steps must be divisible by num_envs * rollout_steps")
    ppo = PPOConfig(
        learning_rate=float(training["learning_rate"]),
        gamma=float(training["gamma"]),
        gae_lambda=float(training["gae_lambda"]),
        clip_range=float(training["clip_range"]),
        value_coefficient=float(training["value_coefficient"]),
        entropy_coefficient=float(training["entropy_coefficient"]),
        max_gradient_norm=float(training["max_gradient_norm"]),
        epochs=int(training["epochs"]),
        minibatches=int(training["minibatches"]),
    )
    return PPORunConfig(
        environment_id=environment_id,
        seed=seed,
        preset=preset,
        protocol_revision=spec.protocol.version,
        environment_steps=environment_steps,
        raw_ale_frames=environment_steps * spec.protocol.frame_skip,
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        evaluation_episodes=evaluation_episodes,
        evaluation_max_agent_steps=int(evaluation["max_agent_steps"]),
        source=raw,
        source_digest=hashlib.sha256(raw_text.encode()).hexdigest(),
        ppo=ppo,
    )
