"""End-to-end smoke and qualifying runner for DQN 2015."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from rl_core.buffers import AtariReplayBuffer
from rl_core.environments import AtariPreprocessingConfig, make_atari_environment
from rl_core.experiments import (
    ResultArtifact,
    RunManifest,
    artifact_digest,
    collect_provenance,
    qualify_result,
    utc_now,
)
from rl_core.integrations import NatureDQN, NatureDQNConfig
from rl_core.reproductions.dqn_2015.config import DQNRunConfig, exploration_epsilon, load_dqn_run_config
from rl_core.reproductions.dqn_2015.evaluation import evaluate
from rl_core.reproductions.dqn_2015.preflight import preflight_dqn_2015
from rl_core.utils.seeding import seed_everything


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_id(config: DQNRunConfig, commit: str | None) -> str:
    payload = json.dumps({"config": asdict(config), "commit": commit}, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    name = config.environment_id.split("/")[-1].lower()
    return f"dqn-2015__{config.preset}__{name}__seed{config.seed}__{digest}"


def _write_status(run_dir: Path, status: str, agent_step: int, started: float) -> None:
    """Atomically expose long-run progress without changing immutable evidence."""
    status_path = run_dir / "status.json"
    temporary = status_path.with_suffix(".json.tmp")
    elapsed = monotonic() - started
    temporary.write_text(
        json.dumps(
            {
                "status": status,
                "agent_step": agent_step,
                "elapsed_seconds": elapsed,
                "agent_steps_per_second": agent_step / elapsed if elapsed else 0.0,
            },
            indent=2,
        )
        + "\n"
    )
    temporary.replace(status_path)


def run_dqn_2015(
    environment_id: str,
    seed: int,
    preset: str,
    runs_dir: Path = Path("runs"),
) -> Path:
    """Run one declared seed/game and emit immutable evidence artifacts."""
    preflight = preflight_dqn_2015()
    config = load_dqn_run_config(environment_id, seed, preset)
    provenance = collect_provenance(
        repo_dir=Path.cwd(),
        packages=("ale-py", "gymnasium", "opencv-python-headless", "torch", "numpy"),
    )
    if preset == "qualifying" and provenance["git"]["dirty"]:
        raise RuntimeError("qualifying runs require a clean Git working tree")
    run_id = _run_id(config, provenance["git"]["commit"])
    run_dir = runs_dir / "dqn-2015" / run_id
    environment_preflight = next(item for item in preflight["environments"] if item["env_id"] == environment_id)
    selected_device = _device()
    manifest = RunManifest(
        run_id=run_id,
        reproduction_key="dqn-2015",
        protocol_revision=str(config.protocol_revision),
        track="paper_reproduction",
        preset=preset,
        seed=seed,
        preset_counters={
            "agent_steps": config.agent_steps,
            "raw_ale_frames": config.raw_ale_frames,
            "evaluation_episodes": config.evaluation_episodes,
        },
        config_digest=config.source_digest,
        configuration={
            "resolved_run": asdict(config),
            "environment_preflight": environment_preflight,
            "device": str(selected_device),
            "checkpoint_selection": "final",
            "qualifying_resume_allowed": False,
        },
        created_at=utc_now(),
        provenance=provenance,
    )
    manifest.write(run_dir / "manifest.json")

    seed_everything(seed)
    rng = np.random.default_rng(seed)
    environment = make_atari_environment(
        AtariPreprocessingConfig(env_id=environment_id, seed=seed, terminal_on_life_loss=True, clip_rewards=True)
    )
    agent = NatureDQN(
        NatureDQNConfig(
            action_dim=int(environment.action_space.n),
            target_update_interval=config.target_update_optimizer_steps,
        ),
        selected_device,
    )
    replay = AtariReplayBuffer(config.replay_capacity, seed=seed)
    started = monotonic()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    status_frequency = max(config.agent_steps // 100, 1)
    checkpoint_frequency = {"smoke": 256, "pilot": 100_000, "qualifying": 1_000_000}[preset]
    observation, _ = environment.reset(seed=seed)
    replay.begin_episode(np.asarray(observation)[-1])
    episode_return = 0.0
    _write_status(run_dir, "running", 0, started)
    try:
        for agent_step in range(1, config.agent_steps + 1):
            action = agent.select_action(np.asarray(observation), exploration_epsilon(agent_step - 1), rng)
            next_observation, reward, terminated, truncated, _ = environment.step(action)
            done = bool(terminated or truncated)
            episode_return += float(reward)
            replay.append(action, float(reward), np.asarray(next_observation)[-1], done)
            if replay.ready(config.learning_starts) and agent_step % config.train_frequency == 0:
                agent.train_step(replay.sample(config.batch_size, agent.device))
            observation = next_observation
            if done:
                observation, _ = environment.reset()
                replay.begin_episode(np.asarray(observation)[-1])
                episode_return = 0.0
            if agent_step % status_frequency == 0:
                _write_status(run_dir, "running", agent_step, started)
                elapsed = monotonic() - started
                print(
                    f"progress: {agent_step}/{config.agent_steps} agent steps ({agent_step / elapsed:.1f} steps/s)",
                    flush=True,
                )
            if agent_step % checkpoint_frequency == 0:
                torch.save(agent.state_dicts(), checkpoint_dir / f"step_{agent_step:09d}.pt")
    except KeyboardInterrupt:
        torch.save(agent.state_dicts(), checkpoint_dir / f"interrupted_step_{agent_step:09d}.pt")
        _write_status(run_dir, "interrupted", agent_step, started)
        raise
    finally:
        environment.close()

    torch.save(agent.state_dicts(), checkpoint_dir / "final.pt")
    summary = evaluate(
        agent,
        environment_id,
        seed + 1_000_000,
        config.evaluation_episodes,
        config.evaluation_max_agent_steps,
        config.evaluation_epsilon,
    )
    evaluation_dir = run_dir / "evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "final.json").write_text(
        json.dumps({"returns": summary.returns, "lengths": summary.lengths}, indent=2) + "\n"
    )
    result = ResultArtifact(
        run_id=run_id,
        manifest_digest=artifact_digest(manifest.to_dict()),
        track="paper_reproduction",
        preset=preset,
        status="completed",
        actual_counters={
            "agent_steps": config.agent_steps,
            "raw_ale_frames": config.raw_ale_frames,
            "evaluation_episodes": config.evaluation_episodes,
            "optimizer_updates": agent.optimizer_updates,
        },
        metrics={
            "mean_episode_return": summary.mean_return,
            "return_standard_deviation": summary.return_std,
            "wall_clock_seconds": monotonic() - started,
        },
        created_at=utc_now(),
        provenance={"evaluation_seed": seed + 1_000_000, "resumed": False},
    )
    result.write(run_dir / "result.json")
    _write_status(run_dir, "completed", config.agent_steps, started)
    decision = qualify_result(manifest, result)
    (run_dir / "qualification.json").write_text(
        json.dumps({"qualified": decision.qualified, "reasons": decision.reasons}, indent=2) + "\n"
    )
    return run_dir
