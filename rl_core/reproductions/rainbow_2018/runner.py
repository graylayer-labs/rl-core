"""End-to-end smoke, pilot, and qualifying runner for Rainbow 2018."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from rl_core.algorithms.rainbow import Rainbow, RainbowConfig
from rl_core.buffers.prioritized_atari_replay_buffer import PrioritizedAtariReplayBuffer
from rl_core.environments import AtariPreprocessingConfig, make_atari_environment
from rl_core.experiments import (
    ResultArtifact,
    RunManifest,
    artifact_digest,
    collect_provenance,
    qualify_result,
    utc_now,
)
from rl_core.reproductions.rainbow_2018.config import RainbowRunConfig, load_rainbow_run_config, priority_beta
from rl_core.reproductions.rainbow_2018.evaluation import evaluate
from rl_core.reproductions.rainbow_2018.preflight import preflight_rainbow_2018
from rl_core.utils.seeding import seed_everything


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_id(config: RainbowRunConfig, commit: str | None) -> str:
    payload = json.dumps({"config": asdict(config), "commit": commit}, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    name = config.environment_id.split("/")[-1].lower()
    return f"rainbow-2018__{config.preset}__{name}__seed{config.seed}__{digest}"


def run_rainbow_2018(environment_id: str, seed: int, preset: str, runs_dir: Path = Path("runs")) -> Path:
    """Run one declared Rainbow seed/game and emit immutable evidence artifacts."""
    preflight = preflight_rainbow_2018()
    config = load_rainbow_run_config(environment_id, seed, preset)
    provenance = collect_provenance(
        repo_dir=Path.cwd(), packages=("ale-py", "gymnasium", "opencv-python-headless", "torch", "numpy")
    )
    if preset == "qualifying" and provenance["git"]["dirty"]:
        raise RuntimeError("qualifying runs require a clean Git working tree")
    run_id = _run_id(config, provenance["git"]["commit"])
    run_dir = runs_dir / "rainbow-2018" / run_id
    environment_preflight = next(item for item in preflight["environments"] if item["env_id"] == environment_id)
    selected_device = _device()
    manifest = RunManifest(
        run_id=run_id,
        reproduction_key="rainbow-2018",
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
    components = config.component_switches
    agent = Rainbow(
        RainbowConfig(
            action_dim=int(environment.action_space.n),
            gamma=config.gamma,
            target_update_interval=config.target_update_optimizer_steps,
            double_q=components["double_q"],
            dueling=components["dueling"],
            prioritized_replay=components["prioritized_replay"],
            n_step=config.n_step if components["n_step"] else 1,
            distributional=components["distributional"],
            noisy=components["noisy"],
        ),
        selected_device,
    )
    replay = PrioritizedAtariReplayBuffer(
        config.replay_capacity,
        n_step=agent.config.n_step,
        gamma=config.gamma,
        priority_alpha=config.priority_alpha if components["prioritized_replay"] else 0.0,
        seed=seed,
    )
    started = monotonic()
    observation, _ = environment.reset(seed=seed)
    replay.begin_episode(np.asarray(observation)[-1])
    try:
        for agent_step in range(1, config.agent_steps + 1):
            action = agent.select_action(np.asarray(observation), 0.0 if components["noisy"] else 0.01, rng)
            next_observation, reward, terminated, truncated, _ = environment.step(action)
            done = bool(terminated or truncated)
            replay.append(action, float(reward), np.asarray(next_observation)[-1], done)
            ready_for_update = replay.ready(max(config.learning_starts, config.batch_size))
            if ready_for_update and agent_step % config.train_frequency == 0:
                batch = replay.sample(config.batch_size, agent.device, priority_beta(agent_step, config))
                metrics = agent.train_step(batch)
                if components["prioritized_replay"]:
                    priority_errors = metrics["priority_errors"]
                    if not isinstance(priority_errors, torch.Tensor):
                        raise TypeError("Rainbow trainer must return tensor priority errors")
                    replay.update_priorities(batch["indices"], priority_errors)
            observation = next_observation
            if done:
                observation, _ = environment.reset()
                replay.begin_episode(np.asarray(observation)[-1])
    finally:
        environment.close()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
    decision = qualify_result(manifest, result)
    (run_dir / "qualification.json").write_text(
        json.dumps({"qualified": decision.qualified, "reasons": decision.reasons}, indent=2) + "\n"
    )
    return run_dir
