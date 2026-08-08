"""End-to-end smoke, pilot, and qualifying runner for PPO 2017."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import numpy as np
import torch

from rl_core.algorithms.ppo import AtariActorCritic, PPOTrainer, RolloutStorage
from rl_core.environments import AtariPreprocessingConfig, SynchronousVectorEnv, make_atari_environment
from rl_core.experiments import (
    ResultArtifact,
    RunManifest,
    artifact_digest,
    collect_provenance,
    qualify_result,
    utc_now,
)
from rl_core.reproductions.ppo_2017.config import PPORunConfig, load_ppo_run_config
from rl_core.reproductions.ppo_2017.preflight import preflight_ppo_2017
from rl_core.utils.seeding import seed_everything


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_id(config: PPORunConfig, commit: str | None) -> str:
    payload = json.dumps({"config": asdict(config), "commit": commit}, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    game = config.environment_id.split("/")[-1].lower()
    return f"ppo-2017__{config.preset}__{game}__seed{config.seed}__{digest}"


def _make_vector_environment(config: PPORunConfig) -> SynchronousVectorEnv:
    def factory(index: int) -> object:
        return make_atari_environment(
            AtariPreprocessingConfig(
                env_id=config.environment_id,
                seed=config.seed + index,
                terminal_on_life_loss=True,
                clip_rewards=True,
            )
        )

    return SynchronousVectorEnv([lambda index=index: factory(index) for index in range(config.num_envs)])


@torch.no_grad()
def _evaluate(
    model: AtariActorCritic,
    environment_id: str,
    seed: int,
    episodes: int,
    max_agent_steps: int,
    device: torch.device,
) -> list[float]:
    """Evaluate greedily on raw (unclipped) rewards using a disjoint seed."""
    environment = make_atari_environment(
        AtariPreprocessingConfig(env_id=environment_id, seed=seed, terminal_on_life_loss=False, clip_rewards=False)
    )
    returns: list[float] = []
    try:
        for episode in range(episodes):
            observation, _ = environment.reset(seed=seed + episode)
            episode_return = 0.0
            for _ in range(max_agent_steps):
                tensor = torch.as_tensor(np.asarray(observation)[None], device=device)
                distribution, _ = model(tensor)
                action = int(distribution.probs.argmax(dim=-1).item())
                observation, reward, terminated, truncated, _ = environment.step(action)
                episode_return += float(reward)
                if terminated or truncated:
                    break
            returns.append(episode_return)
    finally:
        environment.close()
    return returns


def run_ppo_2017(
    environment_id: str,
    seed: int,
    preset: str,
    runs_dir: Path = Path("runs"),
) -> Path:
    """Run one declared PPO seed/game and emit immutable evidence artifacts."""
    preflight = preflight_ppo_2017()
    config = load_ppo_run_config(environment_id, seed, preset)
    provenance = collect_provenance(
        repo_dir=Path.cwd(),
        packages=("ale-py", "gymnasium", "opencv-python-headless", "torch", "numpy"),
    )
    if preset == "qualifying" and provenance["git"]["dirty"]:
        raise RuntimeError("qualifying runs require a clean Git working tree")
    selected_device = _device()
    run_id = _run_id(config, provenance["git"]["commit"])
    run_dir = runs_dir / "ppo-2017" / run_id
    environment_preflight = next(item for item in preflight["environments"] if item["env_id"] == environment_id)
    manifest = RunManifest(
        run_id=run_id,
        reproduction_key="ppo-2017",
        protocol_revision=str(config.protocol_revision),
        track="paper_reproduction",
        preset=preset,
        seed=seed,
        preset_counters={
            "environment_steps": config.environment_steps,
            "raw_ale_frames": config.raw_ale_frames,
            "evaluation_episodes": config.evaluation_episodes,
        },
        config_digest=config.source_digest,
        configuration={
            "resolved_run": asdict(config),
            "environment_preflight": environment_preflight,
            "device": str(selected_device),
            "rollout_collection": "synchronous_vectorized",
            "checkpoint_selection": "final",
            "qualifying_resume_allowed": False,
        },
        created_at=utc_now(),
        provenance=provenance,
    )
    manifest.write(run_dir / "manifest.json")

    seed_everything(seed)
    environments = _make_vector_environment(config)
    started = monotonic()
    try:
        observations, _ = environments.reset(seed=seed)
        model = AtariActorCritic(int(environments.single_action_space.n)).to(selected_device)
        trainer = PPOTrainer(model, config.ppo, selected_device)
        completed_returns: list[float] = []
        running_returns = np.zeros(config.num_envs, dtype=np.float64)
        updates = config.environment_steps // (config.num_envs * config.rollout_steps)
        last_metrics = None
        for _ in range(updates):
            storage = RolloutStorage(
                config.rollout_steps,
                config.num_envs,
                tuple(int(size) for size in observations.shape[1:]),
                selected_device,
            )
            for _ in range(config.rollout_steps):
                observation_tensor = torch.as_tensor(observations, device=selected_device)
                actions, log_probabilities, values = model.act(observation_tensor)
                next_observations, rewards, terminated, truncated, _ = environments.step(actions.cpu().numpy())
                done = np.logical_or(terminated, truncated)
                storage.add(
                    observation_tensor,
                    actions,
                    log_probabilities,
                    torch.as_tensor(rewards, device=selected_device),
                    torch.as_tensor(done, dtype=torch.float32, device=selected_device),
                    values,
                )
                running_returns += rewards
                for index in np.flatnonzero(done):
                    completed_returns.append(float(running_returns[index]))
                    running_returns[index] = 0.0
                observations = next_observations
            with torch.no_grad():
                _, bootstrap_value = model(torch.as_tensor(observations, device=selected_device))
            last_metrics = trainer.update(storage.batch(bootstrap_value, config.ppo.gamma, config.ppo.gae_lambda))
    finally:
        environments.close()

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": trainer.optimizer.state_dict()}, checkpoint_dir / "final.pt")
    evaluation_returns = _evaluate(
        model,
        environment_id,
        seed + 1_000_000,
        config.evaluation_episodes,
        config.evaluation_max_agent_steps,
        selected_device,
    )
    evaluation_dir = run_dir / "evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "final.json").write_text(json.dumps({"returns": evaluation_returns}, indent=2) + "\n")
    mean_training_return = float(np.mean(completed_returns)) if completed_returns else 0.0
    result = ResultArtifact(
        run_id=run_id,
        manifest_digest=artifact_digest(manifest.to_dict()),
        track="paper_reproduction",
        preset=preset,
        status="completed",
        actual_counters={
            "environment_steps": config.environment_steps,
            "raw_ale_frames": config.raw_ale_frames,
            "evaluation_episodes": config.evaluation_episodes,
            "optimizer_updates": trainer.optimizer_steps,
        },
        metrics={
            "mean_episode_return": float(np.mean(evaluation_returns)),
            "return_standard_deviation": float(np.std(evaluation_returns)),
            "mean_training_episode_return": mean_training_return,
            "policy_loss": 0.0 if last_metrics is None else last_metrics.policy_loss,
            "value_loss": 0.0 if last_metrics is None else last_metrics.value_loss,
            "approximate_kl": 0.0 if last_metrics is None else last_metrics.approximate_kl,
            "clip_fraction": 0.0 if last_metrics is None else last_metrics.clip_fraction,
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
