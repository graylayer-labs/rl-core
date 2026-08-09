"""Local-maze NGU component runner with immutable coverage artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from rl_core.environments.disco_maze import RandomDiscoMaze
from rl_core.experiments import (
    ResultArtifact,
    RunManifest,
    artifact_digest,
    collect_provenance,
    qualify_result,
    utc_now,
)
from rl_core.intrinsic import RNDModel, RNDRewardNormalizer
from rl_core.intrinsic.ngu import EpisodicKNNNovelty, InverseDynamicsEmbedding
from rl_core.reproductions.ngu_2020.config import NGURunConfig, load_ngu_run_config
from rl_core.reproductions.ngu_2020.preflight import preflight_ngu_2020
from rl_core.utils.seeding import seed_everything


def _run_id(config: NGURunConfig, commit: str | None) -> str:
    payload = json.dumps({"config": asdict(config), "commit": commit}, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"ngu-2020__{config.lifelong_variant}__{config.preset}__seed{config.seed}__{digest}"


def run_ngu_2020(
    environment_id: str,
    seed: int,
    preset: str,
    lifelong_variant: str = "rnd",
    runs_dir: Path = Path("runs"),
) -> Path:
    """Run the dependency-free Random Disco Maze protocol.

    Actions are intentionally random in this readiness path. It validates the
    episodic/RND signals and reports coverage, but is not a trained NGU claim.
    """
    preflight = preflight_ngu_2020()
    config = load_ngu_run_config(environment_id, seed, preset, lifelong_variant)
    provenance = collect_provenance(repo_dir=Path.cwd(), packages=("numpy", "torch", "pyyaml"))
    run_id = _run_id(config, provenance["git"]["commit"])
    run_dir = runs_dir / "ngu-2020" / run_id
    total_steps = config.episodes * config.max_steps
    manifest = RunManifest(
        run_id=run_id,
        reproduction_key="ngu-2020",
        protocol_revision=str(config.protocol_revision),
        track="paper_reproduction",
        preset=preset,
        seed=seed,
        preset_counters={"environment_steps": total_steps, "episodes": config.episodes},
        config_digest=config.source_digest,
        configuration={
            "resolved_run": asdict(config),
            "preflight": preflight,
            "implementation_boundary": "random-policy local maze component readiness",
        },
        created_at=utc_now(),
        provenance=provenance,
    )
    manifest.write(run_dir / "manifest.json")
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    shape = (21, 21, 3)
    rnd = RNDModel(shape, feature_dim=16, hidden_sizes=(64,))
    rnd_optimizer = torch.optim.Adam(rnd.predictor.parameters(), lr=1e-3)
    rnd_normalizer = RNDRewardNormalizer()
    inverse = InverseDynamicsEmbedding(shape, action_dim=4)
    inverse_optimizer = torch.optim.Adam(inverse.parameters(), lr=1e-3)
    coverage: list[int] = []
    bonuses: list[float] = []
    for episode in range(config.episodes):
        environment = RandomDiscoMaze(max_steps=config.max_steps, seed=seed + episode)
        observation, _ = environment.reset(seed=seed + episode)
        episodic = EpisodicKNNNovelty()
        for _ in range(config.max_steps):
            action = int(rng.integers(4))
            next_observation, _, terminated, truncated, info = environment.step(action)
            current = torch.from_numpy(observation[None]).float() / 255.0
            following = torch.from_numpy(next_observation[None]).float() / 255.0
            embedding = inverse.encode(current).detach().cpu().numpy()[0]
            episodic_bonus = episodic.reward(embedding)
            inverse.train_step(current, following, torch.tensor([action]), inverse_optimizer)
            if config.lifelong_variant == "rnd":
                novelty = float(rnd.prediction_error(current).detach().item())
                lifelong = max(1.0, rnd_normalizer.normalize(novelty, done=terminated or truncated))
                rnd.train_step(current, rnd_optimizer)
            else:
                lifelong = 1.0
            bonuses.append(episodic_bonus * min(lifelong, 5.0))
            observation = next_observation
            if terminated or truncated:
                break
        coverage.append(int(info["coverage"]))
    result = ResultArtifact(
        run_id=run_id,
        manifest_digest=artifact_digest(manifest.to_dict()),
        track="paper_reproduction",
        preset=preset,
        status="completed",
        actual_counters={"environment_steps": sum(config.max_steps for _ in coverage), "episodes": len(coverage)},
        metrics={
            "mean_unique_cells_visited": float(np.mean(coverage)),
            "mean_intrinsic_reward": float(np.mean(bonuses)),
        },
        created_at=utc_now(),
        provenance={"trained_ngu_agent": False, "lifelong_variant": lifelong_variant},
    )
    result.write(run_dir / "result.json")
    decision = qualify_result(manifest, result)
    (run_dir / "qualification.json").write_text(
        json.dumps({"qualified": decision.qualified, "reasons": decision.reasons}, indent=2) + "\n"
    )
    return run_dir
