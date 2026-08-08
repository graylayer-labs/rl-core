"""Artifact-producing, dependency-light RND component runner."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from rl_core.experiments import (
    ResultArtifact,
    RunManifest,
    artifact_digest,
    collect_provenance,
    qualify_result,
    utc_now,
)
from rl_core.intrinsic import RNDModel, RNDRewardNormalizer
from rl_core.reproductions.rnd_2018.config import RNDRunConfig, load_rnd_run_config
from rl_core.reproductions.rnd_2018.preflight import preflight_rnd_2018
from rl_core.utils.seeding import seed_everything


def _run_id(config: RNDRunConfig, commit: str | None) -> str:
    payload = json.dumps({"config": asdict(config), "commit": commit}, sort_keys=True, default=str)
    return f"rnd-2018__{config.preset}__seed{config.seed}__{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def run_rnd_2018(environment_id: str, seed: int, preset: str, runs_dir: Path = Path("runs")) -> Path:
    """Exercise RND math on seeded synthetic observations and write evidence.

    The resulting artifact is deliberately nonqualifying until a complete PPO
    and Atari training loop is connected.
    """
    preflight = preflight_rnd_2018()
    config = load_rnd_run_config(environment_id, seed, preset)
    provenance = collect_provenance(repo_dir=Path.cwd(), packages=("numpy", "torch", "pyyaml"))
    run_id = _run_id(config, provenance["git"]["commit"])
    run_dir = runs_dir / "rnd-2018" / run_id
    manifest = RunManifest(
        run_id=run_id,
        reproduction_key="rnd-2018",
        protocol_revision=str(config.protocol_revision),
        track="paper_reproduction",
        preset=preset,
        seed=seed,
        preset_counters={
            "environment_frames": config.environment_frames,
            "synthetic_observations": config.synthetic_observations,
        },
        config_digest=config.source_digest,
        configuration={
            "resolved_run": asdict(config),
            "preflight": preflight,
            "implementation_boundary": preflight["implementation_boundary"],
        },
        created_at=utc_now(),
        provenance=provenance,
    )
    manifest.write(run_dir / "manifest.json")
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    model = RNDModel((64,), feature_dim=32, hidden_sizes=(64, 64))
    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=1e-3)
    normalizer = RNDRewardNormalizer(config.intrinsic_discount)
    novelty: list[float] = []
    for index in range(config.synthetic_observations):
        batch = torch.from_numpy(rng.normal(size=(1, 64)).astype(np.float32))
        error = float(model.prediction_error(batch).detach().item())
        novelty.append(normalizer.normalize(error, done=index == config.synthetic_observations - 1))
        model.train_step(batch, optimizer)
    result = ResultArtifact(
        run_id=run_id,
        manifest_digest=artifact_digest(manifest.to_dict()),
        track="paper_reproduction",
        preset=preset,
        status="completed",
        actual_counters={
            "environment_frames": config.environment_frames,
            "synthetic_observations": config.synthetic_observations,
        },
        metrics={
            "mean_normalized_intrinsic_reward": float(np.mean(novelty)),
            "final_normalized_intrinsic_reward": novelty[-1],
        },
        created_at=utc_now(),
        provenance={"synthetic_observations": True, "ppo_atari_integrated": False},
    )
    result.write(run_dir / "result.json")
    decision = qualify_result(manifest, result)
    (run_dir / "qualification.json").write_text(
        json.dumps({"qualified": decision.qualified, "reasons": decision.reasons}, indent=2) + "\n"
    )
    return run_dir
