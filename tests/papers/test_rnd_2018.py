from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from rl_core.intrinsic.rnd import RNDModel, RNDRewardNormalizer
from rl_core.reproductions.rnd_2018 import preflight_rnd_2018, run_rnd_2018
from rl_core.reproductions.rnd_2018.aggregation import aggregate_rnd_returns
from rl_core.reproductions.rnd_2018.config import load_rnd_run_config


def test_rnd_target_is_frozen_and_predictor_receives_gradients() -> None:
    model = RNDModel((4,), feature_dim=3, hidden_sizes=(8,))
    inputs = torch.ones((2, 4))

    loss = model.loss(inputs)
    loss.backward()

    assert all(not parameter.requires_grad for parameter in model.target.parameters())
    assert all(parameter.grad is None for parameter in model.target.parameters())
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())


def test_rnd_normalizer_resets_discount_at_episode_boundary() -> None:
    normalizer = RNDRewardNormalizer(gamma=0.5)

    first = normalizer.normalize(2.0)
    normalizer.normalize(2.0, done=True)

    assert first > 0
    assert normalizer._discounted_return == 0.0


def test_rnd_frozen_config_preflight_aggregation_and_smoke(tmp_path) -> None:
    config = load_rnd_run_config("ALE/MontezumaRevenge-v5", 7, "smoke")
    assert config.environment_frames == 256
    assert preflight_rnd_2018()["environment_id"] == config.environment_id
    assert aggregate_rnd_returns([1, 2, 3, 4, 5]).seed_mean_return == pytest.approx(3.0)

    run_dir = run_rnd_2018(config.environment_id, config.seed, config.preset, tmp_path)
    result = json.loads((run_dir / "result.json").read_text())
    assert result["actual_counters"]["synthetic_observations"] == 16
    assert np.isfinite(result["metrics"]["mean_normalized_intrinsic_reward"])
