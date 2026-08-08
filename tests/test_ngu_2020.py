from __future__ import annotations

import json

import numpy as np
import torch

from rl_core.environments.disco_maze import RandomDiscoMaze
from rl_core.intrinsic.ngu import EpisodicKNNNovelty, InverseDynamicsEmbedding
from rl_core.reproductions.ngu_2020 import preflight_ngu_2020, run_ngu_2020
from rl_core.reproductions.ngu_2020.aggregation import aggregate_coverage
from rl_core.reproductions.ngu_2020.config import load_ngu_run_config


def test_disco_maze_is_seeded_visual_and_tracks_coverage() -> None:
    first = RandomDiscoMaze(seed=7)
    second = RandomDiscoMaze(seed=7)
    observation, info = first.reset(seed=7)
    expected, _ = second.reset(seed=7)

    assert observation.shape == (21, 21, 3)
    assert observation.dtype == np.uint8
    assert np.array_equal(observation, expected)
    _, _, _, _, after = first.step(1)
    assert after["coverage"] >= info["coverage"]


def test_episodic_knn_and_inverse_dynamics_are_callable() -> None:
    novelty = EpisodicKNNNovelty(c=1.0)
    first = novelty.reward(np.array([0.0, 0.0]))
    repeated = novelty.reward(np.array([0.0, 0.0]))
    assert repeated < first

    embedding = InverseDynamicsEmbedding((2,), action_dim=4, embedding_dim=3, hidden_dim=8)
    optimizer = torch.optim.Adam(embedding.parameters())
    loss = embedding.train_step(torch.ones((2, 2)), torch.zeros((2, 2)), torch.tensor([0, 1]), optimizer)
    assert loss > 0


def test_ngu_config_preflight_aggregation_and_smoke(tmp_path) -> None:
    config = load_ngu_run_config("RandomDiscoMaze-21x21-v1", 7, "smoke", "random")
    assert config.episodes == 1
    assert preflight_ngu_2020()["action_count"] == 4
    aggregate = aggregate_coverage({"random": [2] * 5, "rnd": [3] * 5})
    assert [item.mean_unique_cells for item in aggregate] == [2.0, 3.0]

    run_dir = run_ngu_2020(config.environment_id, config.seed, config.preset, "random", tmp_path)
    result = json.loads((run_dir / "result.json").read_text())
    assert result["actual_counters"]["episodes"] == 1
    assert result["metrics"]["mean_unique_cells_visited"] >= 1
