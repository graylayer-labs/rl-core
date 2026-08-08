"""Focused checks for the paper-faithful PPO reproduction primitives."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rl_core.algorithms.a2c import a2c_objective
from rl_core.algorithms.ppo import clipped_surrogate_loss, generalized_advantage_estimation
from rl_core.environments.vector import SynchronousVectorEnv
from rl_core.reproductions import load_reproduction_spec
from rl_core.reproductions.ppo_2017.config import load_ppo_run_config


def test_clipped_surrogate_uses_the_lower_objective_for_positive_advantage() -> None:
    new_log_probabilities = torch.log(torch.tensor([1.3, 0.7]))
    old_log_probabilities = torch.zeros(2)
    advantages = torch.tensor([1.0, -1.0])

    loss, ratios, clipped = clipped_surrogate_loss(new_log_probabilities, old_log_probabilities, advantages, 0.2)

    assert loss.item() == pytest.approx(-0.2)
    assert ratios.tolist() == pytest.approx([1.3, 0.7])
    assert clipped.tolist() == [True, True]


def test_gae_does_not_bootstrap_across_a_done_transition() -> None:
    advantages, returns = generalized_advantage_estimation(
        rewards=torch.tensor([[1.0], [2.0]]),
        values=torch.tensor([[0.5], [0.25]]),
        dones=torch.tensor([[1.0], [0.0]]),
        bootstrap_value=torch.tensor([4.0]),
        gamma=0.9,
        gae_lambda=0.95,
    )

    assert advantages[:, 0].tolist() == pytest.approx([0.5, 5.35])
    assert returns[:, 0].tolist() == pytest.approx([1.0, 5.6])


def test_a2c_boundary_has_no_ppo_ratio_or_clip_parameter() -> None:
    objective = a2c_objective(
        torch.tensor([-0.5]), torch.tensor([2.0]), torch.tensor([1.0]), torch.tensor([3.0]), torch.tensor([0.2])
    )

    assert objective.policy_loss.item() == pytest.approx(1.0)
    assert objective.value_loss.item() == pytest.approx(2.0)


class _Space:
    n = 2


class _Environment:
    observation_space = type("ObservationSpace", (), {"shape": (1,), "dtype": np.dtype("uint8")})()
    action_space = _Space()

    def __init__(self) -> None:
        self.resets = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, int | None]]:
        self.resets += 1
        return np.array([self.resets], dtype=np.uint8), {"seed": seed}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, int]]:
        return np.array([9], dtype=np.uint8), float(action), True, False, {"action": action}

    def close(self) -> None:
        return None


def test_synchronous_vector_environment_resets_finished_instances() -> None:
    vector = SynchronousVectorEnv([_Environment, _Environment])
    vector.reset(seed=10)

    observations, rewards, terminated, truncated, infos = vector.step([0, 1])

    assert observations.tolist() == [[2], [2]]
    assert rewards.tolist() == [0.0, 1.0]
    assert terminated.tolist() == [True, True]
    assert truncated.tolist() == [False, False]
    assert [info["final_observation"].tolist() for info in infos] == [[9], [9]]


def test_frozen_ppo_protocol_and_smoke_budget_are_loadable() -> None:
    spec = load_reproduction_spec("papers/ppo-2017/reproduction.yaml")
    config = load_ppo_run_config("ALE/Pong-v5", 7, "smoke")

    assert spec.key == "ppo-2017"
    assert len(spec.protocol.games) == 5
    assert config.environment_steps == config.num_envs * config.rollout_steps * 2
    assert config.ppo.epochs == 4
