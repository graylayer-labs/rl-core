from __future__ import annotations

import numpy as np
import pytest
import torch

from rl_core.algorithms.rainbow import NoisyLinear, Rainbow, RainbowConfig, RainbowNetwork
from rl_core.buffers.prioritized_atari_replay_buffer import PrioritizedAtariReplayBuffer
from rl_core.reproductions import load_reproduction_spec
from rl_core.reproductions.rainbow_2018.aggregation import aggregate_rainbow_2018
from rl_core.reproductions.rainbow_2018.config import load_rainbow_run_config, priority_beta


def test_n_step_replay_records_discounted_return_and_bootstrap_state() -> None:
    replay = PrioritizedAtariReplayBuffer(8, n_step=3, gamma=0.9, frame_shape=(1, 1), stack_size=1, seed=7)
    replay.begin_episode(np.array([[0]], dtype=np.uint8))
    replay.append(1, 1.0, np.array([[1]], dtype=np.uint8), False)
    replay.append(2, 2.0, np.array([[2]], dtype=np.uint8), False)
    replay.append(3, 3.0, np.array([[3]], dtype=np.uint8), False)

    batch = replay.sample(1, "cpu")

    assert len(replay) == 1
    assert batch["reward"].item() == pytest.approx(1.0 + 0.9 * 2.0 + 0.9**2 * 3.0)
    assert batch["discount"].item() == pytest.approx(0.9**3)
    assert batch["obs"].item() == 0
    assert batch["next_obs"].item() == 3


def test_terminal_flushes_short_n_step_returns() -> None:
    replay = PrioritizedAtariReplayBuffer(8, n_step=3, gamma=0.5, frame_shape=(1, 1), stack_size=1, seed=7)
    replay.begin_episode(np.array([[0]], dtype=np.uint8))
    replay.append(0, 2.0, np.array([[1]], dtype=np.uint8), False)
    replay.append(0, 4.0, np.array([[2]], dtype=np.uint8), True)

    assert len(replay) == 2
    batch = replay.sample(2, "cpu")
    assert torch.all(batch["done"] == 1.0)
    assert sorted(batch["reward"].squeeze(1).tolist()) == pytest.approx([4.0, 4.0])


def test_prioritized_replay_emits_indices_and_normalized_weights() -> None:
    replay = PrioritizedAtariReplayBuffer(8, n_step=1, frame_shape=(1, 1), stack_size=1, seed=3)
    replay.begin_episode(np.array([[0]], dtype=np.uint8))
    for frame in range(1, 5):
        replay.append(0, 0.0, np.array([[frame]], dtype=np.uint8), False)
    replay.update_priorities(np.array([0, 1, 2]), np.array([0.0, 1.0, 10.0]))
    batch = replay.sample(4, "cpu", beta=0.6)

    assert batch["indices"].dtype == torch.long
    assert torch.all(batch["weights"] > 0.0)
    assert batch["weights"].max().item() == pytest.approx(1.0)


def test_c51_projection_preserves_probability_mass_and_terminal_support() -> None:
    agent = Rainbow(RainbowConfig(action_dim=2, noisy=False), "cpu")
    distribution = torch.zeros(2, 51)
    distribution[:, 25] = 1.0
    projection = agent.project_distribution(
        distribution, torch.tensor([[0.0], [10.0]]), torch.tensor([[0.0], [1.0]]), torch.tensor([[0.99], [0.99]])
    )

    assert torch.allclose(projection.sum(dim=1), torch.ones(2))
    assert projection[0, 25].item() == pytest.approx(1.0)
    assert projection[1, -1].item() == pytest.approx(1.0)


def test_all_rainbow_components_can_be_disabled() -> None:
    config = RainbowConfig(
        action_dim=3,
        double_q=False,
        dueling=False,
        prioritized_replay=False,
        n_step=1,
        distributional=False,
        noisy=False,
    )
    network = RainbowNetwork(action_dim=3, dueling=False, distributional=False, noisy=False)
    outputs = network(torch.zeros(2, 4, 84, 84, dtype=torch.uint8))

    assert outputs.shape == (2, 3)
    assert config.n_step == 1


def test_noisy_linear_changes_after_noise_reset_but_not_in_mean_mode() -> None:
    layer = NoisyLinear(3, 2)
    inputs = torch.ones(1, 3)
    noisy_before = layer(inputs, use_noise=True)
    layer.reset_noise()
    noisy_after = layer(inputs, use_noise=True)

    assert not torch.equal(noisy_before, noisy_after)
    assert torch.equal(layer(inputs, use_noise=False), layer(inputs, use_noise=False))


def test_rainbow_config_presets_and_protocol_are_frozen() -> None:
    smoke = load_rainbow_run_config("ALE/Pong-v5", 7, "smoke")
    pilot = load_rainbow_run_config("ALE/Pong-v5", 7, "pilot")
    qualifying = load_rainbow_run_config("ALE/Pong-v5", 7, "qualifying")

    assert smoke.agent_steps == 512
    assert pilot.agent_steps == 1_000_000
    assert qualifying.agent_steps == 50_000_000
    assert qualifying.raw_ale_frames == 200_000_000
    assert smoke.component_switches == {
        "double_q": True,
        "dueling": True,
        "prioritized_replay": True,
        "n_step": True,
        "distributional": True,
        "noisy": True,
    }
    assert priority_beta(0, qualifying) == pytest.approx(0.4)
    assert priority_beta(50_000_000, qualifying) == pytest.approx(0.4)


def test_aggregation_requires_all_declared_seeds() -> None:
    spec = load_reproduction_spec("papers/rainbow-2018/reproduction.yaml")
    scores = {game.id: [game.paper_score] * len(spec.seeds) for game in spec.protocol.games}

    decision = aggregate_rainbow_2018(spec, scores)

    assert decision.reproduced
    assert decision.passed_games == 5
