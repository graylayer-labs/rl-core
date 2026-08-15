import numpy as np
import pytest
import torch

from rl_core.integrations import NatureDQN, NatureDQNConfig, NatureQNetwork


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "obs": torch.randint(0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8),
        "action": torch.tensor([[0], [1]][:batch_size], dtype=torch.long),
        "reward": torch.ones((batch_size, 1)),
        "next_obs": torch.randint(0, 256, (batch_size, 4, 84, 84), dtype=torch.uint8),
        "done": torch.zeros((batch_size, 1)),
    }


def test_nature_network_accepts_raw_atari_frames():
    network = NatureQNetwork(action_dim=3)

    values = network(torch.zeros((2, 4, 84, 84), dtype=torch.uint8))

    assert values.shape == (2, 3)


def test_nature_dqn_syncs_target_on_optimizer_update_schedule():
    torch.manual_seed(0)
    dqn = NatureDQN(NatureDQNConfig(action_dim=3, target_update_interval=2))
    batch = _batch()

    first = dqn.train_step(batch)
    assert first["optimizer_updates"] == 1.0
    assert first["target_updated"] == 0.0
    assert any(
        not torch.equal(online, target)
        for online, target in zip(dqn.online_network.parameters(), dqn.target_network.parameters(), strict=True)
    )
    second = dqn.train_step(batch)

    assert second["optimizer_updates"] == 2.0
    assert second["target_updated"] == 1.0
    assert all(
        torch.equal(online, target)
        for online, target in zip(dqn.online_network.parameters(), dqn.target_network.parameters(), strict=True)
    )


def test_nature_dqn_epsilon_one_uses_caller_owned_random_generator():
    dqn = NatureDQN(NatureDQNConfig(action_dim=3))
    rng = np.random.default_rng(9)

    action = dqn.select_action(np.zeros((4, 84, 84), dtype=np.uint8), epsilon=1.0, rng=rng)

    assert 0 <= action < 3


def test_nature_dqn_uses_paper_rmsprop_momentum():
    dqn = NatureDQN(NatureDQNConfig(action_dim=2))
    assert dqn.optimizer.param_groups[0]["momentum"] == pytest.approx(0.95)
