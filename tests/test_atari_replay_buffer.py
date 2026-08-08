import numpy as np
import pytest
import torch

from rl_core.buffers import AtariReplayBuffer


def _frame(value: int) -> np.ndarray:
    return np.full((2, 2), value, dtype=np.uint8)


def test_atari_replay_stores_each_frame_once_and_reconstructs_stacks():
    buffer = AtariReplayBuffer(capacity=8, frame_shape=(2, 2), stack_size=4, seed=0)
    buffer.begin_episode(_frame(1))
    buffer.append(action=0, reward=1.0, next_frame=_frame(2), done=False)
    buffer.append(action=1, reward=2.0, next_frame=_frame(3), done=False)

    assert buffer._frames.shape == (12, 2, 2)
    assert np.array_equal(buffer._stack_for(0)[:, 0, 0], np.array([0, 0, 0, 1]))
    assert np.array_equal(buffer._stack_for(1)[:, 0, 0], np.array([0, 0, 1, 2]))
    batch = buffer.sample(2, "cpu")
    assert batch["obs"].dtype == torch.uint8
    assert batch["obs"].shape == (2, 4, 2, 2)
    assert batch["action"].dtype == torch.long


def test_atari_replay_zero_pads_new_episode_and_terminal_next_state():
    buffer = AtariReplayBuffer(capacity=8, frame_shape=(2, 2), stack_size=4, seed=1)
    buffer.begin_episode(_frame(1))
    buffer.append(action=0, reward=1.0, next_frame=_frame(2), done=True)
    buffer.begin_episode(_frame(9))
    buffer.append(action=1, reward=0.0, next_frame=_frame(10), done=False)

    assert np.array_equal(buffer._stack_for(1)[:, 0, 0], np.array([0, 0, 0, 9]))
    terminal_batch = buffer.sample(1, "cpu")
    # The first seed draw selects the terminal transition; terminal next states
    # are explicitly zeroed rather than leaking a reset observation.
    assert terminal_batch["done"].item() == 1.0
    assert torch.count_nonzero(terminal_batch["next_obs"]) == 0


def test_atari_replay_requires_a_reset_and_uint8_frames():
    buffer = AtariReplayBuffer(capacity=2, frame_shape=(2, 2))
    with pytest.raises(RuntimeError, match="begin_episode"):
        buffer.append(0, 0.0, _frame(1), False)
    with pytest.raises(TypeError, match="uint8"):
        buffer.begin_episode(np.ones((2, 2), dtype=np.float32))
