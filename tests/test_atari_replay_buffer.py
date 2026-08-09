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


def test_atari_replay_samples_after_ring_wrap_without_full_scan():
    buffer = AtariReplayBuffer(capacity=16, frame_shape=(2, 2), stack_size=4, seed=2)
    buffer.begin_episode(_frame(0))
    for step in range(40):
        buffer.append(action=step % 2, reward=float(step), next_frame=_frame(step + 1), done=False)

    batch = buffer.sample(16, "cpu")

    assert len(buffer) == 16
    assert batch["obs"].shape == (16, 4, 2, 2)
    assert torch.all(batch["done"] == 0)


def test_atari_replay_sampling_work_is_independent_of_capacity(monkeypatch):
    buffer = AtariReplayBuffer(capacity=1_000_000, frame_shape=(2, 2), seed=3)
    buffer.begin_episode(_frame(1))
    for step in range(64):
        buffer.append(action=0, reward=0.0, next_frame=_frame(step + 2), done=False)

    sampled_candidate_counts: list[int] = []
    original_integers = buffer._rng.integers

    class _CountingRng:
        def integers(self, *args, **kwargs):
            result = original_integers(*args, **kwargs)
            sampled_candidate_counts.append(int(result.size))
            return result

    monkeypatch.setattr(buffer, "_rng", _CountingRng())
    buffer.sample(32, "cpu")

    assert sum(sampled_candidate_counts) <= 64
