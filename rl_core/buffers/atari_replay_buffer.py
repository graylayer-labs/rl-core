"""Memory-efficient replay storage for frame-stacked Atari observations."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor


class AtariReplayBuffer:
    """Store each Atari frame once and reconstruct stacked observations lazily.

    Call :meth:`begin_episode` with the reset observation, then call
    :meth:`append` after each environment step.  An ``append`` records the
    transition from the current frame to ``next_frame``.  Terminal next frames
    are intentionally not stored because they can never bootstrap a target.
    """

    def __init__(
        self,
        capacity: int,
        frame_shape: Sequence[int] = (84, 84),
        stack_size: int = 4,
        seed: int | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if not frame_shape or any(dimension < 1 for dimension in frame_shape):
            raise ValueError("frame_shape must contain positive dimensions")
        if stack_size < 1:
            raise ValueError("stack_size must be positive")
        self._capacity = capacity
        self._frame_shape = tuple(frame_shape)
        self._stack_size = stack_size
        # Retaining an extra stack horizon keeps every valid current state
        # reconstructable after the transition ring wraps.
        self._frame_capacity = capacity + stack_size
        self._frames = np.zeros((self._frame_capacity, *self._frame_shape), dtype=np.uint8)
        self._frame_ids = np.full(self._frame_capacity, -1, dtype=np.int64)
        self._frame_done = np.zeros(self._frame_capacity, dtype=np.bool_)
        self._transition_frame_ids = np.full(capacity, -1, dtype=np.int64)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.bool_)
        self._next_frame_ids = np.full(capacity, -1, dtype=np.int64)
        self._write_index = 0
        self._size = 0
        self._next_frame_id = 0
        self._current_frame_id: int | None = None
        self._rng = np.random.default_rng(seed)

    @property
    def capacity(self) -> int:
        """Maximum number of transitions retained."""
        return self._capacity

    @property
    def frame_shape(self) -> tuple[int, ...]:
        """Shape of one grayscale frame."""
        return self._frame_shape

    @property
    def stack_size(self) -> int:
        """Number of frames reconstructed for each observation."""
        return self._stack_size

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return self._size

    def ready(self, minimum_size: int) -> bool:
        """Return whether at least ``minimum_size`` transitions are available."""
        return self._size >= minimum_size

    def begin_episode(self, initial_frame: np.ndarray) -> None:
        """Start collecting one episode from its reset observation."""
        if self._current_frame_id is not None:
            raise RuntimeError("append a terminal transition before beginning another episode")
        self._current_frame_id = self._store_frame(initial_frame)

    def append(self, action: int, reward: float, next_frame: np.ndarray, done: bool) -> None:
        """Record one transition from the current frame to ``next_frame``."""
        if self._current_frame_id is None:
            raise RuntimeError("call begin_episode before append")
        current_frame_id = self._current_frame_id
        slot = self._write_index
        self._transition_frame_ids[slot] = current_frame_id
        self._actions[slot] = action
        self._rewards[slot] = reward
        self._dones[slot] = done
        self._frame_done[current_frame_id % self._frame_capacity] = done
        if done:
            self._next_frame_ids[slot] = -1
            self._current_frame_id = None
        else:
            next_frame_id = self._store_frame(next_frame)
            self._next_frame_ids[slot] = next_frame_id
            self._current_frame_id = next_frame_id
        self._write_index = (slot + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int, device: torch.device | str) -> dict[str, Tensor]:
        """Sample tensors preserving Atari frames as ``uint8`` until training."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if batch_size > self._size:
            raise ValueError(f"Requested batch of {batch_size} but only {self._size} transitions are available")
        slots = self._sample_slots(batch_size)
        observations = np.stack([self._stack_for(self._transition_frame_ids[slot]) for slot in slots])
        next_observations = np.zeros_like(observations)
        for output_index, slot in enumerate(slots):
            if not self._dones[slot]:
                next_observations[output_index] = self._stack_for(self._next_frame_ids[slot])
        return {
            "obs": torch.as_tensor(observations, dtype=torch.uint8, device=device),
            "action": torch.as_tensor(self._actions[slots, None], dtype=torch.long, device=device),
            "reward": torch.as_tensor(self._rewards[slots, None], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(next_observations, dtype=torch.uint8, device=device),
            "done": torch.as_tensor(self._dones[slots, None], dtype=torch.float32, device=device),
        }

    def _store_frame(self, frame: np.ndarray) -> int:
        array = np.asarray(frame)
        if array.shape != self._frame_shape:
            raise ValueError(f"Expected frame shape {self._frame_shape}, got {array.shape}")
        if array.dtype != np.uint8:
            raise TypeError(f"Atari frames must be uint8, got {array.dtype}")
        frame_id = self._next_frame_id
        frame_slot = frame_id % self._frame_capacity
        self._frames[frame_slot] = array
        self._frame_ids[frame_slot] = frame_id
        self._frame_done[frame_slot] = False
        self._next_frame_id += 1
        return frame_id

    def _sample_slots(self, batch_size: int) -> np.ndarray:
        """Sample reconstructable slots in time proportional to batch size.

        Live transitions occupy ``[0, size)`` until the ring fills and every
        slot thereafter. Frame-ring overwrite can make a few boundary entries
        temporarily unreconstructable, so candidates are rejected in vectorized
        batches rather than finding every valid slot on every optimizer update.
        """
        upper_bound = self._capacity if self._size == self._capacity else self._size
        selected: list[np.ndarray] = []
        selected_count = 0
        for _ in range(32):
            remaining = batch_size - selected_count
            candidate_count = max(remaining * 2, 8)
            candidates = self._rng.integers(0, upper_bound, size=candidate_count, dtype=np.intp)
            frame_ids = self._transition_frame_ids[candidates]
            frame_slots = np.mod(frame_ids, self._frame_capacity)
            current_valid = (frame_ids >= 0) & (self._frame_ids[frame_slots] == frame_ids)
            next_frame_ids = self._next_frame_ids[candidates]
            next_slots = np.mod(np.maximum(next_frame_ids, 0), self._frame_capacity)
            next_valid = self._dones[candidates] | (
                (next_frame_ids >= 0) & (self._frame_ids[next_slots] == next_frame_ids)
            )
            accepted = candidates[current_valid & next_valid][:remaining]
            if accepted.size:
                selected.append(accepted)
                selected_count += int(accepted.size)
            if selected_count == batch_size:
                return np.concatenate(selected)
        raise RuntimeError("Unable to sample enough reconstructable transitions")

    def _has_frame(self, frame_id: int) -> bool:
        return frame_id >= 0 and self._frame_ids[frame_id % self._frame_capacity] == frame_id

    def _stack_for(self, newest_frame_id: int) -> np.ndarray:
        stack = np.zeros((self._stack_size, *self._frame_shape), dtype=np.uint8)
        frame_id = newest_frame_id
        for output_index in range(self._stack_size - 1, -1, -1):
            if not self._has_frame(frame_id):
                break
            stack[output_index] = self._frames[frame_id % self._frame_capacity]
            previous_id = frame_id - 1
            if previous_id < 0 or self._frame_done[previous_id % self._frame_capacity]:
                break
            frame_id = previous_id
        return stack
