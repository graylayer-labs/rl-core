"""Prioritized, n-step replay storage for frame-stacked Atari observations."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class _PendingTransition:
    """A one-step transition retained until its n-step return is known."""

    frame_id: int
    action: int
    reward: float
    next_frame_id: int
    done: bool


class PrioritizedAtariReplayBuffer:
    """Frame-efficient proportional prioritized replay with exact n-step returns.

    Frames are retained once and states are reconstructed lazily.  A transition
    is committed only after enough following transitions exist to calculate its
    n-step return; terminal episodes flush the remaining shorter returns.
    """

    def __init__(
        self,
        capacity: int,
        *,
        n_step: int = 3,
        gamma: float = 0.99,
        priority_alpha: float = 0.5,
        priority_epsilon: float = 1e-6,
        frame_shape: Sequence[int] = (84, 84),
        stack_size: int = 4,
        seed: int | None = None,
    ) -> None:
        if capacity < 1 or n_step < 1 or stack_size < 1:
            raise ValueError("capacity, n_step, and stack_size must be positive")
        if not frame_shape or any(dimension < 1 for dimension in frame_shape):
            raise ValueError("frame_shape must contain positive dimensions")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if priority_alpha < 0.0 or priority_epsilon <= 0.0:
            raise ValueError("priority_alpha must be non-negative and priority_epsilon positive")
        self._capacity = capacity
        self._n_step = n_step
        self._gamma = gamma
        self._priority_alpha = priority_alpha
        self._priority_epsilon = priority_epsilon
        self._frame_shape = tuple(frame_shape)
        self._stack_size = stack_size
        self._frame_capacity = capacity + stack_size + n_step
        self._frames = np.zeros((self._frame_capacity, *self._frame_shape), dtype=np.uint8)
        self._frame_ids = np.full(self._frame_capacity, -1, dtype=np.int64)
        self._frame_done = np.zeros(self._frame_capacity, dtype=np.bool_)
        self._transition_frame_ids = np.full(capacity, -1, dtype=np.int64)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.bool_)
        self._next_frame_ids = np.full(capacity, -1, dtype=np.int64)
        self._discounts = np.zeros(capacity, dtype=np.float32)
        self._priorities = np.zeros(capacity, dtype=np.float64)
        self._write_index = 0
        self._size = 0
        self._next_frame_id = 0
        self._current_frame_id: int | None = None
        self._pending: deque[_PendingTransition] = deque()
        self._max_priority = 1.0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Return the number of committed n-step transitions."""
        return self._size

    @property
    def capacity(self) -> int:
        """Maximum number of committed transitions retained."""
        return self._capacity

    def ready(self, minimum_size: int) -> bool:
        """Return whether at least ``minimum_size`` transitions are sampleable."""
        return self._size >= minimum_size

    def begin_episode(self, initial_frame: np.ndarray) -> None:
        """Start a new episode from its reset frame."""
        if self._current_frame_id is not None or self._pending:
            raise RuntimeError("append a terminal transition before beginning another episode")
        self._current_frame_id = self._store_frame(initial_frame)

    def append(self, action: int, reward: float, next_frame: np.ndarray, done: bool) -> None:
        """Append one raw transition and commit every newly available n-step item."""
        if self._current_frame_id is None:
            raise RuntimeError("call begin_episode before append")
        current_frame_id = self._current_frame_id
        self._frame_done[current_frame_id % self._frame_capacity] = done
        next_frame_id = -1 if done else self._store_frame(next_frame)
        self._pending.append(
            _PendingTransition(current_frame_id, int(action), float(reward), next_frame_id, bool(done))
        )
        self._current_frame_id = None if done else next_frame_id
        if done:
            while self._pending:
                self._commit_oldest()
        elif len(self._pending) >= self._n_step:
            self._commit_oldest()

    def sample(self, batch_size: int, device: torch.device | str, beta: float = 0.4) -> dict[str, Tensor]:
        """Sample proportional-priority transitions with normalized IS weights."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        slots = self._valid_slots()
        if batch_size > len(slots):
            raise ValueError(f"Requested batch of {batch_size} but only {len(slots)} transitions are available")
        priorities = self._priorities[slots]
        scaled = np.power(priorities, self._priority_alpha)
        probabilities = scaled / scaled.sum()
        chosen_positions = self._rng.choice(len(slots), size=batch_size, replace=True, p=probabilities)
        chosen_slots = slots[chosen_positions]
        selected_probabilities = probabilities[chosen_positions]
        weights = np.power(len(slots) * selected_probabilities, -beta)
        weights /= weights.max()
        observations = np.stack([self._stack_for(self._transition_frame_ids[slot]) for slot in chosen_slots])
        next_observations = np.zeros_like(observations)
        for output_index, slot in enumerate(chosen_slots):
            if not self._dones[slot]:
                next_observations[output_index] = self._stack_for(self._next_frame_ids[slot])
        return {
            "obs": torch.as_tensor(observations, dtype=torch.uint8, device=device),
            "action": torch.as_tensor(self._actions[chosen_slots, None], dtype=torch.long, device=device),
            "reward": torch.as_tensor(self._rewards[chosen_slots, None], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(next_observations, dtype=torch.uint8, device=device),
            "done": torch.as_tensor(self._dones[chosen_slots, None], dtype=torch.float32, device=device),
            "discount": torch.as_tensor(self._discounts[chosen_slots, None], dtype=torch.float32, device=device),
            "weights": torch.as_tensor(weights[:, None], dtype=torch.float32, device=device),
            "indices": torch.as_tensor(chosen_slots, dtype=torch.long, device=device),
        }

    def update_priorities(self, indices: np.ndarray | Tensor, priorities: np.ndarray | Tensor) -> None:
        """Update sampled transition priorities from non-negative TD errors."""
        slot_array = np.asarray(indices.detach().cpu() if isinstance(indices, Tensor) else indices, dtype=np.intp)
        priority_array = np.asarray(
            priorities.detach().cpu() if isinstance(priorities, Tensor) else priorities, dtype=np.float64
        )
        slot_array = slot_array.reshape(-1)
        priority_array = priority_array.reshape(-1)
        if len(slot_array) != len(priority_array):
            raise ValueError("indices and priorities must have the same length")
        if np.any(slot_array < 0) or np.any(slot_array >= self._capacity):
            raise ValueError("priority index is outside the replay capacity")
        if np.any(~np.isfinite(priority_array)) or np.any(priority_array < 0.0):
            raise ValueError("priorities must be finite and non-negative")
        updated = priority_array + self._priority_epsilon
        self._priorities[slot_array] = updated
        self._max_priority = max(self._max_priority, float(updated.max(initial=self._priority_epsilon)))

    def _commit_oldest(self) -> None:
        start = self._pending[0]
        total_reward = 0.0
        discount = 1.0
        done = False
        next_frame_id = -1
        for transition in list(self._pending)[: self._n_step]:
            total_reward += discount * transition.reward
            discount *= self._gamma
            if transition.done:
                done = True
                break
            next_frame_id = transition.next_frame_id
        slot = self._write_index
        self._transition_frame_ids[slot] = start.frame_id
        self._actions[slot] = start.action
        self._rewards[slot] = total_reward
        self._dones[slot] = done
        self._next_frame_ids[slot] = -1 if done else next_frame_id
        self._discounts[slot] = discount
        self._priorities[slot] = self._max_priority
        self._write_index = (slot + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)
        self._pending.popleft()

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

    def _valid_slots(self) -> np.ndarray:
        slots = np.flatnonzero(self._transition_frame_ids >= 0)
        valid = [
            int(slot)
            for slot in slots
            if self._has_frame(self._transition_frame_ids[slot])
            and (self._dones[slot] or self._has_frame(self._next_frame_ids[slot]))
        ]
        if not valid:
            raise RuntimeError("No reconstructable transitions are available")
        return np.asarray(valid, dtype=np.intp)

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
