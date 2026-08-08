"""Small synchronous vector-environment adapter for deterministic PPO rollout."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


class SynchronousVectorEnv:
    """Step several Gymnasium-style environments in deterministic order.

    Finished environments are reset immediately, while their terminal facts
    remain available in ``infos``.  This makes every rollout rectangular and
    preserves the transition boundary required by GAE.
    """

    def __init__(self, factories: Sequence[Callable[[], Any]]) -> None:
        if not factories:
            raise ValueError("at least one environment factory is required")
        self.environments = [factory() for factory in factories]
        self.single_observation_space = self.environments[0].observation_space
        self.single_action_space = self.environments[0].action_space
        if not hasattr(self.single_action_space, "n"):
            raise TypeError("PPO vector environments require a discrete action space")

    @property
    def num_envs(self) -> int:
        """Return the fixed number of managed environments."""
        return len(self.environments)

    def reset(self, seed: int | Sequence[int] | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Reset all environments and stack their observations."""
        if isinstance(seed, int):
            seeds: Sequence[int | None] = [seed + index for index in range(self.num_envs)]
        elif seed is None:
            seeds = [None] * self.num_envs
        else:
            if len(seed) != self.num_envs:
                raise ValueError("one reset seed is required for each environment")
            seeds = seed
        outcomes = [environment.reset(seed=item) for environment, item in zip(self.environments, seeds, strict=True)]
        observations, infos = zip(*outcomes, strict=True)
        return np.stack(observations), list(infos)

    def step(
        self, actions: Sequence[int] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Step all environments and immediately reset completed instances."""
        if len(actions) != self.num_envs:
            raise ValueError("one action is required for each environment")
        observations: list[np.ndarray] = []
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        infos: list[dict[str, Any]] = []
        for environment, action in zip(self.environments, actions, strict=True):
            observation, reward, did_terminate, did_truncate, info = environment.step(int(action))
            done = bool(did_terminate or did_truncate)
            saved_info = dict(info)
            if done:
                saved_info["final_observation"] = observation
                reset_observation, reset_info = environment.reset()
                saved_info["reset_info"] = reset_info
                observation = reset_observation
            observations.append(np.asarray(observation))
            rewards.append(float(reward))
            terminated.append(bool(did_terminate))
            truncated.append(bool(did_truncate))
            infos.append(saved_info)
        return (
            np.stack(observations),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
            infos,
        )

    def close(self) -> None:
        """Close all child environments even if one close operation fails."""
        errors: list[Exception] = []
        for environment in self.environments:
            try:
                environment.close()
            except Exception as error:  # pragma: no cover - defensive cleanup
                errors.append(error)
        if errors:
            raise errors[0]
