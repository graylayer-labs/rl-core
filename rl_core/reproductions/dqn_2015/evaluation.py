"""Paper-comparable final-checkpoint evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_core.environments import AtariPreprocessingConfig, make_atari_environment
from rl_core.integrations import NatureDQN


@dataclass(frozen=True)
class EvaluationSummary:
    """Raw episode evidence and its seed-level aggregate."""

    returns: tuple[float, ...]
    lengths: tuple[int, ...]

    @property
    def mean_return(self) -> float:
        """Mean raw return across evaluation episodes."""
        return float(np.mean(self.returns))

    @property
    def return_std(self) -> float:
        """Population standard deviation across evaluation episodes."""
        return float(np.std(self.returns))


def evaluate(
    agent: NatureDQN,
    environment_id: str,
    seed: int,
    episodes: int,
    max_agent_steps: int,
    epsilon: float,
) -> EvaluationSummary:
    """Evaluate with raw rewards and whole-game episode boundaries."""
    environment = make_atari_environment(
        AtariPreprocessingConfig(
            env_id=environment_id,
            seed=seed,
            terminal_on_life_loss=False,
            clip_rewards=False,
        )
    )
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    lengths: list[int] = []
    try:
        for episode in range(episodes):
            observation, _ = environment.reset(seed=seed + episode)
            episode_return = 0.0
            length = 0
            for _length in range(1, max_agent_steps + 1):
                action = agent.select_action(np.asarray(observation), epsilon, rng)
                observation, reward, terminated, truncated, _ = environment.step(action)
                episode_return += float(reward)
                length = _length
                if terminated or truncated:
                    break
            returns.append(episode_return)
            lengths.append(length)
    finally:
        environment.close()
    return EvaluationSummary(tuple(returns), tuple(lengths))
