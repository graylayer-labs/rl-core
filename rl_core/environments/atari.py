"""Paper-oriented Atari environment construction.

Gymnasium and ALE are deliberately imported lazily.  Importing :mod:`rl_core`
therefore remains possible on machines used for non-Atari development.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


class AtariDependencyError(ImportError):
    """Raised when an Atari environment is requested without its optional extras."""


@dataclass(frozen=True)
class AtariPreprocessingConfig:
    """The fixed visual-observation protocol used by the Nature DQN paper.

    ``frame_skip`` is applied only by ``AtariPreprocessing``; the raw ALE
    environment is always created with ``frameskip=1`` to prevent accidental
    double action-repeat.
    """

    env_id: str
    seed: int | None = None
    frame_skip: int = 4
    screen_size: int = 84
    frame_stack: int = 4
    noop_max: int = 30
    grayscale_observation: bool = True
    terminal_on_life_loss: bool = False
    clip_rewards: bool = True
    sticky_actions: bool = False
    full_action_space: bool = False

    def __post_init__(self) -> None:
        """Validate the fixed preprocessing protocol."""
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be positive")
        if self.screen_size < 1 or self.frame_stack < 1:
            raise ValueError("screen_size and frame_stack must be positive")
        if self.noop_max < 0:
            raise ValueError("noop_max cannot be negative")


@dataclass(frozen=True)
class AtariEnvironmentInfo:
    """Serializable facts about a constructed Atari environment."""

    env_id: str
    observation_shape: tuple[int, ...]
    observation_dtype: str
    action_count: int
    wrappers: tuple[str, ...]
    preprocessing: dict[str, Any]


def _import_gymnasium() -> Any:
    try:
        import ale_py
        import gymnasium
    except ImportError as error:
        message = (
            "Atari support requires the optional Gymnasium/ALE dependencies. "
            "Install the project's Atari extra, then install the ROMs accepted by ALE."
        )
        raise AtariDependencyError(message) from error
    gymnasium.register_envs(ale_py)
    return gymnasium


def _wrapper_names(environment: Any) -> tuple[str, ...]:
    names: list[str] = []
    current = environment
    while hasattr(current, "env"):
        names.append(type(current).__name__)
        current = current.env
    names.append(type(current).__name__)
    return tuple(names)


def inspect_atari_environment(
    environment: Any,
    config: AtariPreprocessingConfig,
) -> AtariEnvironmentInfo:
    """Return the observable protocol facts needed for a run manifest."""
    observation_space = environment.observation_space
    action_space = environment.action_space
    if not hasattr(observation_space, "shape") or observation_space.shape is None:
        raise TypeError("Atari environment must expose a shaped observation space")
    if not hasattr(action_space, "n"):
        raise TypeError("Atari environment must expose a discrete action space")
    return AtariEnvironmentInfo(
        env_id=config.env_id,
        observation_shape=tuple(int(size) for size in observation_space.shape),
        observation_dtype=np.dtype(observation_space.dtype).name,
        action_count=int(action_space.n),
        wrappers=_wrapper_names(environment),
        preprocessing=asdict(config),
    )


def make_atari_environment(
    config: AtariPreprocessingConfig,
    *,
    gym_module: Any | None = None,
    environment_factory: Callable[..., Any] | None = None,
) -> Any:
    """Construct a deterministic, frame-stacked ALE environment.

    The optional injection points keep this module unit-testable without an
    Atari installation.  Production callers should use the defaults.
    """
    gym = gym_module if gym_module is not None else _import_gymnasium()
    make = environment_factory if environment_factory is not None else gym.make
    environment = make(
        config.env_id,
        frameskip=1,
        repeat_action_probability=0.0 if not config.sticky_actions else 0.25,
        full_action_space=config.full_action_space,
    )
    environment = gym.wrappers.AtariPreprocessing(
        environment,
        noop_max=config.noop_max,
        frame_skip=config.frame_skip,
        screen_size=config.screen_size,
        terminal_on_life_loss=config.terminal_on_life_loss,
        grayscale_obs=config.grayscale_observation,
        scale_obs=False,
    )
    # Gymnasium renamed FrameStack to FrameStackObservation.  Supporting both
    # avoids coupling the protocol to a particular minor Gymnasium release.
    frame_stack = getattr(gym.wrappers, "FrameStackObservation", None)
    if frame_stack is None:
        frame_stack = gym.wrappers.FrameStack
    environment = frame_stack(environment, config.frame_stack)
    if config.clip_rewards:
        environment = gym.wrappers.TransformReward(environment, lambda reward: float(np.clip(reward, -1, 1)))
    if config.seed is not None:
        environment.reset(seed=config.seed)
        environment.action_space.seed(config.seed)
    return environment
