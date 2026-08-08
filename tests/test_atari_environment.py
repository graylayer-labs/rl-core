from types import SimpleNamespace

import numpy as np

from rl_core.environments import AtariPreprocessingConfig, inspect_atari_environment, make_atari_environment


class _Space:
    def __init__(self, shape=(84, 84), dtype=np.uint8, n=4):
        self.shape = shape
        self.dtype = dtype
        self.n = n
        self.seed_value = None

    def seed(self, value):
        self.seed_value = value


class _Environment:
    def __init__(self):
        self.observation_space = _Space()
        self.action_space = _Space()
        self.reset_seed = None

    def reset(self, *, seed=None):
        self.reset_seed = seed


class _Wrapper:
    def __init__(self, environment, *args, **kwargs):
        self.env = environment
        self.observation_space = environment.observation_space
        self.action_space = environment.action_space
        self.args = args
        self.kwargs = kwargs

    def reset(self, *, seed=None):
        return self.env.reset(seed=seed)


class _AtariPreprocessing(_Wrapper):
    pass


class _FrameStack(_Wrapper):
    def __init__(self, environment, stack_size):
        super().__init__(environment, stack_size)
        self.observation_space = _Space(shape=(stack_size, *environment.observation_space.shape))


class _TransformReward(_Wrapper):
    pass


def test_make_atari_environment_applies_paper_preprocessing_without_double_frameskip():
    raw_environment = _Environment()
    captured = {}

    def make(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return raw_environment

    gym = SimpleNamespace(
        make=make,
        wrappers=SimpleNamespace(
            AtariPreprocessing=_AtariPreprocessing,
            FrameStackObservation=_FrameStack,
            TransformReward=_TransformReward,
        ),
    )
    config = AtariPreprocessingConfig(env_id="ALE/Breakout-v5", seed=7)

    environment = make_atari_environment(config, gym_module=gym)

    assert captured == {
        "args": ("ALE/Breakout-v5",),
        "kwargs": {"frameskip": 1, "repeat_action_probability": 0.0, "full_action_space": False},
    }
    preprocessing = environment.env.env
    assert preprocessing.kwargs["frame_skip"] == 4
    assert preprocessing.kwargs["noop_max"] == 30
    assert raw_environment.reset_seed == 7
    assert raw_environment.action_space.seed_value == 7


def test_inspect_atari_environment_records_effective_spaces_and_wrapper_chain():
    raw_environment = _Environment()
    environment = _TransformReward(_FrameStack(_AtariPreprocessing(raw_environment), 4), lambda reward: reward)
    config = AtariPreprocessingConfig(env_id="ALE/Pong-v5")

    info = inspect_atari_environment(environment, config)

    assert info.observation_shape == (4, 84, 84)
    assert info.observation_dtype == "uint8"
    assert info.action_count == 4
    assert info.wrappers == ("_TransformReward", "_FrameStack", "_AtariPreprocessing", "_Environment")
    assert info.preprocessing["frame_skip"] == 4
