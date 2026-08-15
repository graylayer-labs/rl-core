# Test Suite

Tests are organized by scope:

## `algorithms/`
Core algorithm implementations and components. These validate individual pieces (DQN, RMSProp, replay buffers) in isolation.

- `test_nature_dqn.py` — DQN value network and updates
- `test_deepmind_rmsprop.py` — RMSProp optimizer behavior
- `test_atari_replay_buffer.py` — Replay buffer storage and sampling

## `papers/`
Paper-specific reproduction tests. Each test validates that a paper's implementation matches its protocol and produces expected learning signals.

- `test_dqn_2015.py` — DQN integration and learning behavior
- `test_dqn_2015_aggregation.py` — DQN result aggregation and qualification
- `test_rainbow_2018.py` — Rainbow components and ablations
- `test_ppo_2017.py` — PPO collection and updates
- `test_rnd_2018.py` — RND predictor and reward normalization
- `test_ngu_2020.py` — NGU episodic and lifelong novelty

## `framework/`
Experiment infrastructure and system tests. These validate the reproduction protocol, artifact handling, environment wrappers, and leaderboard logic.

- `test_reproductions.py` — Reproduction runner and command dispatch
- `test_reproduction_registry.py` — Protocol loading and validation
- `test_experiments.py` — Experiment orchestration
- `test_artifacts.py` — Run artifacts and checkpointing
- `test_leaderboard.py` — Results aggregation and qualification
- `test_qualification.py` — Qualification logic and evidence gates
- `test_atari_environment.py` — ALE wrapper and preprocessing
- `test_provenance.py` — Experiment provenance recording

## Running tests

```bash
# All tests
pytest

# By category
pytest tests/algorithms/
pytest tests/papers/
pytest tests/framework/

# Specific paper
pytest tests/papers/test_dqn_2015.py
```
