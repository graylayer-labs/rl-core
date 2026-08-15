# RL Core

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/packaging-poetry-lightblue.svg)](https://python-poetry.org/)
[![PyTorch](https://img.shields.io/badge/deep--learning-pytorch-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Do influential reinforcement-learning results survive contact with an
independent implementation?**

RL Core is my public research notebook for finding out. I am reproducing a
sequence of important RL papers, beginning with DQN and moving toward the
exploration and novelty methods that interest me most.

The aim is not to collect implementations that merely run. Each project starts
with a claim from a paper, turns it into an explicit experiment, and records
what the evidence actually supports—including failures and scaled-down
deviations.

## The research arc

```text
DQN → Rainbow → PPO → RND → NGU
value learning            exploration → novelty
```

This sequence builds the experimental foundation I need before investigating
my own questions about agents that explore when rewards are sparse, deceptive,
or absent.

## Paper Projects

Each paper has its own research journal documenting the learning process:

- **[DQN (Mnih et al., 2015)](papers/dqn-2015/)** — Starting point: value learning from pixels
- **[Rainbow (Hessel et al., 2018)](papers/rainbow-2018/)** — Six improvements combined and ablated
- **[PPO (Schulman et al., 2017)](papers/ppo-2017/)** — Policy gradient alternative to value learning
- **[RND (Burda et al., 2018)](papers/rnd-2018/)** — Intrinsic motivation via prediction error
- **[NGU (Badia et al., 2020)](papers/ngu-2020/)** — Episodic and lifelong novelty combined

Each paper's README documents: why it was chosen, what the paper does, implementation details, reproduction results (as they complete), and what was learned.

## Start with DQN

Python 3.12 and Poetry are required. Atari experiments also require compatible
ALE ROMs.

```bash
poetry install --with dev,atari
poetry run python -m rl_core.reproductions preflight dqn-2015
poetry run python -m rl_core.reproductions run dqn-2015 \
  --preset smoke --environment ALE/Pong-v5 --seed 7
```

The smoke preset checks the machinery; it does not count as reproduction
evidence. The [DQN project](papers/dqn-2015/README.md) describes the experiment
and the larger qualifying run.

## How evidence moves through the project

```text
paper claim → frozen protocol → run artifacts → findings → leaderboard
```

- `papers/` contains the story, experiment contract, and findings for each paper.
- `rl_core/` contains reusable algorithms and experiment machinery.
- `benchmarks/` defines comparisons that can legitimately share a leaderboard.
- `results/` contains compact outputs derived from completed runs.
- `tests/` protects algorithm behavior and the evidence pipeline.

The exact protocols are machine-readable. The paper pages explain why I made
the choices, what I implemented, what happened, and what I learned.

## Rules of the lab

- A successful run is not automatically a reproduction.
- Any departure from the original experiment is stated explicitly.
- Failed, partial, and ambiguous results remain part of the record.
- Scores are compared only when their protocols are compatible.

The current focus is completing the DQN reproduction before treating the later
projects as scientific results.
