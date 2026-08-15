# RL Core

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

## Where the work stands

| Project | What exists | Evidence so far | Reproduction status |
|---|---|---|---|
| [DQN (2015)](papers/dqn-2015/README.md) | Paper-aligned Atari implementation and frozen protocol | ALE preflight and Pong smoke run | No qualifying results |
| [Rainbow (2018)](papers/rainbow-2018/README.md) | Six-component implementation and Atari runner | Automated tests | Not run |
| [PPO (2017)](papers/ppo-2017/README.md) | Clipped PPO with vectorized Atari collection | Automated tests | Not run |
| [RND (2018)](papers/rnd-2018/README.md) | Intrinsic-reward components and protocol | Synthetic component checks | Atari learner incomplete |
| [NGU (2020)](papers/ngu-2020/README.md) | Episodic/lifelong novelty mechanism study | Automated tests | Disco Maze study not run |

**Implemented** means the code exists. **Validated** means the experiment
machinery has been exercised. **Reproduced** is reserved for a completed
protocol whose declared success criterion was met.

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
