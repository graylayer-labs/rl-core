# rl-core

Reproducible evaluations of influential reinforcement-learning research.

`rl-core` is a research playground: state the claim being tested, run it on the
original testbed where practical, and compare it under a small shared
benchmark. Results include failures and deviations rather than only successful
runs.

Use maintained implementations such as Stable-Baselines3 when they cover the
algorithm faithfully. Custom implementations are reserved for paper-specific
gaps, controlled modifications, instrumentation, or ideas that existing
libraries do not support. The goal is to explore research, not rebuild an RL
framework.

This repository is not a shared dependency for the other Graylayer research
projects. `rl-evo-lab` and `lang-goal-rl` remain independent investigations.

## Current scope

| Algorithm | Paper reproduction | Implementation | Reproduction status |
|---|---|---:|---|
| DQN | [Human-level control through deep reinforcement learning](papers/dqn-2015/README.md) | Local paper-faithful components | Ready for pilot |
| Rainbow | [Rainbow: Combining Improvements in Deep Reinforcement Learning](papers/rainbow-2018/README.md) | Six switchable components | Ready for pilot |
| PPO | [Proximal Policy Optimization Algorithms](papers/ppo-2017/README.md) | Local clipped PPO and A2C boundary | Ready for pilot |
| RND | [Exploration by Random Network Distillation](papers/rnd-2018/README.md) | RND components; PPO/Atari integration pending | Component-ready |
| NGU | [Never Give Up](papers/ngu-2020/README.md) | Random Disco Maze mechanism track | Ready for mechanism pilot |

“Reproduced” means that a preregistered success criterion was met across the
declared seeds and compute budget. It does not mean that the paper was proved.

## Repository layout

```text
rl_core/
  algorithms/       # custom components only when maintained libraries do not fit
  buffers/          # reusable data structures
  experiments/      # repeatable run lifecycle and metrics
  nn/               # small neural-network building blocks
  reproductions/    # reproduction-spec loading and validation
  utils/            # config, device, logging, checkpointing, seeding
papers/
  dqn-2015/         # claim, protocol, deviations, and findings
benchmarks/         # fixed cross-algorithm evaluation tracks
results/            # generated summaries and leaderboard data
tests/              # unit and reproduction-spec validation
```

## Scientific workflow

Every paper reproduction must declare before results are recorded:

1. the exact claim being tested;
2. the original testbed and any unavoidable deviations;
3. environments, seeds, budgets, metrics, and baselines;
4. a measurable success criterion;
5. the implementation version used for the run.

There are two separate result tracks:

- **Paper reproduction:** did the implementation recover a central published
  claim under a comparable protocol?
- **Standard benchmark:** how does the implementation compare under this
  repository’s fixed environments, seeds, budgets, and metrics?

Scores from incompatible environment families are never collapsed into one
universal ranking. See [the methodology](docs/methodology.md) and
[results policy](results/README.md).

## Getting started

Requires Python 3.12 and Poetry.

```bash
poetry install
poetry run python -m rl_core.reproductions validate
poetry run pytest tests/ -q
```

Atari reproductions use an optional dependency group:

```bash
poetry install --with dev,atari
poetry run python -m rl_core.reproductions preflight rainbow-2018
poetry run python -m rl_core.reproductions run rainbow-2018 \
  --preset smoke --environment ALE/Pong-v5 --seed 7
```

Smoke runs validate the full pipeline but can never support a reproduction
claim. The fixed one-million-step `pilot` preset provides a meaningful
single-seed systems and learning-signal check while remaining nonqualifying.
Qualifying runs additionally require a clean Git worktree and the frozen paper
budget.

Quality checks:

```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run ty check
```

## Adding a reproduction

1. Select and pin a maintained implementation where one fits.
2. Add a thin experiment adapter; write custom algorithm code only when needed.
3. Create `papers/<paper-key>/reproduction.yaml` from the documented schema.
4. Write a concise paper page describing the claim and known deviations.
5. Add tests for the adapter, custom behavior, and specification validation.
6. Run the declared seeds without changing the success criterion.
7. Record aggregate results and uncertainty in the paper directory.
8. Add comparable runs to the appropriate benchmark track.

Start with one claim and a small environment set. New environments should test
a meaningful capability rather than expand the matrix for its own sake.

## Existing building blocks

- `rl_core.algorithms.dqn`: Q-network and DQN trainer
- `rl_core.algorithms.rainbow`: independently switchable Rainbow components
- `rl_core.algorithms.ppo`: clipped PPO, Atari actor-critic, rollout storage, and GAE
- `rl_core.algorithms.a2c`: separate matched actor-critic objective
- `rl_core.algorithms.sac`: policy, twin Q-network, and SAC trainer
- `rl_core.buffers`: standard, Atari, and prioritized n-step replay
- `rl_core.intrinsic`: RND, episodic kNN novelty, and inverse dynamics
- `rl_core.experiments`: deterministic run IDs, status, checkpoints, and resume
- `rl_core.results`: benchmark-only leaderboard generation
- `rl_core.utils`: configuration, devices, logging, and reproducible seeding
- `rl_core.environments`: lazily loaded paper-oriented environment construction
- `rl_core.integrations.NatureDQN`: Nature CNN and paper-specific DQN update
- `rl_core.experiments.artifacts`: immutable manifests, results, and qualification

These modules are available for experiments that need custom behavior. They are
not mandatory alternatives to mature external libraries. Public APIs should
stay small and understandable; shared abstractions are introduced only after
multiple experiments demonstrate the same stable need.

## Status

All five paper tracks have validated specifications, callable preflights, and
nonqualifying readiness paths. DQN, Rainbow, and PPO run on real ALE; RND has a
component readiness path before full distributed PPO integration; NGU targets
the paper's Random Disco Maze mechanism rather than the full distributed Atari
agent. No paper claim is marked as reproduced yet.

## Research arc

The planned sequence is DQN (2015), Rainbow (2018), PPO (2017), Random Network
Distillation (2018), and Never Give Up (2020). Shared artifacts, environment
construction, metrics, and benchmark tracks will carry across papers. Algorithm
interfaces will remain paper-appropriate: PPO will not be forced through a
DQN-shaped abstraction, while Rainbow can reuse DQN components for controlled
ablations and RND/NGU can add intrinsic-reward metrics.

See the [research roadmap](docs/research-roadmap.md) for the environment tracks,
pilot order, qualifying gates, and results policy.
