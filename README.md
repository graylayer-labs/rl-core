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
| DQN | [Human-level control through deep reinforcement learning](papers/dqn-2015/README.md) | Stable-Baselines3 planned | Planned |
| SAC | Reproduction specification not yet written | Backend not selected | Not assessed |

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
- `rl_core.algorithms.sac`: policy, twin Q-network, and SAC trainer
- `rl_core.buffers`: field-based replay buffer
- `rl_core.experiments`: deterministic run IDs, status, checkpoints, and resume
- `rl_core.utils`: configuration, devices, logging, and reproducible seeding

These modules are available for experiments that need custom behavior. They are
not mandatory alternatives to mature external libraries. Public APIs should
stay small and understandable; shared abstractions are introduced only after
multiple experiments demonstrate the same stable need.

## Status

The reproduction framework and DQN protocol are now scaffolded, but no paper
claim is marked as reproduced yet. The leaderboard intentionally contains no
scores until runs satisfying the declared protocols have completed.

## License

MIT
