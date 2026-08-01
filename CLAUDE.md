# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Commands

```bash
# Install all dependencies (including dev)
poetry install

# Lint
poetry run ruff check .

# Format
poetry run ruff format .

# Type check
poetry run ty check

# Run all tests
poetry run pytest tests/ -v

# Run a single test file
poetry run pytest tests/path/to/test_file.py -v

# Run a single test by name
poetry run pytest tests/ -k "test_name" -v
```

## Repository purpose

`rl-core` is a paper-reproduction and benchmarking lab. It evaluates specific
claims from established RL research under declared protocols.

Prefer maintained implementations such as Stable-Baselines3 when they cover the
required behavior. Write custom algorithm code only for unsupported
paper-specific components, controlled modifications, instrumentation, or new
research ideas. Do not rebuild an RL framework for its own sake.

Sibling research repositories (`rl-evo-lab` and `lang-goal-rl`) are
intentionally independent rather than consumers of this package.

## Module layout

```
rl_core/
├── buffers/
│   └── replay_buffer.py      # Generic ReplayBuffer — field-spec dict, returns torch tensors
├── utils/
│   ├── seeding.py            # seed_everything(seed)
│   ├── device.py             # get_device(prefer="auto")  →  cuda > mps > cpu
│   ├── logging.py            # Logger protocol + StdoutLogger, CSVLogger, WandbLogger, CompositeLogger
│   ├── config.py             # load_config(path, DataclassType)  — YAML → frozen dataclass
│   └── checkpoint.py         # Checkpoint + save/load_checkpoint + capture/restore_rng_state
├── nn/
│   ├── mlp.py                # build_mlp(input_dim, output_dim, hidden_dims, ...)
│   └── flat_params.py        # FlatParamsMixin — get/set_flat_params for ES-compatible nets
├── experiments/
│   ├── run_manager.py        # RunManager + ExperimentRun — lifecycle, status, checkpointing, resume
│   └── metrics.py            # NamespacedLogger — routes keys to algo/ or research/ prefix
└── algorithms/
    ├── dqn/
    │   ├── network.py        # QNetwork(FlatParamsMixin) — usable as DQN learner or ES actor
    │   └── trainer.py        # DQNConfig (frozen dataclass) + DQNTrainer
    └── sac/
        ├── network.py        # GaussianPolicy (shared trunk + mean/log_std heads), TwinQNetwork
        └── trainer.py        # SACConfig (frozen dataclass) + SACTrainer

papers/                       # one declared claim and protocol per paper
benchmarks/                   # fixed cross-algorithm evaluation tracks
results/                      # aggregate, reviewable result summaries
docs/methodology.md            # scientific and reporting rules
```

## Key design decisions

**ReplayBuffer** is field-spec generic: fields are declared at construction as `dict[str, tuple[int, ...]]` rather than hard-coded per algorithm. Both `push(**kwargs)` and `sample(batch_size, device)` use the same field names.

**Logging** uses a `Logger` Protocol so any backend can be substituted or composed. `NamespacedLogger` wraps any Logger and routes declared metric keys to `algo/` (algorithm health) or `research/` (research signal) prefixes — this separation is important for diagnosing whether a base algorithm or the research contribution is failing.

**Config** uses frozen dataclasses as the canonical config type. `load_config(path, cls)` loads a YAML file and coerces it into the dataclass. All algo configs (`DQNConfig`, `SACConfig`) follow this pattern.

**Algorithm trainers** (`DQNTrainer`, `SACTrainer`) are decoupled from the training loop. They expose `select_action()`, `train_step(batch)`, `state_dicts()`, and `load_state_dicts()`. The outer loop lives in the downstream repo.

**RunManager** derives a deterministic SHA-1 run ID from the config so the same config always maps to the same directory. `is_done()` checks `status.json` for idempotency. `ExperimentRun.resume()` finds the latest checkpoint and restores full RNG state (numpy + torch) so resumed runs are numerically identical to uninterrupted ones.

**`FlatParamsMixin`** is mixed into `QNetwork` so the same class serves as a DQN learner and an ES actor population member.

**SAC** uses a shared trunk with two separate linear heads (mean, log_std) for `GaussianPolicy`. `TwinQNetwork` contains two independent MLPs. Temperature α is automatically tuned with `target_entropy = -action_dim`.

**DQN** uses soft target-network updates (Polyak averaging via `tau`) rather than periodic hard copies.

## Reproduction rules

- Say “reproduce a claim,” not “prove a paper.”
- Freeze the claim, seeds, budget, metric, and success criterion before
  qualifying results are recorded.
- Record whether an implementation is external, local, or hybrid.
- Keep original-testbed results separate from standard-benchmark results.
- Never mark a protocol `reproduced` without completing every declared seed.
- Report negative, partial, and inconclusive outcomes.
- Do not add placeholder scores to `results/leaderboard.csv`.

Validate specifications with:

```bash
poetry run python -m rl_core.reproductions validate
```

## Versioning rules

This repo uses semver with git tags. Understand these before modifying public APIs:

| Change type | Version bump | PR label |
|---|---|---|
| Bug fix, no API change | patch | `fix` |
| New feature, backwards compatible | minor | `feat` |
| Rename / remove / signature change | major | `breaking` |

**A change is breaking if it affects anything in a module's `__all__` or directly importable namespace.**

When asked to make a breaking change:
1. Label the PR `breaking`
2. List affected algorithms and reproduction protocols
3. Add a migration note to the CHANGELOG entry
4. Bump major version in `pyproject.toml`

## Release process

When asked to cut a release:
1. Confirm all tests pass: `poetry run pytest tests/ -v`
2. Update `CHANGELOG.md` — add entry above the previous release
3. Bump `version` in `pyproject.toml`
4. Commit: `git commit -m "Release vX.Y.Z"`
5. Tag: `git tag vX.Y.Z`
6. Push: `git push origin main --tags`

Use the `/release` slash command to be walked through this interactively.

## New algorithm checklist

New algorithms go in `rl_core/algorithms/<name>/` and must:
- Have a frozen dataclass config (e.g. `MyAlgoConfig`)
- Expose `select_action()`, `train_step(batch)`, `state_dicts()`, `load_state_dicts()`
- Return metric keys from `train_step` using `loss/`, `q/`, `entropy/` prefixes
- Have tests in `tests/algorithms/test_myalgo.py`
- Be exported from `rl_core/algorithms/<name>/__init__.py`
- Have a paper protocol under `papers/<paper-key>/reproduction.yaml`
- State one measurable claim and any deviations before qualifying runs

## What NOT to do

- Do not modify existing public function signatures without a breaking PR and major version bump
- Do not push directly to `main` — use a branch and PR
- Do not merge without all three checks passing (ruff, ty, pytest)
- Do not add dependencies to `[project]` until a runnable experiment needs them
- Do not write tests that depend on wandb being installed — mock it or skip
- Do not tune success criteria after inspecting qualifying results
- Do not combine unrelated environment families into a universal score
