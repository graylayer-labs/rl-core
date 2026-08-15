# DQN (Mnih et al., 2015)

Paper: *Human-level control through deep reinforcement learning*

## Why I chose this paper

This is the starting point of the research arc. My personal motivation and the
questions I brought to the paper still need to be written from my own notes;
they will not be inferred from the implementation.

## What the paper does

DQN showed that a neural network could learn action values directly from Atari
pixels across many games using one general training recipe. Experience replay,
a separately updated target network, and carefully processed observations made
that learning stable enough to work.

## Claim under test

An agent trained directly from visual observations with experience replay and a
target network can reach human-comparable performance across a representative
subset of Atari 2600 games.

The first qualifying reproduction will use a declared Atari subset rather than
silently generalizing from Classic Control. Classic Control remains a smoke and
regression benchmark only.

## My implementation

**Planned.** A compact paper-specific implementation provides the replay,
optimizer, and target-update semantics that maintained DQN libraries do not
expose. The Atari dependency group and protocol are pinned, but qualifying runs
do not yet exist. No reproduction claim is currently made.

## The reproduction experiment

The qualifying subset is **Beam Rider, Breakout, Pong, Seaquest, and Space
Invaders** (`ALE/*-v5`). These are the five games on which the paper reports
informally selecting its shared hyperparameters. Every game uses the same five
seeds declared in `reproduction.yaml`; game and seed selection are fixed before
any qualifying run.

- The paper's 50 million action-selection steps per seed/game; with a frame
  skip of four this is 200 million raw ALE frames. The adapter must record both
  counters.
- Grayscale 84×84 observations, four stacked frames, deterministic ALE actions,
  up to 30 no-op reset actions, minimal action set, and learner episode
  boundaries on life loss during training.
- The DQN settings are fixed in each game config: replay capacity 1M, discount
  0.99, RMSProp learning rate 0.00025, batch size 32, updates every four agent
  steps, target update every 40,000 agent steps, and rewards clipped to [-1, 1].
- Evaluation uses epsilon 0.05 for 30 episodes of up to 18,000 raw frames each
  (five minutes at 60 Hz), and the same randomized reset protocol.

Install the optional runtime with `poetry install --with atari`. The adapter
must record the installed package versions and ALE ROM identifier in each run
artifact.

After preflight and smoke, begin with the fixed nonqualifying pilot:

```bash
poetry run python -m rl_core.reproductions run dqn-2015 \
  --preset pilot --environment ALE/Pong-v5 --seed 7
```

The pilot runs one million agent steps. It is intended to expose learning,
throughput, memory, and artifact problems before any 50-million-step qualifying
run is scheduled.

## Planned evidence

- frozen ALE environment configuration and per-game configs;
- five declared seeds;
- fixed environment-frame budget;
- human-normalized score where reference scores are available;
- aggregate results with uncertainty and full deviation notes.

See `reproduction.yaml` for the machine-validated protocol and `findings.md` for
the eventual outcome.

## What I learned

This section will be written from the completed pilot and qualifying-run notes.
The current smoke run establishes engineering behavior, not a scientific
conclusion.

## What comes next

First, establish a credible learning signal in the fixed Pong pilot. Only then
does it make sense to commit qualifying compute or compare DQN with Rainbow.
