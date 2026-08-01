# DQN (Mnih et al., 2015)

Paper: *Human-level control through deep reinforcement learning*

## Claim under test

An agent trained directly from visual observations with experience replay and a
target network can reach human-comparable performance across a representative
subset of Atari 2600 games.

The first qualifying reproduction will use a declared Atari subset rather than
silently generalizing from Classic Control. Classic Control remains a smoke and
regression benchmark only.

## Current status

**Planned.** Stable-Baselines3 DQN is the intended maintained backend. The
existing compact local trainer is useful for focused experiments, but it will
not be expanded into a parallel Atari framework. The required Atari adapter,
paper-comparable preprocessing, and qualifying runs do not yet exist. No
reproduction claim is currently made.

## Planned evidence

- deterministic ALE environment configuration;
- paper-comparable preprocessing and action repeat;
- five declared seeds;
- fixed environment-frame budget;
- human-normalized score where reference scores are available;
- aggregate results with uncertainty and full deviation notes.

See `reproduction.yaml` for the machine-validated protocol and `findings.md` for
the eventual outcome.
