# RL research roadmap

This roadmap turns the five-paper research arc into an ordered experimental
programme. It distinguishes engineering checks from scientific evidence: smoke
and pilot runs validate the system, while only complete qualifying runs can
support a reproduction claim.

## Environment tracks

| Track | Environments | Purpose |
|---|---|---|
| Development | `CartPole-v1`, `Acrobot-v1`, `ALE/Pong-v5` | Fast correctness, learning-signal, and throughput checks |
| Dense Atari | Beam Rider, Breakout, Pong, Seaquest, Space Invaders | Shared DQN, Rainbow, and PPO comparison |
| Sparse exploration | Montezuma's Revenge; Pitfall later | Room discovery, coverage, and long-horizon exploration |
| Mechanism studies | `RandomDiscoMaze-21x21-v1` | Isolate episodic and lifelong novelty mechanisms |

Results remain separate across these tracks. In particular, Classic Control
scores, dense Atari returns, Montezuma room coverage, and Disco Maze position
coverage are never collapsed into one universal ranking.

## Execution stages

Every paper advances through the same evidence gates:

1. **Preflight:** validate the protocol, dependencies, ROMs, wrappers, and
   configuration without training.
2. **Smoke:** exercise collection, at least one optimizer update, checkpointing,
   evaluation, and artifacts at negligible budget.
3. **Pilot:** run one declared seed at a reduced fixed budget to establish a
   learning signal, throughput, memory use, and expected qualifying cost.
4. **Qualifying runs:** complete every preregistered environment, seed, budget,
   and evaluation without changing the success criterion.
5. **Publication:** aggregate seed-level results, record uncertainty and
   deviations, update findings, and generate eligible benchmark rows.

Smoke and pilot artifacts are permanently nonqualifying. Qualifying runs require
a clean Git worktree and exact protocol/configuration digests.

## Paper sequence

### 1. DQN (Mnih et al., 2015)

- **Environments:** the five dense Atari games.
- **First pilot:** Pong, seed 7, one million selected actions.
- **Next gate:** verify sustained learning on Pong, then repeat on Breakout.
- **Qualifying criterion:** reach the paper's human-normalized score on at least
  three of the five games across all five declared seeds.

```bash
uv run python -m rl_core.reproductions preflight dqn-2015
uv run python -m rl_core.reproductions run dqn-2015 \
  --preset pilot --environment ALE/Pong-v5 --seed 7
```

### 2. Rainbow (Hessel et al., 2018)

- **Environments:** the same five Atari games, enabling paired comparison with
  DQN.
- **First pilot:** Pong, seed 7, with all six Rainbow components enabled.
- **Next gate:** compare full Rainbow against matched DQN on Pong and Breakout;
  then run controlled component ablations.
- **Qualifying criterion:** reach the declared Rainbow reference on at least
  three of the five games across all seeds.

```bash
uv run python -m rl_core.reproductions preflight rainbow-2018
uv run python -m rl_core.reproductions run rainbow-2018 \
  --preset pilot --environment ALE/Pong-v5 --seed 7
```

### 3. PPO (Schulman et al., 2017)

- **Environments:** the same five Atari games for a policy-gradient comparison.
- **First pilot:** Pong, seed 7, using vectorized clipped-PPO rollouts.
- **Next gate:** compare learning signal and throughput with the matched A2C
  objective, then repeat on Breakout.
- **Qualifying criterion:** reach the preregistered PPO reference on at least
  three of the five games across all seeds.

```bash
uv run python -m rl_core.reproductions preflight ppo-2017
uv run python -m rl_core.reproductions run ppo-2017 \
  --preset pilot --environment ALE/Pong-v5 --seed 7
```

### 4. Random Network Distillation (Burda et al., 2018)

- **Paper environment:** `ALE/MontezumaRevenge-v5`.
- **Current gate:** RND target/predictor learning and reward normalization are
  component-ready, but the complete PPO/ALE learner is not yet connected.
- **First pilot after integration:** one declared seed on Montezuma, reporting
  unique-room coverage, return, novelty statistics, and predictor loss.
- **Qualifying gate:** no qualifying command may run until PPO/ALE integration,
  ROM provenance, room tracking, and the full preregistered budget are present.

```bash
uv run python -m rl_core.reproductions preflight rnd-2018
uv run python -m rl_core.reproductions run rnd-2018 \
  --preset smoke --environment ALE/MontezumaRevenge-v5 --seed 7
```

The current RND smoke is a component check, not an Atari reproduction result.

### 5. Never Give Up (Badia et al., 2020)

- **Initial environment:** `RandomDiscoMaze-21x21-v1`.
- **Variants:** random lifelong multiplier and RND lifelong novelty, both using
  episodic k-nearest-neighbour novelty in a learned controllable embedding.
- **First pilot:** both variants on seed 7; then all five paired seeds.
- **Primary metric:** unique physical-position coverage, which prevents random
  wall colours from inflating exploration.
- **Scope boundary:** this is a mechanism replication, not the full distributed
  Atari NGU agent.

```bash
uv run python -m rl_core.reproductions preflight ngu-2020
uv run python -m rl_core.reproductions run ngu-2020 \
  --preset pilot --environment RandomDiscoMaze-21x21-v1 \
  --seed 7 --variant rnd
```

Run the matching `--variant random` pilot before comparing results.

## Immediate run order

1. DQN pilot on Pong.
2. Rainbow pilot on Pong.
3. PPO pilot on Pong.
4. Review learning curves, throughput, memory, checkpoints, and artifact
   qualification for all three.
5. Repeat viable dense-control pilots on Breakout.
6. Run both NGU mechanism variants on Disco Maze across the declared seeds.
7. Connect RND to PPO/ALE and run a nonqualifying Montezuma pilot.
8. Estimate and approve qualifying compute before launching any full matrix.

## Results and leaderboards

- Paper findings live beside each paper and include every declared seed,
  failures, deviations, wall time, hardware, and uncertainty.
- Standard benchmark results are generated only from completed artifacts marked
  with the `benchmark` track.
- The dense Atari leaderboard compares compatible DQN, Rainbow, and PPO runs.
- Sparse-exploration reporting keeps Montezuma room coverage separate from
  Disco Maze position coverage.
- Leaderboards remain empty until every required seed for a compatible track is
  complete; smoke, pilot, partial, and paper-reproduction artifacts are rejected.

## Definition of ready for qualifying compute

A paper may enter qualifying execution only when its full protocol validates,
preflight passes on the target machine, its pilot shows a credible learning
signal, artifacts validate, required aggregation is implemented, and estimated
compute has been reviewed. Status remains `planned` until all qualifying runs
finish; results determine whether it becomes `reproduced`, `partial`, or
`not_reproduced`.
