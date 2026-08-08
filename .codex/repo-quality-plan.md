# Repository quality plan

## Baseline

The repository has sound Python tooling and reusable experiment components. It
now has a scientific protocol layer but no completed reproduction.

## Immediate

- [x] Reposition the repository around claim reproduction and benchmarking.
- [x] Prefer maintained implementations and document when custom code is justified.
- [x] Add validated reproduction metadata and an honest empty result structure.
- [x] Define the first small standard benchmark.
- [x] Align public GitHub metadata and repository settings.
- [x] Freeze DQN protocol revision 1 before implementation:
      - declared Atari game subset and justification;
      - exact training-step/emulator-frame accounting and action repeat;
      - preprocessing, reward clipping, life-loss, and reset semantics;
      - evaluation starts, duration, epsilon, checkpoint selection, and episodes;
      - paper, random, and human reference scores plus aggregation criterion.
- [x] Select and pin a compatible Gymnasium, ALE, OpenCV, and bundled-ROM stack
      in an optional DQN/Atari dependency group.
- [x] Add a preflight command that verifies Atari environments and ROM access
      without beginning a costly training run.
- [x] Add a thin runner that records config, dependency versions, commit,
      hardware, seeds, environment steps, emulator frames, protocol revision,
      evaluation settings, and wall-clock time.
- [x] Define and validate the immutable run/result manifest consumed by findings
      and leaderboard generation.

## Then

- [x] Add a cheap CPU/MPS-capable DQN smoke preset that validates environment setup,
      checkpoint/resume, evaluation, and the full artifact pipeline without
      being treated as reproduction evidence.
- [x] Add a manual CI integration workflow and complete the smoke locally.
- [x] Add direct behavioral tests for the paper-specific Nature DQN components.
- [x] Add machine-readable published reference scores and cross-seed aggregation.
- [ ] Run a one-seed reduced-budget pilot before qualifying compute.
- [ ] Add direct behavioral tests for the older local DQN and SAC trainers.
- [ ] Generate leaderboard rows from validated artifacts rather than manual entry.
- [ ] Complete qualifying DQN runs and publish findings, including negative outcomes.

## Five-paper arc

- [x] Add Rainbow components, protocol, preflight, aggregation, and real-ALE smoke.
- [x] Add clipped PPO, GAE, vector collection, A2C boundary, protocol, and real-ALE smoke.
- [x] Add RND components, normalization, protocol, component smoke, and aggregation.
- [x] Add Random Disco Maze, NGU novelty components, protocol, smoke, and coverage aggregation.
- [x] Add shared workflow registry and benchmark-only leaderboard generation.
- [ ] Run the DQN one-million-step pilot.
- [ ] Run the Rainbow one-million-step pilot.
- [ ] Run the PPO pilot and compare learning/throughput against the matched A2C objective.
- [ ] Connect RND to the PPO/ALE learner before any qualifying Montezuma run.
- [ ] Run the preregistered NGU mechanism variants across all five seeds.

## Optional later

- [ ] Add Double DQN as a controlled extension on the same benchmark.
- [ ] Add a continuous-control benchmark only when SAC or PPO work begins.
- [ ] Introduce a shared adapter protocol after two external implementations expose
      a genuinely common boundary.

## Good enough

A new reader can identify the claim, backend, protocol, deviations, commands,
and evidence for each reproduction. Results are reproducible from pinned
configuration, and no leaderboard mixes incomparable tasks or incomplete seeds.
For DQN specifically, a fresh checkout can pass preflight and complete a cheap
smoke run before any qualifying Atari compute is scheduled.
