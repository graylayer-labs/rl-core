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
- [ ] Select and pin a Stable-Baselines3 version for the first runnable experiment.
- [ ] Add a thin adapter that records config, dependency versions, commit, hardware,
      seeds, environment steps, and wall-clock time.
- [ ] Freeze the DQN Atari subset, preprocessing, and success criterion.

## Then

- [ ] Run a low-budget DQN smoke experiment to validate the full artifact pipeline.
- [ ] Add direct behavioral tests for the existing local DQN and SAC trainers.
- [ ] Generate leaderboard rows from validated artifacts rather than manual entry.
- [ ] Complete qualifying DQN runs and publish findings, including negative outcomes.

## Optional later

- [ ] Add Double DQN as a controlled extension on the same benchmark.
- [ ] Add a continuous-control benchmark only when SAC or PPO work begins.
- [ ] Introduce a shared adapter protocol after two external implementations expose
      a genuinely common boundary.

## Good enough

A new reader can identify the claim, backend, protocol, deviations, commands,
and evidence for each reproduction. Results are reproducible from pinned
configuration, and no leaderboard mixes incomparable tasks or incomplete seeds.
