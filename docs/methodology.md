# Reproduction methodology

## What this repository tests

A reproduction targets a specific, measurable claim from a paper. It does not
attempt to validate every conclusion in the paper, and a successful run does
not prove the algorithm generally correct or superior.

Some papers make full-system claims whose original compute or distributed
infrastructure is impractical for the initial track. A declared
`mechanism_replication` may test an experiment from the paper at smaller scale,
but it must never be reported as reproducing the full system. NGU's Random
Disco Maze track follows this rule.

Each reproduction is evaluated in two contexts:

1. **Original-testbed track:** match the paper's environment, preprocessing,
   budget, and evaluation protocol as closely as practical.
2. **Standard-benchmark track:** use the fixed protocol declared under
   `benchmarks/` so implementations can be compared consistently.

Results from the two tracks must remain separate.

## Implementation policy

Prefer maintained, widely used implementations when they expose the algorithm
and controls required by the protocol. Stable-Baselines3 is the default first
candidate for standard deep-RL algorithms.

Write custom code only when:

- the paper component is unavailable or materially different;
- the experiment requires controlled access or instrumentation;
- the research question modifies the established algorithm; or
- a compact independent implementation is necessary to diagnose behavior.

Record the library, version, policy class, and relevant deviations. A library
implementation is not automatically paper-identical: defaults, preprocessing,
network architecture, and training schedules must still be compared with the
source protocol.

## Required protocol

Before running an experiment, `reproduction.yaml` must declare:

- the paper and algorithm;
- one central claim;
- the original testbed;
- the implementation entry point;
- whether the implementation is external, local, or hybrid;
- seeds and training budget;
- primary metric and success criterion;
- known deviations;
- current status.

Status values have precise meanings:

- `planned`: protocol exists, but qualifying runs are incomplete;
- `running`: qualifying runs are in progress;
- `partial`: some evidence supports the claim, but the criterion was not met;
- `reproduced`: the preregistered criterion was met;
- `not_reproduced`: completed qualifying runs did not meet the criterion.

Changing the claim, budget, or success criterion after seeing results creates a
new protocol revision. The old result should remain available for context.

## Reporting

Report all declared seeds. Summaries should include the central tendency,
dispersion or confidence interval, environment steps, wall-clock time, hardware,
dependency lockfile, and Git commit.

Each executable run writes an immutable manifest before training and an
immutable result after completion. The manifest freezes the resolved
environment and algorithm settings, protocol revision, seed, dependency
versions, ROM identity, device, Git state, and required counters. Smoke presets
exercise the same artifact pipeline but are permanently nonqualifying.

Negative and inconclusive outcomes are first-class findings. Exclude a run only
for a documented operational failure, never because its score is inconvenient.

Qualifying runs must use a clean Git worktree, a declared game and seed, the
qualifying preset, the complete budget, final-checkpoint selection, and every
declared evaluation episode. Revision 1 does not permit resumed DQN runs to
qualify because ALE and wrapper state cannot yet be restored bit-for-bit.

## Benchmark growth

Benchmark tracks stay deliberately small. Add an environment when it tests a
capability not already represented, such as sparse reward, long-horizon credit
assignment, pixel observations, or continuous control.

Do not produce a single aggregate score across unrelated environment families.
