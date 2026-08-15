# PPO (Schulman et al., 2017)

Paper: *Proximal Policy Optimization Algorithms*

## Why I chose this paper

PPO broadens the project from value-based learning into policy optimization and
creates the learner needed by later intrinsic-motivation work. My own reasons
and initial questions still need to be added from my notes.

## What the paper does

This track implements the clipped-surrogate method from Schulman et al.
(2017) with synchronous vectorized Atari collection, GAE-Lambda, and multiple
minibatch epochs per behaviour-policy rollout.

## The reproduction experiment

The frozen protocol is a five-game Atari subset: Beam Rider, Breakout, Pong,
Seaquest, and Space Invaders.  Smoke and pilot presets are explicitly
nonqualifying.  A qualifying result requires five declared seeds per game and
the success criterion in `reproduction.yaml`.

## My implementation

`rl_core.algorithms.a2c` is intentionally separate: it provides an unclipped,
single-policy objective only and must not be reported as PPO.

## What happened

The implementation passes automated tests, but no experimental run has been
recorded. There is no reproduction result yet.

## What I learned

To be written from rollout, optimization, and evaluation notes.

## What comes next

Run the fixed Pong pilot and compare its learning behavior and throughput with
the matched value-based baselines.
