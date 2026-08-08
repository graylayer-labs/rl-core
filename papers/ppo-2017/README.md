# PPO 2017 reproduction track

This track implements the clipped-surrogate method from Schulman et al.
(2017) with synchronous vectorized Atari collection, GAE-Lambda, and multiple
minibatch epochs per behaviour-policy rollout.

The frozen protocol is a five-game Atari subset: Beam Rider, Breakout, Pong,
Seaquest, and Space Invaders.  Smoke and pilot presets are explicitly
nonqualifying.  A qualifying result requires five declared seeds per game and
the success criterion in `reproduction.yaml`.

`rl_core.algorithms.a2c` is intentionally separate: it provides an unclipped,
single-policy objective only and must not be reported as PPO.
