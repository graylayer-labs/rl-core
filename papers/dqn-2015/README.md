# DQN (Mnih et al., 2015)

**Paper:** *Human-level control through deep reinforcement learning* (Mnih et al., Nature 2015)

**Full text:**
- [arXiv:1312.5602](https://arxiv.org/abs/1312.5602) (open access)
- Download: `curl -o papers/dqn-2015/nature14236.pdf https://arxiv.org/pdf/1312.5602.pdf`

Full re-implementation of DQN, evaluated on the paper's original Atari testbed.

## Why this paper

DQN is the first algorithm to scale deep RL to Atari from raw pixels. Every algorithm after it (Rainbow, A3C, PPO) builds on its core ideas. Reproducing it requires engagement with the engineering (experience replay, target networks, frame preprocessing) rather than theory alone.

## What the paper does

CNN + Q-learning. Takes raw Atari frames as input, outputs Q-values per action. Two core mechanisms:

1. **Experience replay**: Buffer past transitions, sample random batches during training. Breaks temporal correlations.
2. **Target network**: Separate network for bootstrap targets, updated infrequently. Stabilizes learning.

## Claim under test

DQN learns Atari policies from pixels. Performance exceeds random baseline, approaches human-level play.

## Implementation

See `rl_core.integrations.NatureDQN` for algorithm code.

## Reproduction experiment

See `reproduction.yaml` for the frozen protocol and `findings.md` for results.

## Results

## Key insights

## Next steps
