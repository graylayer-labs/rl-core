# Rainbow (Hessel et al., 2018)

Paper: *Rainbow: Combining Improvements in Deep Reinforcement Learning*

## Why I chose this paper

Rainbow follows DQN because it turns several influential improvements into one
system and, crucially, studies them through ablation. My personal expectations
and questions for those ablations still need to be added from my notes.

## What the paper does

This track implements the six components reported by Hessel et al.: Double DQN,
dueling networks, prioritized replay, multi-step returns, C51, and noisy networks.

## My implementation

The five Atari games, seeds, budgets, and reference scores are frozen in
`reproduction.yaml`. Smoke and pilot presets are deliberately nonqualifying;
only the 50M-agent-step preset can contribute to the declared reproduction claim.

The runner records a manifest, final checkpoint, raw per-episode evaluation data,
result artifact, and qualification decision for every run.

## What happened

The implementation is covered by automated tests, but no experimental run has
been recorded. There is no reproduction result yet.

## What I learned

To be written from run and ablation notes, not inferred from the code.

## What comes next

Run a matched Pong pilot after DQN establishes a credible baseline, then decide
which component ablations are worth the compute.
