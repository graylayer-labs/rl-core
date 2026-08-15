# Random Network Distillation (Burda et al., 2018)

Paper: *Exploration by Random Network Distillation*

## Why I chose this paper

RND is where the project turns directly toward intrinsic motivation: novelty is
represented by prediction error rather than external reward. My personal
motivation and expectations still need to be written from my notes.

## What the paper does

RND trains a predictor to match the output of a fixed random network. Large
prediction error acts as an intrinsic reward for observations the agent has not
learned to predict well.

## My implementation

This track freezes a Montezuma's Revenge protocol and exercises the RND target,
predictor, and reward normalization components. Its smoke and pilot runners are
synthetic component checks, not Atari/PPO reproduction runs; their artifacts
carry that boundary explicitly.

## What happened

The components have synthetic checks, but the complete PPO/ALE learner is not
connected. This is not yet an Atari reproduction.

## What I learned

To be written after the integrated learner produces real novelty and exploration
traces.

## What comes next

Connect RND to the PPO/ALE learner, then run a nonqualifying Montezuma pilot that
records return, room coverage, predictor loss, and intrinsic reward statistics.
