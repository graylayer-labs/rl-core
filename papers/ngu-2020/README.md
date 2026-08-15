# Never Give Up (Badia et al., 2020)

Paper: *Never Give Up: Learning Directed Exploration Strategies*

## Why I chose this paper

NGU brings episodic and lifelong novelty together, making it the closest paper
in this sequence to the exploration questions motivating the wider project. My
personal thesis connection still needs to be written in my own words.

## What the paper does

NGU combines short-term episodic novelty with a longer-term novelty signal so
that an agent can revisit useful behavior across training without repeatedly
mistaking the same states for new ones within an episode.

## My implementation

The local Random Disco Maze gives a fast visual controllability diagnostic. It
logs coverage for both declared lifelong-novelty variants (`random` and `rnd`).
It is deliberately separate from any claim to reproduce NGU's Atari results.

## What happened

The mechanism implementation passes automated tests, but the paired Disco Maze
study has not been run. No empirical conclusion is claimed.

## What I learned

To be written from the paired coverage runs and their novelty traces.

## What comes next

Run both lifelong-novelty variants over the declared seeds and compare physical
position coverage before considering a larger environment.
