# Rainbow (2018) readiness track

This track implements the six components reported by Hessel et al.: Double DQN,
dueling networks, prioritized replay, multi-step returns, C51, and noisy networks.
The five Atari games, seeds, budgets, and reference scores are frozen in
`reproduction.yaml`. Smoke and pilot presets are deliberately nonqualifying;
only the 50M-agent-step preset can contribute to the declared reproduction claim.

The runner records a manifest, final checkpoint, raw per-episode evaluation data,
result artifact, and qualification decision for every run.
