# DQN reproduction findings

## Status

No qualifying runs have been completed. Phase 1 (code validation) is complete.

## Engineering evidence (non-qualifying)

### Preflight (2026-08-16)
Protocol revision 1, 5 games declared (Beam Rider, Breakout, Pong, Seaquest,
Space Invaders). All dependencies present and pinned correctly:
- ale-py 0.11.2
- gymnasium 1.2.3
- opencv-python-headless 4.13.0.92

All 5 ALE ROMs verified by SHA-256. Environment observation shape validated as
(4, 84, 84) uint8 for each game.

### Smoke test: Pong, seed 7 (2026-08-16)
- Budget: 512 agent steps (2,048 raw ALE frames)
- Evaluation: 2 episodes, 100 steps each
- Mean episode return: **-1.5 ± 0.5** (better than random -20.7)
- Wall clock: 2.3 seconds
- Status: completed
- Artifacts: manifest, status, result, qualification (false), evaluations, metrics, checkpoints all present and valid
- Git commit: 63b09d3 (uv migration)

**Machinery verdict:** Pipeline executes end-to-end without error. Learning signal
is positive relative to random. The smoke run is engineering evidence only and
cannot support a reproduction claim.

## Planned next steps

1. Pilot run: Pong, seed 7, 1M agent steps (reduced budget, ~30-60 minutes)
   - Confirm sustained learning signal over 1M steps
   - Validate throughput and memory use
   
2. After pilot validation: decide whether to run full qualifying matrix (5 games × 5 seeds × 50M steps)
