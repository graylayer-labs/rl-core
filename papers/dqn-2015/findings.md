# DQN reproduction findings

## Status

Qualifying runs (5 games × 5 seeds × 50M steps) not yet started.

## Engineering evidence

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

**Machinery verdict:** Pipeline executes end-to-end without error.
