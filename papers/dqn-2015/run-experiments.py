#!/usr/bin/env python3
"""Generate and manage DQN 2015 reproduction experiments.

Usage:
  python run-experiments.py list          # Show all 25 configs
  python run-experiments.py generate      # Generate all run commands
  python run-experiments.py launch N      # Launch first N runs sequentially
  python run-experiments.py status        # Check completion status
"""

import subprocess
import sys
from pathlib import Path
from typing import Generator

GAMES = [
    "ALE/BeamRider-v5",
    "ALE/Breakout-v5",
    "ALE/Pong-v5",
    "ALE/Seaquest-v5",
    "ALE/SpaceInvaders-v5",
]

SEEDS = [7, 42, 123, 2026, 9001]

AGENT_STEPS = 50_000_000


def configs() -> Generator[tuple[str, int], None, None]:
    """Generate all 25 (game, seed) configurations."""
    for game in GAMES:
        for seed in SEEDS:
            yield game, seed


def run_command(game: str, seed: int) -> str:
    """Generate the uv run command for one config."""
    return (
        f"uv run python -m rl_core.reproductions run dqn-2015 "
        f"--agent-steps {AGENT_STEPS} "
        f"--environment {game} "
        f"--seed {seed}"
    )


def list_configs() -> None:
    """Print all 25 configs in a table."""
    print("DQN 2015 Reproduction Matrix (50M steps, 5 seeds each)")
    print("=" * 70)
    idx = 1
    for game in GAMES:
        game_name = game.split("/")[1].split("-")[0]
        print(f"\n{game_name}:")
        for seed in SEEDS:
            print(f"  Run {idx:2d}: seed {seed}")
            idx += 1
    print(f"\nTotal: {len(GAMES) * len(SEEDS)} runs")
    print(f"Expected wall time (sequential): ~{73 * 25 / 24:.0f} days")
    print(f"Expected wall time (1 per night): ~{25 * 3:.0f} days")


def generate_commands() -> None:
    """Print all 25 uv commands (one per line, ready for copy/paste)."""
    for game, seed in configs():
        print(run_command(game, seed))


def launch_runs(count: int) -> None:
    """Launch first N runs sequentially."""
    runs = list(configs())
    if count > len(runs):
        print(f"Error: only {len(runs)} runs available, requested {count}")
        sys.exit(1)

    print(f"Launching {count} runs sequentially...")
    print("Each run takes ~73 hours.")
    print()

    for i, (game, seed) in enumerate(runs[:count], 1):
        print(f"[{i}/{count}] {game} seed {seed}")
        cmd = run_command(game, seed)
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"Run {i} failed with exit code {result.returncode}")
            print("Stopping batch.")
            sys.exit(1)
        print()


def check_status() -> None:
    """Check which runs have completed."""
    runs_dir = Path("runs/dqn-2015")
    if not runs_dir.exists():
        print("No runs directory found yet.")
        return

    completed = 0
    total = len(GAMES) * len(SEEDS)

    for game, seed in configs():
        game_name = game.split("/")[1].split("-")[0]
        # Find any run directory matching this game/seed
        matches = list(runs_dir.glob(f"*__{game_name.lower()}-v5__seed{seed}__*"))
        if matches and (matches[0] / "result.json").exists():
            completed += 1
            status = "✓"
        else:
            status = " "
        print(f"[{status}] {game_name:12} seed {seed}")

    print()
    print(f"Completed: {completed}/{total}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    if command == "list":
        list_configs()
    elif command == "generate":
        generate_commands()
    elif command == "launch":
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        launch_runs(count)
    elif command == "status":
        check_status()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
