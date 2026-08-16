#!/usr/bin/env python3
"""Compare MPS vs CPU performance on 1M steps.

Usage:
  python device-comparison.py mps    # Run 1M on MPS
  python device-comparison.py cpu    # Run 1M on CPU
  python device-comparison.py both   # Run both sequentially
"""

import subprocess
import sys
from pathlib import Path

AGENT_STEPS = 1_000_000
GAME = "ALE/BeamRider-v5"
SEED = 7


def run_with_device(device: str) -> Path:
    """Run 1M steps and force a specific device."""
    print(f"\n{'='*70}")
    print(f"Starting 1M step run on {device.upper()}")
    print(f"Game: {GAME}, Seed: {SEED}")
    print(f"Expected time: ~30-50 min on {device.upper()}")
    print(f"{'='*70}\n")

    cmd = (
        f"RLCORE_DEVICE={device} "
        f"uv run python -m rl_core.reproductions run dqn-2015 "
        f"--agent-steps {AGENT_STEPS} "
        f"--environment {GAME} "
        f"--seed {SEED}"
    )

    result = subprocess.run(cmd, shell=True)

    if result.returncode == 0:
        print(f"\n✓ {device.upper()} run completed successfully")
    else:
        print(f"\n✗ {device.upper()} run failed with exit code {result.returncode}")

    return Path("runs/dqn-2015")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "mps":
        run_with_device("mps")
    elif mode == "cpu":
        run_with_device("cpu")
    elif mode == "both":
        print("Running 1M on MPS first, then CPU")
        run_with_device("mps")
        run_with_device("cpu")
        print("\n" + "="*70)
        print("Both runs completed. Check runs/dqn-2015/ for results.")
        print("Compare metrics.csv files for throughput comparison.")
        print("="*70)
    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
