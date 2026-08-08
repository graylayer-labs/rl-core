"""Fast local-environment validation for the NGU readiness protocol."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from rl_core.environments.disco_maze import RandomDiscoMaze
from rl_core.reproductions import load_reproduction_spec


class NGUPreflightError(RuntimeError):
    """Raised when the frozen local NGU protocol cannot be run."""


def preflight_ngu_2020(papers_dir: Path = Path("papers")) -> dict[str, Any]:
    """Construct the exact 21x21 maze and validate its visual observation."""
    packages: dict[str, str] = {}
    for package in ("numpy", "torch", "pyyaml"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError as exc:
            raise NGUPreflightError(f"{package} is required") from exc
    spec = load_reproduction_spec(papers_dir / "ngu-2020" / "reproduction.yaml")
    game = spec.protocol.games[0]
    if game.id != "RandomDiscoMaze-21x21-v1":
        raise NGUPreflightError("unexpected local environment")
    environment = RandomDiscoMaze(seed=spec.seeds[0])
    observation, info = environment.reset(seed=spec.seeds[0])
    if observation.shape != (21, 21, 3) or observation.dtype.name != "uint8":
        raise NGUPreflightError("maze observation must be uint8 RGB 21x21")
    return {
        "packages": packages,
        "protocol_revision": spec.protocol.version,
        "environment_id": game.id,
        "observation_shape": list(observation.shape),
        "action_count": environment.action_space.n,
        "initial_coverage": int(info["coverage"]),
    }
