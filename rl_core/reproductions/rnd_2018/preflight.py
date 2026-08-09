"""No-ALE preflight for the explicitly component-scoped RND track."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from rl_core.reproductions import load_reproduction_spec


class RNDPreflightError(RuntimeError):
    """Raised when the local RND readiness runner cannot be executed."""


def preflight_rnd_2018(papers_dir: Path = Path("papers")) -> dict[str, Any]:
    """Validate frozen Montezuma configuration and core runtime dependencies.

    This intentionally does not import ALE: the implementation boundary is a
    cheap RND component track until the PPO/Atari integration is added.
    """
    installed: dict[str, str] = {}
    for package in ("numpy", "torch", "pyyaml"):
        try:
            installed[package] = version(package)
        except PackageNotFoundError as exc:
            raise RNDPreflightError(f"{package} is required") from exc
    spec = load_reproduction_spec(papers_dir / "rnd-2018" / "reproduction.yaml")
    game = spec.protocol.games[0]
    config_path = papers_dir / "rnd-2018" / game.config
    if game.id != "ALE/MontezumaRevenge-v5" or not config_path.is_file():
        raise RNDPreflightError("frozen Montezuma's Revenge protocol is incomplete")
    return {
        "packages": installed,
        "protocol_revision": spec.protocol.version,
        "environment_id": game.id,
        "implementation_boundary": "RND component readiness; PPO/ALE training unavailable",
    }
