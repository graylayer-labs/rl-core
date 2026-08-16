"""Fast dependency, ROM, and protocol checks before DQN compute."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from rl_core.environments import AtariPreprocessingConfig, inspect_atari_environment, make_atari_environment
from rl_core.reproductions import load_reproduction_spec

EXPECTED_VERSIONS = {
    "ale-py": "0.11.2",
    "gymnasium": "1.2.3",
    "opencv-python-headless": "4.13.0.92",
}


class PreflightError(RuntimeError):
    """Raised when the local machine cannot run the frozen protocol."""


def _rom_digest(environment_id: str) -> str:
    import ale_py

    rom_name = environment_id.removeprefix("ALE/").removesuffix("-v5")
    snake_name = "".join(
        f"_{character.lower()}" if character.isupper() else character for character in rom_name
    ).lstrip("_")
    rom_path = Path(ale_py.__file__).parent / "roms" / f"{snake_name}.bin"
    if not rom_path.is_file():
        raise PreflightError(f"ROM file not found for {environment_id}: {rom_path.name}")
    return hashlib.sha256(rom_path.read_bytes()).hexdigest()


def preflight_dqn_2015(papers_dir: Path = Path("papers")) -> dict[str, Any]:
    """Open and step every declared game, returning reviewable facts."""
    installed: dict[str, str] = {}
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            actual = version(package)
        except PackageNotFoundError as exc:
            raise PreflightError(f"{package} is missing; run 'uv sync --group atari'") from exc
        if actual != expected:
            raise PreflightError(f"{package}=={expected} required, found {actual}")
        installed[package] = actual

    spec = load_reproduction_spec(papers_dir / "dqn-2015" / "reproduction.yaml")
    environments: list[dict[str, Any]] = []
    for game in spec.protocol.games:
        config = AtariPreprocessingConfig(env_id=game.id, seed=spec.seeds[0], terminal_on_life_loss=True)
        environment = make_atari_environment(config)
        try:
            environment.reset(seed=spec.seeds[0])
            environment.step(environment.action_space.sample())
            info = inspect_atari_environment(environment, config)
            if info.observation_shape != (4, 84, 84) or info.observation_dtype != "uint8":
                raise PreflightError(f"{game.id} produced {info.observation_shape}/{info.observation_dtype}")
            environments.append({**asdict(info), "rom_sha256": _rom_digest(game.id)})
        finally:
            environment.close()
    return {"packages": installed, "protocol_revision": spec.protocol.version, "environments": environments}
