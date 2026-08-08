"""Small dispatch registry for paper-shaped reproduction workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PreflightFunction = Callable[[], dict[str, Any]]
RunFunction = Callable[[str, int, str, Path, str | None], Path]


@dataclass(frozen=True)
class ReproductionWorkflow:
    """CLI-facing operations without imposing a learner interface."""

    key: str
    preflight: PreflightFunction
    run: RunFunction
    variants: tuple[str, ...] = ()


def _dqn() -> ReproductionWorkflow:
    from rl_core.reproductions.dqn_2015.preflight import preflight_dqn_2015
    from rl_core.reproductions.dqn_2015.runner import run_dqn_2015

    return ReproductionWorkflow(
        "dqn-2015",
        preflight_dqn_2015,
        lambda environment, seed, preset, runs_dir, _variant: run_dqn_2015(environment, seed, preset, runs_dir),
    )


def _rainbow() -> ReproductionWorkflow:
    from rl_core.reproductions.rainbow_2018 import preflight_rainbow_2018, run_rainbow_2018

    return ReproductionWorkflow(
        "rainbow-2018",
        preflight_rainbow_2018,
        lambda environment, seed, preset, runs_dir, _variant: run_rainbow_2018(environment, seed, preset, runs_dir),
    )


def _ppo() -> ReproductionWorkflow:
    from rl_core.reproductions.ppo_2017 import preflight_ppo_2017, run_ppo_2017

    return ReproductionWorkflow(
        "ppo-2017",
        preflight_ppo_2017,
        lambda environment, seed, preset, runs_dir, _variant: run_ppo_2017(environment, seed, preset, runs_dir),
    )


def _rnd() -> ReproductionWorkflow:
    from rl_core.reproductions.rnd_2018 import preflight_rnd_2018, run_rnd_2018

    return ReproductionWorkflow(
        "rnd-2018",
        preflight_rnd_2018,
        lambda environment, seed, preset, runs_dir, _variant: run_rnd_2018(environment, seed, preset, runs_dir),
    )


def _ngu() -> ReproductionWorkflow:
    from rl_core.reproductions.ngu_2020 import preflight_ngu_2020, run_ngu_2020

    return ReproductionWorkflow(
        "ngu-2020",
        preflight_ngu_2020,
        lambda environment, seed, preset, runs_dir, variant: run_ngu_2020(
            environment,
            seed,
            preset,
            variant or "rnd",
            runs_dir,
        ),
        variants=("random", "rnd"),
    )


_LOADERS = {
    "dqn-2015": _dqn,
    "rainbow-2018": _rainbow,
    "ppo-2017": _ppo,
    "rnd-2018": _rnd,
    "ngu-2020": _ngu,
}


def reproduction_keys() -> tuple[str, ...]:
    """Return registered workflows in research-roadmap order."""
    return tuple(_LOADERS)


def get_reproduction_workflow(key: str) -> ReproductionWorkflow:
    """Load one workflow lazily so optional dependencies remain optional."""
    try:
        return _LOADERS[key]()
    except KeyError as exc:
        raise KeyError(f"unknown reproduction {key!r}; choose from {', '.join(_LOADERS)}") from exc
