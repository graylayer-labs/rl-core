from __future__ import annotations

import pytest

from rl_core.reproductions.registry import get_reproduction_workflow, reproduction_keys


def test_registry_preserves_research_arc_order() -> None:
    assert reproduction_keys() == (
        "dqn-2015",
        "rainbow-2018",
        "ppo-2017",
        "rnd-2018",
        "ngu-2020",
    )


def test_registry_loads_paper_shaped_workflows_lazily() -> None:
    assert get_reproduction_workflow("dqn-2015").variants == ()
    assert get_reproduction_workflow("ngu-2020").variants == ("random", "rnd")


def test_registry_rejects_unknown_paper() -> None:
    with pytest.raises(KeyError, match="unknown reproduction"):
        get_reproduction_workflow("unknown")
