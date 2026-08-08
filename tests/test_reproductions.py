from pathlib import Path

import pytest
import yaml

from rl_core.reproductions import (
    SpecValidationError,
    discover_reproduction_specs,
    load_reproduction_spec,
)


def test_repository_reproduction_specs_are_valid() -> None:
    paths = discover_reproduction_specs()
    assert paths

    specs = [load_reproduction_spec(path) for path in paths]

    assert len({spec.key for spec in specs}) == len(specs)
    assert all(spec.schema_version == 1 for spec in specs)


def test_dqn_protocol_is_frozen_to_five_games() -> None:
    spec = load_reproduction_spec("papers/dqn-2015/reproduction.yaml")

    assert spec.protocol.version == 1
    assert [game.id for game in spec.protocol.games] == [
        "ALE/BeamRider-v5",
        "ALE/Breakout-v5",
        "ALE/Pong-v5",
        "ALE/Seaquest-v5",
        "ALE/SpaceInvaders-v5",
    ]
    assert spec.protocol.frame_skip == 4
    assert spec.protocol.no_op_max == 30
    assert spec.protocol.evaluation_epsilon == 0.05
    assert spec.budget.unit == "environment_steps"
    assert spec.budget.value == 50_000_000
    assert spec.protocol.games[0].paper_score == pytest.approx(6846.0)

    for game in spec.protocol.games:
        config_path = Path("papers/dqn-2015") / game.config
        config = yaml.safe_load(config_path.read_text())
        assert config["environment_id"] == game.id
        assert config["protocol_version"] == spec.protocol.version
        assert config["training_agent_steps"] == spec.budget.value
        assert config["training_raw_ale_frames"] == 200_000_000
        assert config["training"]["rmsprop_momentum"] == pytest.approx(0.95)


def test_load_reproduction_spec_rejects_duplicate_seeds(tmp_path: Path) -> None:
    path = tmp_path / "reproduction.yaml"
    path.write_text(
        """
schema_version: 1
key: example
paper:
  title: Example Paper
  authors: [A. Researcher]
  year: 2026
  url: https://example.com/paper
algorithm: example
implementation: rl_core.algorithms.example
implementation_kind: local
claim: A measurable claim.
original_testbed: Example-v0
status: planned
seeds: [7, 7]
budget:
  unit: environment_steps
  value: 1000
primary_metric: mean_return
success_criterion: Mean return exceeds 1.
deviations: [None.]
protocol:
  version: 1
  games:
    - id: Example-v0
      config: configs/example.yaml
      random_score: 0
      human_score: 1
      paper_score: 0.5
  frame_skip: 1
  no_op_max: 0
  evaluation_epsilon: 0.0
""".strip()
    )

    with pytest.raises(SpecValidationError, match="seeds must be unique"):
        load_reproduction_spec(path)


def test_load_reproduction_spec_rejects_duplicate_games(tmp_path: Path) -> None:
    path = tmp_path / "reproduction.yaml"
    path.write_text(
        """
schema_version: 1
key: example
paper: {title: Example Paper, authors: [A. Researcher], year: 2026, url: https://example.com/paper}
algorithm: example
implementation: rl_core.algorithms.example
implementation_kind: local
claim: A measurable claim.
original_testbed: Example-v0
status: planned
seeds: [7]
budget: {unit: environment_steps, value: 1000}
primary_metric: mean_return
success_criterion: Mean return exceeds 1.
deviations: [None.]
protocol:
  version: 1
  games:
    - {id: Example-v0, config: configs/a.yaml, random_score: 0, human_score: 1, paper_score: 0.5}
    - {id: Example-v0, config: configs/b.yaml, random_score: 0, human_score: 1, paper_score: 0.5}
  frame_skip: 1
  no_op_max: 0
  evaluation_epsilon: 0.0
""".strip()
    )

    with pytest.raises(SpecValidationError, match="game ids must be unique"):
        load_reproduction_spec(path)


def test_discover_reproduction_specs_is_sorted(tmp_path: Path) -> None:
    for key in ("z-paper", "a-paper"):
        directory = tmp_path / key
        directory.mkdir()
        (directory / "reproduction.yaml").touch()

    paths = discover_reproduction_specs(tmp_path)

    assert [path.parent.name for path in paths] == ["a-paper", "z-paper"]
