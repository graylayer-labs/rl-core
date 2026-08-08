from __future__ import annotations

from pathlib import Path

from rl_core.experiments.provenance import collect_provenance


def test_collect_provenance_records_requested_package_and_no_environment(tmp_path: Path) -> None:
    provenance = collect_provenance(tmp_path, packages=["definitely-not-installed"])
    assert provenance["packages"] == {"definitely-not-installed": None}
    assert provenance["git"] == {"commit": None, "dirty": None, "diff_sha256": None}
    assert "environment" not in provenance
    assert "cwd" not in provenance


def test_collect_provenance_has_runtime_basics() -> None:
    provenance = collect_provenance(packages=["pytest"])
    assert provenance["schema_version"] == 1
    assert provenance["python"]["version"]
    assert provenance["platform"]["system"]
    assert "pytest" in provenance["packages"]
