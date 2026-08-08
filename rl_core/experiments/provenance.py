"""Safe, minimal runtime provenance for research artifacts."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def _git(repo_dir: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip()
    return output or None


def collect_provenance(
    repo_dir: Path | str | None = None,
    packages: Iterable[str] = (),
) -> dict[str, Any]:
    """Capture stable execution context without collecting environment secrets.

    This intentionally never records environment variables, command-line
    arguments, usernames, absolute working directories, or unredacted diffs.
    A dirty working tree is represented only by a SHA-256 digest of its diff.
    """
    resolved_repo = Path(repo_dir).resolve() if repo_dir is not None else None
    git: dict[str, Any] = {"commit": None, "dirty": None, "diff_sha256": None}
    if resolved_repo is not None:
        commit = _git(resolved_repo, "rev-parse", "HEAD")
        diff = _git(resolved_repo, "diff", "--binary", "HEAD")
        git = {
            "commit": commit,
            "dirty": None if diff is None else bool(diff),
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
        }
    package_versions: dict[str, str | None] = {}
    for package in sorted(set(packages)):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = None
    return {
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "python": {"implementation": platform.python_implementation(), "version": sys.version.split()[0]},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "git": git,
        "packages": package_versions,
    }
