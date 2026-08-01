"""Command-line validation for reproduction specifications."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_core.reproductions import (
    SpecValidationError,
    discover_reproduction_specs,
    load_reproduction_spec,
)


def main() -> int:
    """Validate repository reproduction specifications."""
    parser = argparse.ArgumentParser(prog="python -m rl_core.reproductions")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate every papers/*/reproduction.yaml file")
    validate.add_argument("--papers-dir", type=Path, default=Path("papers"))
    args = parser.parse_args()

    if args.command != "validate":
        parser.error(f"unsupported command: {args.command}")

    paths = discover_reproduction_specs(args.papers_dir)
    if not paths:
        parser.error(f"no reproduction specifications found under {args.papers_dir}")

    failures: list[str] = []
    for path in paths:
        try:
            spec = load_reproduction_spec(path)
        except SpecValidationError as exc:
            failures.append(str(exc))
        else:
            print(f"valid: {spec.key} ({spec.status})")

    if failures:
        for failure in failures:
            print(f"invalid: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
