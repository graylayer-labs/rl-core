"""Command-line entry point for paper reproductions."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_core.reproductions import (
    SpecValidationError,
    discover_reproduction_specs,
    load_reproduction_spec,
)
from rl_core.reproductions.registry import get_reproduction_workflow, reproduction_keys


def main() -> int:
    """Validate, preflight, or run a paper reproduction."""
    parser = argparse.ArgumentParser(prog="python -m rl_core.reproductions")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate every papers/*/reproduction.yaml file")
    validate.add_argument("--papers-dir", type=Path, default=Path("papers"))
    keys = reproduction_keys()
    preflight = subparsers.add_parser("preflight", help="verify dependencies, environments, and protocol")
    preflight.add_argument("key", choices=keys)
    run = subparsers.add_parser("run", help="run one declared paper experiment")
    run.add_argument("key", choices=keys)
    run.add_argument("--preset", choices=("smoke", "pilot", "qualifying"), required=True)
    run.add_argument("--environment", required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--variant")
    run.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    if args.command == "preflight":
        workflow = get_reproduction_workflow(args.key)
        report = workflow.preflight()
        print(f"ready: {args.key} protocol revision {report['protocol_revision']}")
        for environment in report.get("environments", ()):
            rom = environment.get("rom_sha256")
            suffix = f": ROM {rom[:12]}…" if rom else ""
            print(f"  {environment['env_id']}{suffix}")
        return 0
    if args.command == "run":
        workflow = get_reproduction_workflow(args.key)
        if args.variant and args.variant not in workflow.variants:
            parser.error(f"{args.key} variant must be one of: {', '.join(workflow.variants) or 'none'}")
        run_dir = workflow.run(args.environment, args.seed, args.preset, args.runs_dir, args.variant)
        print(f"completed: {run_dir}")
        return 0

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
