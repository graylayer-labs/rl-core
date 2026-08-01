"""Paper-reproduction specifications and validation."""

from rl_core.reproductions.spec import (
    ReproductionSpec,
    SpecValidationError,
    discover_reproduction_specs,
    load_reproduction_spec,
)

__all__ = [
    "ReproductionSpec",
    "SpecValidationError",
    "discover_reproduction_specs",
    "load_reproduction_spec",
]
