"""mobpy package entrypoint (src/ layout).

This package provides:
- Core algorithms (PAVA, merging, constraints, utilities)
- A high-level binner that orchestrates PAVA + constraint-aware merging
- Optional plotting helpers
"""

__all__ = [
    "core",
    "binning",
    "plot",
]
