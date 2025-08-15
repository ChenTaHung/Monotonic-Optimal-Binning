"""MOBPY: Monotonic Optimal Binning for Python.

A fast, deterministic library for creating monotonic bins with respect to a target variable.
Implements PAVA (Pool-Adjacent-Violators Algorithm) followed by constrained adjacent merging.

Basic Usage:
    >>> from MOBPY import MonotonicBinner, BinningConstraints
    >>> binner = MonotonicBinner(df, x="feature", y="target")
    >>> binner.fit()
    >>> bins = binner.bins_()
    >>> summary = binner.summary_()
"""

__version__ = "2.0.0"

# Core public API - everything users need at top level
from MOBPY.binning.mob import MonotonicBinner
from MOBPY.core.constraints import BinningConstraints

# Optional: Advanced users can access internals
from MOBPY import core, binning, plot

# Define what's available with "from MOBPY import *"
__all__ = [
    "MonotonicBinner",
    "BinningConstraints",
    "__version__",
    "core",
    "binning", 
    "plot",
]

def get_version() -> str:
    """Get the current version of MOBPY.
    
    Returns:
        str: Version string in semantic versioning format.
    """
    return __version__