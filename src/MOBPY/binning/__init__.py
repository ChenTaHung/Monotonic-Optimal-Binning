"""Binning orchestrators and public interfaces.

This module provides the main user-facing API for monotonic optimal binning.
The MonotonicBinner class orchestrates the complete pipeline from data
partitioning through PAVA and merging to final bin creation.

Main Classes:
    MonotonicBinner: End-to-end monotonic optimal binning orchestrator that
                     handles data partitioning, PAVA fitting, constraint-based
                     merging, and WoE/IV calculation for binary targets.

Example:
    >>> from MOBPY.binning import MonotonicBinner
    >>> from MOBPY.core import BinningConstraints
    >>> 
    >>> # Create constraints
    >>> constraints = BinningConstraints(max_bins=5, min_samples=0.05)
    >>> 
    >>> # Fit binner
    >>> binner = MonotonicBinner(df, x='age', y='default', constraints=constraints)
    >>> binner.fit()
    >>> 
    >>> # Get results
    >>> bins = binner.bins_()
    >>> summary = binner.summary_()
"""

from .mob import MonotonicBinner

__all__ = ["MonotonicBinner"]