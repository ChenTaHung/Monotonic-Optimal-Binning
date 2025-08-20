"""Visualization tools for monotonic optimal binning.

This module provides comprehensive plotting functions for visualizing both
the PAVA algorithm process and the final binning results. It includes tools
for understanding how monotonicity is achieved and assessing bin quality.

Main Components:
    CSD/GCM Plots (from csd_gcm):
        plot_gcm: Greatest Convex Minorant showing monotonic fit.
        plot_pava_comparison: Side-by-side CSD and GCM plots.
        plot_pava_process: PAVA merging process visualization.
        plot_pava_animation: Animated visualization of PAVA iterations.
    
    MOB Result Plots (from mob_plot):
        plot_woe_bars: Weight of Evidence visualization for binary targets.
        plot_event_rate: Event rate and sample distribution by bin.
        plot_bin_statistics: Comprehensive multi-panel binning results.
        plot_sample_distribution: Distribution of samples across bins.
        plot_bin_boundaries: Bin cuts overlaid on feature distribution.
        plot_binning_stability: Compare binning on train vs test data.

Example:
    >>> from MOBPY import MonotonicBinner
    >>> from MOBPY.plot import plot_bin_statistics, plot_pava_comparison
    >>> 
    >>> # Fit binner
    >>> binner = MonotonicBinner(df, x='age', y='default')
    >>> binner.fit()
    >>> 
    >>> # Visualize PAVA process
    >>> fig1 = plot_pava_comparison(binner)
    >>> 
    >>> # Visualize final results
    >>> fig2 = plot_bin_statistics(binner)
    >>> plt.show()

Note:
    All plotting functions return matplotlib objects (Axes or Figure) that
    can be further customized. Most functions accept an 'ax' parameter to
    plot on existing axes for creating custom layouts.
"""

# Import all plotting functions
from .csd_gcm import (
    # plot_cumulative_mean,  # Commented out for now, keeping for future reference
    plot_gcm,
    plot_pava_comparison,
    plot_pava_process,
    plot_pava_animation,
)

from .mob_plot import (
    plot_woe_bars,
    plot_event_rate,
    plot_bin_statistics,
    plot_sample_distribution,
    plot_bin_boundaries,
    plot_binning_stability,
)

__all__ = [
    # PAVA visualization
    # "plot_cumulative_mean",  # Commented out for now
    "plot_gcm", 
    "plot_pava_comparison",
    "plot_pava_process",
    "plot_pava_animation",
    
    # MOB results visualization
    "plot_woe_bars",
    "plot_event_rate",
    "plot_bin_statistics",
    "plot_sample_distribution",
    "plot_bin_boundaries",
    "plot_binning_stability"
]