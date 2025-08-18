"""Monotonic Optimal Binning results visualization.

This module provides plotting functions for visualizing the final binning results,
including WoE patterns, event rates, and bin distributions.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.ticker as mticker

from MOBPY.exceptions import DataError, NotFittedError
from MOBPY.config import get_config
from MOBPY.logging_utils import get_logger

logger = get_logger(__name__)


def plot_woe_bars(
    summary_df: pd.DataFrame,
    *,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    bar_color: str = "#1976D2",  # Vivid blue
    positive_color: str = "#388E3C",  # Vivid green
    negative_color: str = "#D32F2F",  # Vivid red
    show_values: bool = True,
    show_iv: bool = True,
    rotation: int = 45,
    bar_width: float = 0.8,
) -> Axes:
    """Plot Weight of Evidence (WoE) as bars for each bin.
    
    For binary targets, visualizes the WoE values which indicate the
    predictive power of each bin. Positive WoE suggests lower risk,
    negative WoE suggests higher risk.
    
    Args:
        summary_df: Summary DataFrame from binner.summary_() with WoE column.
        ax: Matplotlib axes to plot on. If None, creates new figure.
        figsize: Figure size if creating new figure.
        title: Plot title. If None, uses default.
        bar_color: Color for bars (used if not using positive/negative colors).
        positive_color: Color for positive WoE bars.
        negative_color: Color for negative WoE bars.
        show_values: Whether to show WoE values on bars.
        show_iv: Whether to show IV values in subtitle.
        rotation: Rotation angle for x-axis labels.
        bar_width: Width of bars (0-1).
        
    Returns:
        Axes object with the plot.
        
    Raises:
        DataError: If required columns are missing.
        
    Examples:
        >>> binner = MonotonicBinner(df, x='age', y='default')
        >>> binner.fit()
        >>> summary = binner.summary_()
        >>> ax = plot_woe_bars(summary)
        >>> plt.show()
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Remove background and spines
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Validate input
    if 'woe' not in summary_df.columns:
        raise DataError("summary_df must have 'woe' column. Ensure target is binary.")
    
    if 'bucket' not in summary_df.columns:
        raise DataError("summary_df must have 'bucket' column")
    
    # Filter to bins with valid WoE (exclude Missing/Excluded with NaN WoE)
    plot_df = summary_df[summary_df['woe'].notna()].copy()
    
    if len(plot_df) == 0:
        raise DataError("No bins with valid WoE values to plot")
    
    # Prepare data
    buckets = plot_df['bucket'].values
    woe_values = plot_df['woe'].values
    positions = np.arange(len(buckets))
    
    # Determine bar colors
    if positive_color and negative_color:
        colors = [positive_color if w >= 0 else negative_color for w in woe_values]
    else:
        colors = bar_color
    
    # Create bars
    bars = ax.bar(
        positions,
        woe_values,
        width=bar_width,
        color=colors,
        edgecolor='white',
        linewidth=1.5,
        alpha=0.9
    )
    
    # Add value labels on bars
    if show_values:
        for i, (bar, woe) in enumerate(zip(bars, woe_values)):
            height = bar.get_height()
            y_pos = height + 0.02 if height >= 0 else height - 0.02
            va = 'bottom' if height >= 0 else 'top'
            
            ax.text(
                bar.get_x() + bar.get_width()/2,
                y_pos,
                f'{woe:.3f}',
                ha='center',
                va=va,
                fontsize=9
            )
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Styling
    ax.set_xticks(positions)
    ax.set_xticklabels(buckets, rotation=rotation, ha='right' if rotation > 0 else 'center')
    ax.set_xlabel('Bins', fontsize=11)
    ax.set_ylabel('Weight of Evidence (WoE)', fontsize=11)
    
    # Title
    if title is None:
        title = 'Weight of Evidence by Bin'
    
    # Add IV to title if requested
    if show_iv and 'iv' in summary_df.columns:
        total_iv = summary_df['iv'].sum()
        title += f'\n(Total IV: {total_iv:.4f})'
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Remove grid
    ax.grid(False)
    
    # Add legend if using colored bars
    if positive_color and negative_color:
        pos_patch = mpatches.Patch(color=positive_color, label='Positive WoE (lower risk)')
        neg_patch = mpatches.Patch(color=negative_color, label='Negative WoE (higher risk)')
        ax.legend(handles=[pos_patch, neg_patch], loc='best', frameon=True, fancybox=False, shadow=False)
    
    return ax


def plot_event_rate(
    summary_df: pd.DataFrame,
    *,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    bar_color: str = "#64B5F6",  # Light blue
    line_color: str = "#E53935",  # Vivid red
    show_counts: bool = True,
    show_rate_values: bool = True,
    rotation: int = 45,
    y_format: str = "percentage",
) -> Axes:
    """Plot event rate (mean of y) by bin with sample sizes.
    
    Shows both the event rate as bars and optionally the sample count,
    useful for assessing both the pattern and the reliability of each bin.
    
    Args:
        summary_df: Summary DataFrame from binner.summary_().
        ax: Matplotlib axes to plot on. If None, creates new figure.
        figsize: Figure size if creating new figure.
        title: Plot title. If None, uses default.
        bar_color: Color for count bars.
        line_color: Color for event rate line.
        show_counts: Whether to show sample counts as bars.
        show_rate_values: Whether to show rate values on the line.
        rotation: Rotation angle for x-axis labels.
        y_format: Format for y-axis ('percentage' or 'decimal').
        
    Returns:
        Axes object with the plot.
        
    Examples:
        >>> summary = binner.summary_()
        >>> ax = plot_event_rate(summary, show_counts=True)
        >>> plt.show()
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Remove background and spines
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Validate input
    required_cols = {'bucket', 'mean', 'count'}
    if not required_cols.issubset(summary_df.columns):
        missing = required_cols - set(summary_df.columns)
        raise DataError(f"summary_df missing required columns: {missing}")
    
    # Prepare data
    buckets = summary_df['bucket'].values
    event_rates = summary_df['mean'].values
    counts = summary_df['count'].values
    positions = np.arange(len(buckets))
    
    # Create twin axis for different scales
    ax2 = ax.twinx()
    ax2.spines['top'].set_visible(False)
    
    # Plot counts as bars (if requested)
    if show_counts:
        bars = ax.bar(
            positions,
            counts,
            width=0.8,
            color=bar_color,
            edgecolor='white',
            linewidth=1.5,
            alpha=0.7,
            label='Sample count'
        )
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + ax.get_ylim()[1] * 0.01,
                f'{int(count):,}',
                ha='center',
                va='bottom',
                fontsize=8
            )
    
    # Plot event rate as line
    if y_format == "percentage":
        plot_rates = event_rates * 100
        rate_label = 'Event rate (%)'
    else:
        plot_rates = event_rates
        rate_label = 'Event rate'
    
    line = ax2.plot(
        positions,
        plot_rates,
        'o-',
        color=line_color,
        linewidth=2.5,
        markersize=8,
        markeredgecolor='white',
        markeredgewidth=2,
        label=rate_label
    )
    
    # Add rate values on the line
    if show_rate_values:
        for i, (pos, rate) in enumerate(zip(positions, plot_rates)):
            format_str = f'{rate:.1f}%' if y_format == "percentage" else f'{rate:.3f}'
            ax2.text(
                pos,
                rate + (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.02,
                format_str,
                ha='center',
                va='bottom',
                fontsize=9,
                color=line_color
            )
    
    # Styling
    ax.set_xticks(positions)
    ax.set_xticklabels(buckets, rotation=rotation, ha='right' if rotation > 0 else 'center')
    ax.set_xlabel('Bins', fontsize=11)
    
    if show_counts:
        ax.set_ylabel('Sample Count', fontsize=11, color='black')
        ax.tick_params(axis='y', labelcolor='black')
    
    ax2.set_ylabel(rate_label, fontsize=11, color=line_color)
    ax2.tick_params(axis='y', labelcolor=line_color)
    
    # Format y-axis
    if y_format == "percentage":
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
    
    # Title
    if title is None:
        title = 'Event Rate and Sample Distribution by Bin'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Remove grid
    ax.grid(False)
    ax2.grid(False)
    
    # Legend
    if show_counts:
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', frameon=True, fancybox=False, shadow=False)
    
    return ax


def plot_bin_statistics(
    binner,
    *,
    figsize: Tuple[float, float] = (15, 10),
    title: Optional[str] = None,
) -> Figure:
    """Create comprehensive visualization of binning results.
    
    Creates a multi-panel plot showing:
    1. WoE pattern (if binary target)
    2. Event rate by bin
    3. Sample distribution
    4. Bin boundaries on original data
    
    Args:
        binner: Fitted MonotonicBinner instance.
        figsize: Figure size.
        title: Overall figure title.
        
    Returns:
        Figure object with subplots.
        
    Raises:
        NotFittedError: If binner is not fitted.
        
    Examples:
        >>> binner = MonotonicBinner(df, x='age', y='default')
        >>> binner.fit()
        >>> fig = plot_bin_statistics(binner)
        >>> plt.show()
    """
    # Check if fitted
    if not hasattr(binner, '_is_fitted') or not binner._is_fitted:
        raise NotFittedError("Binner must be fitted first")
    
    # Get data
    summary = binner.summary_()
    bins = binner.bins_()
    is_binary = binner._is_binary_y
    
    # Determine layout based on target type
    if is_binary:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
    
    # Set white background for all subplots
    for ax in axes:
        ax.set_facecolor('white')
    
    # Plot 1: WoE bars (binary only) or Event rate
    if is_binary and 'woe' in summary.columns:
        plot_woe_bars(summary, ax=axes[0], title='Weight of Evidence Pattern')
    else:
        # For non-binary, show event rate in first panel
        plot_event_rate(
            summary,
            ax=axes[0],
            title='Mean Target Value by Bin',
            show_counts=False,
            y_format='decimal'
        )
    
    # Plot 2: Event rate with counts
    plot_event_rate(
        summary,
        ax=axes[1],
        title='Event Rate and Sample Distribution',
        show_counts=True,
        y_format='percentage' if is_binary else 'decimal'
    )
    
    # Plot 3: Sample distribution
    plot_sample_distribution(
        summary,
        ax=axes[2],
        title='Sample Distribution Across Bins'
    )
    
    # Plot 4: Bin boundaries on data
    plot_bin_boundaries(
        binner,
        ax=axes[3],
        title='Bin Boundaries on Original Data'
    )
    
    # Overall title
    if title is None:
        target_type = "Binary" if is_binary else "Continuous"
        title = f'Monotonic Optimal Binning Results: {binner.x} → {binner.y} ({target_type} Target)'
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig


def plot_sample_distribution(
    summary_df: pd.DataFrame,
    *,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    color: str = "#42A5F5",  # Vivid light blue
    show_percentages: bool = True,
    show_cumulative: bool = True,
    rotation: int = 45,
) -> Axes:
    """Plot distribution of samples across bins.
    
    Shows how data is distributed across bins, useful for identifying
    bins that may be too small or too large.
    
    Args:
        summary_df: Summary DataFrame from binner.summary_().
        ax: Matplotlib axes. If None, creates new figure.
        figsize: Figure size if creating new figure.
        title: Plot title.
        color: Bar color.
        show_percentages: Whether to show percentage labels.
        show_cumulative: Whether to show cumulative percentage line.
        rotation: Rotation for x-axis labels.
        
    Returns:
        Axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Remove background and spines
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Prepare data
    buckets = summary_df['bucket'].values
    counts = summary_df['count'].values
    percentages = summary_df['count_pct'].values if 'count_pct' in summary_df else counts / counts.sum() * 100
    positions = np.arange(len(buckets))
    
    # Create bars
    bars = ax.bar(
        positions,
        percentages,
        color=color,
        edgecolor='white',
        linewidth=1.5,
        alpha=0.8
    )
    
    # Add percentage labels
    if show_percentages:
        for bar, pct in zip(bars, percentages):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{pct:.1f}%',
                ha='center',
                va='bottom',
                fontsize=9
            )
    
    # Add cumulative line
    if show_cumulative:
        ax2 = ax.twinx()
        ax2.spines['top'].set_visible(False)
        cumulative = np.cumsum(percentages)
        ax2.plot(
            positions,
            cumulative,
            'o-',
            color='#E53935',  # Vivid red
            linewidth=2.5,
            markersize=6,
            markeredgecolor='white',
            markeredgewidth=1.5,
            label='Cumulative %'
        )
        ax2.set_ylabel('Cumulative Percentage', fontsize=11, color='#E53935')
        ax2.tick_params(axis='y', labelcolor='#E53935')
        ax2.set_ylim(0, 105)
        
        # Add 50% reference line
        ax2.axhline(y=50, color='#E53935', linestyle='--', alpha=0.3)
    
    # Styling
    ax.set_xticks(positions)
    ax.set_xticklabels(buckets, rotation=rotation, ha='right' if rotation > 0 else 'center')
    ax.set_xlabel('Bins', fontsize=11)
    ax.set_ylabel('Percentage of Samples', fontsize=11)
    ax.set_ylim(0, max(percentages) * 1.15)
    
    if title is None:
        title = 'Sample Distribution Across Bins'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Remove grid
    ax.grid(False)
    if show_cumulative:
        ax2.grid(False)
    
    return ax


def plot_bin_boundaries(
    binner,
    *,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    n_samples: int = 1000,
    show_density: bool = True,
    show_boundaries: bool = True,
    show_means: bool = True,
    alpha: float = 0.6,
) -> Axes:
    """Plot bin boundaries overlaid on the original data distribution.
    
    Visualizes where the bin cuts are made on the original feature distribution.
    
    Args:
        binner: Fitted MonotonicBinner instance.
        ax: Matplotlib axes. If None, creates new figure.
        figsize: Figure size if creating new figure.
        title: Plot title.
        n_samples: Max samples to plot (for performance).
        show_density: Whether to show density plot.
        show_boundaries: Whether to show bin boundaries.
        show_means: Whether to show bin means.
        alpha: Transparency for density plot.
        
    Returns:
        Axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Remove background and spines
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Get clean data
    clean_data = binner._parts.clean if hasattr(binner, '_parts') else binner.df
    x_data = clean_data[binner.x].dropna()
    y_data = clean_data[binner.y].dropna()
    
    # Sample if too large
    if len(x_data) > n_samples:
        indices = np.random.choice(len(x_data), n_samples, replace=False)
        x_data = x_data.iloc[indices]
        y_data = y_data.iloc[indices]
    
    # Get bins
    bins_df = binner.bins_()
    
    # Plot data distribution
    if show_density:
        # Create histogram
        counts, bin_edges, patches = ax.hist(
            x_data,
            bins=50,
            density=True,
            alpha=alpha,
            color='#64B5F6',  # Light blue
            edgecolor='white',
            linewidth=0.5
        )
        ax.set_ylabel('Density', fontsize=11)
    else:
        # Scatter plot
        ax.scatter(
            x_data,
            y_data,
            alpha=0.5,
            s=10,
            c='#1976D2'  # Blue
        )
        ax.set_ylabel(binner.y, fontsize=11)
    
    # Plot bin boundaries
    if show_boundaries:
        # Get unique boundaries (excluding -inf and inf)
        boundaries = []
        for _, row in bins_df.iterrows():
            if not np.isneginf(row['left']):
                boundaries.append(row['left'])
            if not np.isposinf(row['right']):
                boundaries.append(row['right'])
        
        boundaries = sorted(set(boundaries))
        
        # Plot vertical lines at boundaries
        for boundary in boundaries:
            ax.axvline(
                x=boundary,
                color='#E53935',  # Red
                linestyle='--',
                linewidth=2,
                alpha=0.8,
                label='Bin boundary' if boundary == boundaries[0] else None
            )
            
            # Add boundary value
            ax.text(
                boundary,
                ax.get_ylim()[1] * 0.95,
                f'{boundary:.2f}',
                rotation=90,
                va='top',
                ha='right',
                fontsize=8,
                color='#E53935'
            )
    
    # Plot bin means if requested
    if show_means and not show_density:
        ax2 = ax.twinx()
        ax2.spines['top'].set_visible(False)
        
        # Calculate mean x position for each bin
        x_positions = []
        y_means = []
        
        for _, row in bins_df.iterrows():
            # Get data in this bin
            mask = True
            if not np.isneginf(row['left']):
                mask &= (clean_data[binner.x] >= row['left'])
            if not np.isposinf(row['right']):
                mask &= (clean_data[binner.x] < row['right'])
            
            bin_data = clean_data[mask]
            if len(bin_data) > 0:
                x_positions.append(bin_data[binner.x].mean())
                y_means.append(row['mean'])
        
        # Plot means
        ax2.plot(
            x_positions,
            y_means,
            'o-',
            color='#E53935',  # Red
            linewidth=2,
            markersize=8,
            markeredgecolor='white',
            markeredgewidth=1.5,
            label='Bin means'
        )
        ax2.set_ylabel('Bin Mean', fontsize=11, color='#E53935')
        ax2.tick_params(axis='y', labelcolor='#E53935')
        ax2.grid(False)
    
    # Styling
    ax.set_xlabel(binner.x, fontsize=11)
    
    if title is None:
        title = f'Bin Boundaries on {binner.x} Distribution'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Remove grid
    ax.grid(False)
    
    # Legend
    if show_boundaries:
        ax.legend(loc='best', frameon=True, fancybox=False, shadow=False)
    
    return ax


def plot_binning_stability(
    binner,
    test_df: pd.DataFrame,
    *,
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = None,
) -> Figure:
    """Compare binning performance on training vs test data.
    
    Useful for assessing whether the binning generalizes well to new data.
    
    Args:
        binner: Fitted MonotonicBinner instance.
        test_df: Test DataFrame with same x and y columns.
        figsize: Figure size.
        title: Plot title.
        
    Returns:
        Figure with comparison plots.
        
    Examples:
        >>> binner = MonotonicBinner(train_df, x='age', y='default')
        >>> binner.fit()
        >>> fig = plot_binning_stability(binner, test_df)
        >>> plt.show()
    """
    # Transform test data
    test_bins = binner.transform(test_df[binner.x])
    
    # Calculate test statistics per bin
    test_stats = []
    for bin_label in binner.summary_()['bucket']:
        if bin_label.startswith('Missing') or bin_label.startswith('Excluded'):
            continue
        
        mask = test_bins == bin_label
        if mask.sum() > 0:
            test_stats.append({
                'bucket': bin_label,
                'count': mask.sum(),
                'mean': test_df.loc[mask, binner.y].mean(),
                'std': test_df.loc[mask, binner.y].std()
            })
    
    test_stats_df = pd.DataFrame(test_stats)
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Set white background
    for ax in [ax1, ax2]:
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Plot 1: Sample distribution comparison
    train_summary = binner.summary_()
    numeric_mask = ~train_summary['bucket'].str.contains('Missing|Excluded')
    
    x_pos = np.arange(len(test_stats_df))
    width = 0.35
    
    ax1.bar(
        x_pos - width/2,
        train_summary.loc[numeric_mask, 'count_pct'],
        width,
        label='Train',
        alpha=0.8,
        color='#1976D2',  # Blue
        edgecolor='white',
        linewidth=1.5
    )
    
    test_pct = test_stats_df['count'] / test_stats_df['count'].sum() * 100
    ax1.bar(
        x_pos + width/2,
        test_pct,
        width,
        label='Test',
        alpha=0.8,
        color='#388E3C',  # Green
        edgecolor='white',
        linewidth=1.5
    )
    
    ax1.set_xlabel('Bins', fontsize=11)
    ax1.set_ylabel('Percentage of Samples', fontsize=11)
    ax1.set_title('Sample Distribution: Train vs Test', fontsize=13)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(test_stats_df['bucket'], rotation=45, ha='right')
    ax1.legend(frameon=True, fancybox=False, shadow=False)
    ax1.grid(False)
    
    # Plot 2: Event rate comparison
    ax2.plot(
        x_pos,
        train_summary.loc[numeric_mask, 'mean'] * 100,
        'o-',
        linewidth=2.5,
        markersize=8,
        color='#1976D2',  # Blue
        markeredgecolor='white',
        markeredgewidth=1.5,
        label='Train'
    )
    ax2.plot(
        x_pos,
        test_stats_df['mean'] * 100,
        's-',
        linewidth=2.5,
        markersize=8,
        color='#388E3C',  # Green
        markeredgecolor='white',
        markeredgewidth=1.5,
        label='Test'
    )
    
    ax2.set_xlabel('Bins', fontsize=11)
    ax2.set_ylabel('Event Rate (%)', fontsize=11)
    ax2.set_title('Event Rate: Train vs Test', fontsize=13)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(test_stats_df['bucket'], rotation=45, ha='right')
    ax2.legend(frameon=True, fancybox=False, shadow=False)
    ax2.grid(False)
    
    # Overall title
    if title is None:
        title = f'Binning Stability Analysis: {binner.x}'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    return fig