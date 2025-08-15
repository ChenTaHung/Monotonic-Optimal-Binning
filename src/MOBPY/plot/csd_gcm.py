"""Cumulative Sum Diagram (CSD) and Greatest Convex Minorant (GCM) plotting.

This module provides visualization tools for understanding the PAVA algorithm's
behavior. The CSD shows cumulative sums, while the GCM represents the monotonic
fit produced by PAVA.
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from MOBPY.core.pava import PAVA
from MOBPY.exceptions import DataError
from MOBPY.config import get_config
from MOBPY.logging_utils import get_logger

logger = get_logger(__name__)


def plot_csd(
    groups_df: pd.DataFrame,
    blocks: List[Dict[str, Any]],
    *,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    show_points: bool = True,
    show_blocks: bool = True,
    point_color: str = "darkblue",
    block_color: str = "red",
    block_alpha: float = 0.3,
    point_size: float = 50,
    line_width: float = 2,
    show_legend: bool = True,
) -> Axes:
    """Plot Cumulative Sum Diagram (CSD) with PAVA blocks.
    
    The CSD visualizes the cumulative sum of y values against cumulative count,
    showing how PAVA creates monotonic blocks by pooling adjacent groups.
    
    Args:
        groups_df: DataFrame from PAVA.groups_ with columns ['x', 'count', 'sum'].
        blocks: List of block dictionaries from PAVA.export_blocks(as_dict=True).
        ax: Matplotlib axes to plot on. If None, creates new figure.
        figsize: Figure size if creating new figure.
        title: Plot title. If None, uses default.
        show_points: Whether to show individual group points.
        show_blocks: Whether to show PAVA block regions.
        point_color: Color for group points.
        block_color: Color for block regions.
        block_alpha: Transparency for block regions.
        point_size: Size of scatter points.
        line_width: Width of connecting lines.
        show_legend: Whether to show legend.
        
    Returns:
        Axes object with the plot.
        
    Examples:
        >>> pava = PAVA(df=data, x='age', y='default')
        >>> pava.fit()
        >>> ax = plot_csd(pava.groups_, pava.export_blocks(as_dict=True))
        >>> plt.show()
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Validate input
    required_cols = {'count', 'sum'}
    if not required_cols.issubset(groups_df.columns):
        missing = required_cols - set(groups_df.columns)
        raise DataError(f"groups_df missing required columns: {missing}")
    
    # Calculate cumulative values
    cum_count = groups_df['count'].cumsum()
    cum_sum = groups_df['sum'].cumsum()
    
    # Add origin point for better visualization
    cum_count_with_origin = np.concatenate([[0], cum_count.values])
    cum_sum_with_origin = np.concatenate([[0], cum_sum.values])
    
    # Plot connecting line
    ax.plot(
        cum_count_with_origin,
        cum_sum_with_origin,
        'o-',
        color=point_color,
        linewidth=line_width,
        alpha=0.6,
        markersize=4,
        label='Cumulative path'
    )
    
    # Plot individual points
    if show_points:
        ax.scatter(
            cum_count,
            cum_sum,
            c=point_color,
            s=point_size,
            alpha=0.8,
            edgecolors='white',
            linewidth=1,
            label='Group points',
            zorder=5
        )
    
    # Plot PAVA blocks
    if show_blocks and blocks:
        # Calculate block cumulative positions
        block_cum_n = 0
        block_cum_sum = 0
        
        for i, block in enumerate(blocks):
            # Block start position
            start_n = block_cum_n
            start_sum = block_cum_sum
            
            # Block end position
            end_n = start_n + block['n']
            end_sum = start_sum + block['sum']
            
            # Draw block line (GCM segment)
            ax.plot(
                [start_n, end_n],
                [start_sum, end_sum],
                color=block_color,
                linewidth=line_width * 1.5,
                alpha=0.8,
                label='PAVA blocks' if i == 0 else None
            )
            
            # Shade block region
            if block_alpha > 0:
                # Create vertices for the polygon
                vertices = [
                    (start_n, start_sum),
                    (end_n, end_sum),
                    (end_n, 0),
                    (start_n, 0)
                ]
                polygon = mpatches.Polygon(
                    vertices,
                    facecolor=block_color,
                    alpha=block_alpha,
                    edgecolor='none'
                )
                ax.add_patch(polygon)
            
            # Update cumulative position
            block_cum_n = end_n
            block_cum_sum = end_sum
    
    # Styling
    ax.set_xlabel('Cumulative Count', fontsize=12)
    ax.set_ylabel('Cumulative Sum of Y', fontsize=12)
    
    if title is None:
        title = 'Cumulative Sum Diagram with PAVA Blocks'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    if show_legend:
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Set axis limits with some padding
    ax.set_xlim(-cum_count.max() * 0.02, cum_count.max() * 1.02)
    
    return ax


def plot_gcm(
    groups_df: pd.DataFrame,
    blocks: List[Dict[str, Any]],
    *,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    show_means: bool = True,
    show_blocks: bool = True,
    mean_color: str = "blue",
    block_color: str = "red",
    block_style: str = "step",
    mean_marker: str = "o",
    mean_size: float = 50,
    line_width: float = 2,
    show_legend: bool = True,
    show_violations: bool = False,
) -> Axes:
    """Plot Greatest Convex Minorant (GCM) showing group means and PAVA fit.
    
    The GCM visualizes how PAVA transforms non-monotonic group means into
    a monotonic step function by pooling adjacent violating groups.
    
    Args:
        groups_df: DataFrame from PAVA.groups_ with columns ['x', 'group_mean'].
        blocks: List of block dictionaries from PAVA.export_blocks(as_dict=True).
        ax: Matplotlib axes to plot on. If None, creates new figure.
        figsize: Figure size if creating new figure.
        title: Plot title. If None, uses default.
        show_means: Whether to show original group means.
        show_blocks: Whether to show PAVA block means.
        mean_color: Color for original means.
        block_color: Color for PAVA blocks.
        block_style: Style for blocks ('step' or 'line').
        mean_marker: Marker style for means.
        mean_size: Size of mean markers.
        line_width: Width of block lines.
        show_legend: Whether to show legend.
        show_violations: Whether to highlight monotonicity violations.
        
    Returns:
        Axes object with the plot.
        
    Examples:
        >>> pava = PAVA(df=data, x='age', y='default', sign='+')
        >>> pava.fit()
        >>> ax = plot_gcm(pava.groups_, pava.export_blocks(as_dict=True))
        >>> plt.show()
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Validate input
    if 'x' not in groups_df.columns:
        raise DataError("groups_df must have 'x' column")
    
    # Calculate group means if not present
    if 'group_mean' not in groups_df.columns:
        if 'sum' in groups_df.columns and 'count' in groups_df.columns:
            groups_df = groups_df.copy()
            groups_df['group_mean'] = groups_df['sum'] / groups_df['count']
        else:
            raise DataError("groups_df must have 'group_mean' or 'sum'/'count' columns")
    
    # Extract data
    x_values = groups_df['x'].values
    y_means = groups_df['group_mean'].values
    
    # Plot original means
    if show_means:
        ax.scatter(
            x_values,
            y_means,
            c=mean_color,
            s=mean_size,
            marker=mean_marker,
            alpha=0.7,
            edgecolors='white',
            linewidth=1,
            label='Original means',
            zorder=5
        )
        
        # Connect with light line
        ax.plot(
            x_values,
            y_means,
            color=mean_color,
            alpha=0.3,
            linewidth=1,
            linestyle='--'
        )
    
    # Highlight violations if requested
    if show_violations and len(y_means) > 1:
        # Detect monotonicity direction from blocks
        if blocks and len(blocks) > 1:
            # Infer from block means
            block_means = [b['sum'] / b['n'] for b in blocks if b['n'] > 0]
            is_increasing = block_means[-1] >= block_means[0]
        else:
            is_increasing = True
        
        # Find violations
        for i in range(1, len(y_means)):
            if is_increasing and y_means[i] < y_means[i-1]:
                # Increasing violation
                ax.plot(
                    [x_values[i-1], x_values[i]],
                    [y_means[i-1], y_means[i]],
                    color='red',
                    linewidth=3,
                    alpha=0.5,
                    zorder=4
                )
            elif not is_increasing and y_means[i] > y_means[i-1]:
                # Decreasing violation
                ax.plot(
                    [x_values[i-1], x_values[i]],
                    [y_means[i-1], y_means[i]],
                    color='red',
                    linewidth=3,
                    alpha=0.5,
                    zorder=4
                )
    
    # Plot PAVA blocks
    if show_blocks and blocks:
        block_x = []
        block_y = []
        
        for block in blocks:
            block_mean = block['sum'] / block['n'] if block['n'] > 0 else 0
            
            if block_style == 'step':
                # Create step function representation
                left = block['left'] if not np.isneginf(block['left']) else x_values.min() - 0.1 * (x_values.max() - x_values.min())
                right = block['right'] if not np.isposinf(block['right']) else x_values.max() + 0.1 * (x_values.max() - x_values.min())
                
                block_x.extend([left, right])
                block_y.extend([block_mean, block_mean])
            else:
                # Line style: use block center
                if not np.isneginf(block['left']) and not np.isposinf(block['right']):
                    center = (block['left'] + block['right']) / 2
                elif not np.isneginf(block['left']):
                    center = block['left']
                elif not np.isposinf(block['right']):
                    center = block['right']
                else:
                    center = x_values.mean()
                
                block_x.append(center)
                block_y.append(block_mean)
        
        # Plot blocks
        if block_style == 'step':
            # Sort for proper step display
            sorted_indices = np.argsort(block_x)
            block_x = np.array(block_x)[sorted_indices]
            block_y = np.array(block_y)[sorted_indices]
            
            ax.step(
                block_x,
                block_y,
                where='post',
                color=block_color,
                linewidth=line_width,
                alpha=0.8,
                label='PAVA fit'
            )
        else:
            ax.plot(
                block_x,
                block_y,
                'o-',
                color=block_color,
                linewidth=line_width,
                markersize=8,
                alpha=0.8,
                label='PAVA fit'
            )
    
    # Styling
    ax.set_xlabel('X values', fontsize=12)
    ax.set_ylabel('Mean of Y', fontsize=12)
    
    if title is None:
        title = 'Greatest Convex Minorant (PAVA Fit)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Legend
    if show_legend:
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # Set reasonable axis limits
    x_margin = (x_values.max() - x_values.min()) * 0.05
    ax.set_xlim(x_values.min() - x_margin, x_values.max() + x_margin)
    
    return ax


def plot_pava_animation(
    pava: PAVA,
    *,
    interval: int = 1000,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
    show_csd: bool = True,
    show_gcm: bool = True,
) -> Union[None, Any]:
    """Create animated visualization of PAVA algorithm progress.
    
    Shows how PAVA progressively merges groups to achieve monotonicity.
    This requires matplotlib animation support and is optional.
    
    Args:
        pava: Fitted PAVA instance.
        interval: Milliseconds between frames.
        save_path: If provided, saves animation as GIF/MP4.
        figsize: Figure size.
        show_csd: Whether to show CSD plot.
        show_gcm: Whether to show GCM plot.
        
    Returns:
        Animation object if successful, None if animation not available.
        
    Notes:
        Requires additional dependencies:
        - For GIF: pillow or imagemagick
        - For MP4: ffmpeg
    """
    try:
        from matplotlib import animation
    except ImportError:
        warnings.warn(
            "Animation requires matplotlib.animation. "
            "Install with: pip install matplotlib[animation]",
            UserWarning
        )
        return None
    
    # Implementation of animation would go here
    # For now, we'll leave this as a stub
    logger.info("Animation feature not yet implemented")
    return None


def plot_pava_comparison(
    binner,
    *,
    figsize: Tuple[float, float] = (15, 6),
    title: Optional[str] = None,
) -> Figure:
    """Plot side-by-side comparison of data before and after PAVA.
    
    Shows original group means and final PAVA blocks for easy comparison.
    
    Args:
        binner: Fitted MonotonicBinner instance.
        figsize: Figure size.
        title: Overall figure title.
        
    Returns:
        Figure object with subplots.
        
    Examples:
        >>> binner = MonotonicBinner(df, x='age', y='default')
        >>> binner.fit()
        >>> fig = plot_pava_comparison(binner)
        >>> plt.show()
    """
    # Get PAVA data
    pava = getattr(binner, '_pava', None)
    if pava is None or pava.groups_ is None:
        raise RuntimeError("Binner must be fitted first")
    
    groups_df = pava.groups_
    blocks = pava.export_blocks(as_dict=True)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot CSD
    plot_csd(
        groups_df,
        blocks,
        ax=ax1,
        title='Cumulative Sum Diagram',
        show_legend=True
    )
    
    # Plot GCM
    plot_gcm(
        groups_df,
        blocks,
        ax=ax2,
        title='Group Means vs PAVA Fit',
        show_legend=True,
        show_violations=True
    )
    
    # Overall title
    if title is None:
        direction = "increasing" if binner.resolved_sign_ == "+" else "decreasing"
        title = f'PAVA Analysis: {binner.x} vs {binner.y} ({direction} monotonicity)'
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout()
    
    return fig