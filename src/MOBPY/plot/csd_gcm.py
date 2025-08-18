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


def plot_cumulative_mean(
    groups_df: pd.DataFrame,
    blocks: List[Dict[str, Any]],
    *,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    show_points: bool = True,
    show_blocks: bool = True,
    point_color: str = "#1E88E5",  # Vivid blue
    block_color: str = "#E53935",   # Vivid red
    block_alpha: float = 0.15,
    point_size: float = 60,
    line_width: float = 2,
    show_legend: bool = True,
    x_column: str = 'x',
) -> Axes:
    """Plot Cumulative Mean Diagram with PAVA blocks.
    
    Visualizes how the cumulative mean of the target variable evolves as we
    accumulate data points, showing how PAVA creates monotonic blocks.
    
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
        x_column: Name of the x column for labeling.
        
    Returns:
        Axes object with the plot.
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Remove background and spines
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Validate input
    required_cols = {'count', 'sum'}
    if not required_cols.issubset(groups_df.columns):
        missing = required_cols - set(groups_df.columns)
        raise DataError(f"groups_df missing required columns: {missing}")
    
    # Calculate cumulative values
    cum_count = groups_df['count'].cumsum()
    cum_sum = groups_df['sum'].cumsum()
    cum_mean = cum_sum / cum_count
    
    # Add origin point for better visualization
    cum_count_with_origin = np.concatenate([[0], cum_count.values])
    cum_mean_with_origin = np.concatenate([[0], cum_mean.values])
    
    # Plot connecting line
    ax.plot(
        cum_count_with_origin[1:],  # Skip origin for cleaner look
        cum_mean_with_origin[1:],
        '-',
        color=point_color,
        linewidth=line_width,
        alpha=0.7,
        label='Cumulative mean path'
    )
    
    # Plot individual points
    if show_points:
        ax.scatter(
            cum_count,
            cum_mean,
            c=point_color,
            s=point_size,
            alpha=0.9,
            edgecolors='white',
            linewidth=1.5,
            label='Group cumulative means',
            zorder=5
        )
    
    # Plot PAVA blocks as cumulative means
    if show_blocks and blocks:
        # Calculate block cumulative positions
        block_cum_n = 0
        block_cum_sum = 0
        
        for i, block in enumerate(blocks):
            # Block start position
            start_n = block_cum_n
            start_mean = block_cum_sum / start_n if start_n > 0 else 0
            
            # Block end position
            end_n = start_n + block['n']
            block_cum_sum += block['sum']
            end_mean = block_cum_sum / end_n
            
            # Draw block line segment
            ax.plot(
                [start_n, end_n],
                [start_mean, end_mean],
                color=block_color,
                linewidth=line_width * 1.8,
                alpha=0.9,
                label='PAVA blocks' if i == 0 else None
            )
            
            # Update cumulative position
            block_cum_n = end_n
    
    # Styling
    ax.set_xlabel(f'Cumulative Count (ordered by {x_column})', fontsize=11)
    ax.set_ylabel('Cumulative Mean of Target', fontsize=11)
    
    if title is None:
        title = 'Cumulative Mean Evolution with PAVA Blocks'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Remove grid
    ax.grid(False)
    
    # Legend
    if show_legend:
        ax.legend(loc='best', frameon=True, fancybox=False, shadow=False)
    
    # Set axis limits with some padding
    if len(cum_count) > 0:
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
    show_block_points: bool = True,
    mean_color: str = "#64B5F6",  # Light blue
    block_color: str = "#E53935",  # Vivid red
    block_point_color: str = "#D32F2F",  # Darker red
    mean_marker: str = "o",
    mean_size: float = 40,
    block_point_size: float = 100,
    line_width: float = 2.5,
    show_legend: bool = True,
    x_column: str = 'x',
    y_column: str = 'y',
) -> Axes:
    """Plot original group means with PAVA monotonic fit.
    
    Visualizes how PAVA transforms non-monotonic group means into a monotonic
    step function, highlighting where the final bins are formed.
    
    Args:
        groups_df: DataFrame from PAVA.groups_ with columns ['x', 'group_mean'].
        blocks: List of block dictionaries from PAVA.export_blocks(as_dict=True).
        ax: Matplotlib axes to plot on. If None, creates new figure.
        figsize: Figure size if creating new figure.
        title: Plot title. If None, uses default.
        show_means: Whether to show original group means.
        show_blocks: Whether to show PAVA block means.
        show_block_points: Whether to show points where PAVA forms bins.
        mean_color: Color for original means.
        block_color: Color for PAVA blocks.
        block_point_color: Color for block formation points.
        mean_marker: Marker style for means.
        mean_size: Size of mean markers.
        block_point_size: Size of block formation points.
        line_width: Width of block lines.
        show_legend: Whether to show legend.
        x_column: Name of x column for labeling.
        y_column: Name of y column for labeling.
        
    Returns:
        Axes object with the plot.
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Remove background and spines
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
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
        # Connect with light line first
        ax.plot(
            x_values,
            y_means,
            color=mean_color,
            alpha=0.3,
            linewidth=1.5,
            linestyle='-'
        )
        
        # Then scatter points
        ax.scatter(
            x_values,
            y_means,
            c=mean_color,
            s=mean_size,
            marker=mean_marker,
            alpha=0.8,
            edgecolors='white',
            linewidth=1,
            label='Original group means',
            zorder=5
        )
    
    # Plot PAVA blocks
    if show_blocks and blocks:
        # First pass: draw step function
        for i, block in enumerate(blocks):
            block_mean = block['sum'] / block['n'] if block['n'] > 0 else 0
            
            # Get x range for this block
            left = block['left'] if not np.isneginf(block['left']) else x_values.min() - 0.1 * (x_values.max() - x_values.min())
            right = block['right'] if not np.isposinf(block['right']) else x_values.max() + 0.1 * (x_values.max() - x_values.min())
            
            # Draw horizontal line for this block
            ax.plot(
                [left, right],
                [block_mean, block_mean],
                color=block_color,
                linewidth=line_width,
                alpha=0.9,
                label='PAVA monotonic fit' if i == 0 else None
            )
            
            # Draw vertical connectors between blocks
            if i > 0:
                prev_block = blocks[i-1]
                prev_mean = prev_block['sum'] / prev_block['n'] if prev_block['n'] > 0 else 0
                ax.plot(
                    [left, left],
                    [prev_mean, block_mean],
                    color=block_color,
                    linewidth=line_width,
                    alpha=0.9
                )
        
        # Second pass: show block formation points
        if show_block_points:
            block_centers = []
            block_means = []
            
            for block in blocks:
                # Calculate center of block based on included x values
                block_x_values = x_values[(x_values >= block['left']) & (x_values < block['right'])]
                if len(block_x_values) > 0:
                    center = block_x_values.mean()
                else:
                    # Fallback to block midpoint
                    if not np.isneginf(block['left']) and not np.isposinf(block['right']):
                        center = (block['left'] + block['right']) / 2
                    elif not np.isneginf(block['left']):
                        center = block['left'] + 1
                    else:
                        center = block['right'] - 1
                
                block_centers.append(center)
                block_means.append(block['sum'] / block['n'] if block['n'] > 0 else 0)
            
            ax.scatter(
                block_centers,
                block_means,
                c=block_point_color,
                s=block_point_size,
                marker='s',  # Square markers
                alpha=0.9,
                edgecolors='white',
                linewidth=2,
                label='Final bin positions',
                zorder=10
            )
    
    # Styling
    ax.set_xlabel(x_column, fontsize=11)
    ax.set_ylabel(f'Mean of {y_column}', fontsize=11)
    
    if title is None:
        title = 'Group Means and PAVA Monotonic Fit'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Remove grid
    ax.grid(False)
    
    # Legend
    if show_legend:
        ax.legend(loc='best', frameon=True, fancybox=False, shadow=False)
    
    # Set reasonable axis limits
    if len(x_values) > 0:
        x_margin = (x_values.max() - x_values.min()) * 0.05
        ax.set_xlim(x_values.min() - x_margin, x_values.max() + x_margin)
    
    return ax


def plot_pava_process(
    groups_df: pd.DataFrame,
    blocks: List[Dict[str, Any]],
    *,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (12, 7),
    title: Optional[str] = None,
    fallback_color: str = "#2196F3",  # Blue for merging process
    anchor_color: str = "#F44336",     # Red for completed bins
    initial_color: str = "#9E9E9E",    # Gray for initial points
    line_alpha: float = 0.4,
    point_size: float = 80,
    anchor_size: float = 120,
    show_legend: bool = True,
    x_column: str = 'x',
    y_column: str = 'y',
    show_annotations: bool = True,
) -> Axes:
    """Visualize PAVA's progressive merging process.
    
    Shows how PAVA iteratively merges groups to achieve monotonicity:
    - Blue dots: Intermediate merging steps (fallbacks)
    - Red dots: Completed bin anchors where merging stops
    - Lines show the cumulative mean evolution during the process
    
    Args:
        groups_df: DataFrame from PAVA.groups_ with columns ['x', 'count', 'sum'].
        blocks: Final blocks from PAVA.export_blocks(as_dict=True).
        ax: Matplotlib axes. If None, creates new figure.
        figsize: Figure size if creating new figure.
        title: Plot title.
        fallback_color: Color for intermediate merge points.
        anchor_color: Color for completed bin anchors.
        initial_color: Color for initial data points.
        line_alpha: Transparency for connecting lines.
        point_size: Size of fallback points.
        anchor_size: Size of anchor points.
        show_legend: Whether to show legend.
        x_column: Name of x column.
        y_column: Name of y column.
        show_annotations: Whether to annotate anchor points.
        
    Returns:
        Axes object with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Remove background and spines
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Calculate cumulative statistics
    cum_count = groups_df['count'].cumsum()
    cum_sum = groups_df['sum'].cumsum()
    cum_mean = cum_sum / cum_count
    x_values = groups_df['x'].values
    
    # Plot initial cumulative means as gray background
    ax.plot(cum_count, cum_mean, 'o-', color=initial_color, alpha=0.3, 
            markersize=4, linewidth=1, label='Initial cumulative means')
    
    # Map blocks to groups based on x values
    block_group_mapping = []
    for block in blocks:
        # Find which groups belong to this block
        group_indices = []
        for i, x in enumerate(x_values):
            if x >= block['left'] and x < block['right']:
                group_indices.append(i)
        
        if group_indices:
            block_group_mapping.append({
                'block': block,
                'group_indices': group_indices,
                'start_idx': min(group_indices),
                'end_idx': max(group_indices) + 1
            })
    
    # Visualize the merging process for each block
    for block_idx, mapping in enumerate(block_group_mapping):
        block = mapping['block']
        indices = mapping['group_indices']
        
        if len(indices) > 0:
            # Get cumulative values for groups in this block
            block_cum_counts = cum_count.iloc[indices]
            block_cum_means = cum_mean.iloc[indices]
            
            # Show merging process (blue dots for intermediate steps)
            if len(indices) > 1:
                # Plot intermediate merging steps (all but the last)
                for i in range(len(indices) - 1):
                    idx = indices[i]
                    ax.scatter(cum_count.iloc[idx], cum_mean.iloc[idx],
                             c=fallback_color, s=point_size, alpha=0.7,
                             edgecolors='white', linewidth=1, zorder=5)
                
                # Connect with blue lines to show merging path
                ax.plot(block_cum_counts, block_cum_means, '-', 
                       color=fallback_color, alpha=line_alpha, linewidth=2)
            
            # Plot anchor point (red) where this block is finalized
            final_idx = indices[-1]
            final_count = cum_count.iloc[final_idx]
            final_mean = cum_mean.iloc[final_idx]
            
            ax.scatter(final_count, final_mean, c=anchor_color, s=anchor_size,
                     marker='s', alpha=0.9, edgecolors='white', linewidth=2,
                     zorder=10, label='Completed bin anchors' if block_idx == 0 else None)
            
            # Add annotation for anchor points
            if show_annotations and block_idx < min(5, len(block_group_mapping)):
                ax.annotate(f'Bin {block_idx+1}', 
                          xy=(final_count, final_mean),
                          xytext=(10, 10), textcoords='offset points',
                          fontsize=9, color=anchor_color,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor=anchor_color, alpha=0.8))
    
    # Add visual connections between completed bins
    if len(block_group_mapping) > 1:
        anchor_counts = []
        anchor_means = []
        for mapping in block_group_mapping:
            if mapping['group_indices']:
                final_idx = mapping['group_indices'][-1]
                anchor_counts.append(cum_count.iloc[final_idx])
                anchor_means.append(cum_mean.iloc[final_idx])
        
        # Draw faint lines connecting anchors to show progression
        ax.plot(anchor_counts, anchor_means, '--', color=anchor_color, 
               alpha=0.3, linewidth=1.5, zorder=1)
    
    # Add arrows to show direction of process
    if len(blocks) > 1 and len(cum_count) > 0:
        # Add subtle arrow annotation
        ax.annotate('', xy=(cum_count.iloc[-1] * 0.9, ax.get_ylim()[1] * 0.9),
                   xytext=(cum_count.iloc[-1] * 0.1, ax.get_ylim()[1] * 0.9),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=2))
        ax.text(cum_count.iloc[-1] * 0.5, ax.get_ylim()[1] * 0.92, 
               'PAVA Progress →', ha='center', fontsize=10, color='gray', alpha=0.7)
    
    # Styling
    ax.set_xlabel(f'Cumulative Count (ordered by {x_column})', fontsize=11)
    ax.set_ylabel(f'Cumulative Mean of {y_column}', fontsize=11)
    
    if title is None:
        title = 'PAVA Algorithm Process: Merging and Anchor Formation'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Remove grid
    ax.grid(False)
    
    # Legend
    if show_legend:
        # Add custom legend entries
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=initial_color,
                   markersize=8, alpha=0.5, label='Initial groups'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=fallback_color,
                   markersize=10, label='Merging steps'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor=anchor_color,
                   markersize=12, label='Completed bins')
        ]
        ax.legend(handles=legend_elements, loc='best', frameon=True, 
                 fancybox=False, shadow=False)
    
    return ax


def plot_pava_animation(
    pava: PAVA,
    *,
    interval: int = 1000,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
    show_cumulative: bool = True,
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
        show_cumulative: Whether to show cumulative mean plot.
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
    """Plot side-by-side comparison of PAVA process visualizations.
    
    Shows three complementary views:
    1. Cumulative mean evolution
    2. Group means with PAVA fit
    3. PAVA merging process
    
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
    fig = plt.figure(figsize=figsize)
    
    # Use GridSpec for better control
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.2])
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    
    # Plot 1: Cumulative Mean
    plot_cumulative_mean(
        groups_df,
        blocks,
        ax=ax1,
        title='Cumulative Mean Evolution',
        show_legend=True,
        x_column=binner.x
    )
    
    # Plot 2: Group Means vs PAVA Fit
    plot_gcm(
        groups_df,
        blocks,
        ax=ax2,
        title='Group Means vs PAVA Fit',
        show_legend=True,
        x_column=binner.x,
        y_column=binner.y
    )
    
    # Plot 3: PAVA Process
    plot_pava_process(
        groups_df,
        blocks,
        ax=ax3,
        title='PAVA Merging Process',
        show_legend=True,
        x_column=binner.x,
        y_column=binner.y,
        show_annotations=True
    )
    
    # Overall title
    if title is None:
        direction = "increasing" if binner.resolved_sign_ == "+" else "decreasing"
        title = f'PAVA Analysis: {binner.x} vs {binner.y} ({direction} monotonicity)'
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig