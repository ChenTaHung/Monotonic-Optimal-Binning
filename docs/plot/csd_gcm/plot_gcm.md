# `plot_gcm` Function Documentation

## Overview
The `plot_gcm` function visualizes the Greatest Convex Minorant (GCM) produced by the PAVA algorithm. It shows how the algorithm creates a monotonic step function from non-monotonic data by pooling adjacent violators.

## Function Signature
```python
def plot_gcm(
    groups_df: pd.DataFrame,
    blocks: List[Dict[str, Any]],
    *,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    show_points: bool = True,
    show_steps: bool = True,
    show_shaded: bool = True,
    point_color: str = "#1E88E5",
    step_color: str = "#E53935", 
    shade_color: str = "#FFA726",
    point_size: float = 60,
    line_width: float = 2.5,
    shade_alpha: float = 0.15,
    show_legend: bool = True,
    x_column: str = 'x',
    y_column: str = 'mean',
    show_grid: bool = True,
    grid_alpha: float = 0.3
) -> Axes
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **groups_df** | `pd.DataFrame` | required | DataFrame from PAVA.groups_ with columns ['x', 'count', 'mean', 'sum'] |
| **blocks** | `List[Dict]` | required | List of block dictionaries from PAVA.export_blocks(as_dict=True) |
| **ax** | `Optional[Axes]` | `None` | Matplotlib axes to plot on. If None, creates new figure |
| **figsize** | `Tuple[float, float]` | `(12, 6)` | Figure size (width, height) if creating new figure |
| **title** | `Optional[str]` | `None` | Plot title. If None, uses "Greatest Convex Minorant (GCM)" |
| **subtitle** | `Optional[str]` | `None` | Plot subtitle for additional context |
| **show_points** | `bool` | `True` | Whether to show original group mean points |
| **show_steps** | `bool` | `True` | Whether to show the GCM step function |
| **show_shaded** | `bool` | `True` | Whether to shade the pooled regions |
| **point_color** | `str` | `"#1E88E5"` | Color for original data points (vivid blue) |
| **step_color** | `str` | `"#E53935"` | Color for GCM step function (vivid red) |
| **shade_color** | `str` | `"#FFA726"` | Color for shaded pooled regions (orange) |
| **point_size** | `float` | `60` | Size of scatter points |
| **line_width** | `float` | `2.5` | Width of step function lines |
| **shade_alpha** | `float` | `0.15` | Transparency of shaded regions |
| **show_legend** | `bool` | `True` | Whether to display legend |
| **x_column** | `str` | `'x'` | Name of x column for axis label |
| **y_column** | `str` | `'mean'` | Name of y column for axis label |
| **show_grid** | `bool` | `True` | Whether to show grid lines |
| **grid_alpha** | `float` | `0.3` | Grid transparency |

## Returns
- **Axes**: Matplotlib Axes object containing the plot

## Usage Examples

### Basic Usage
```python
from MOBPY.plot.csd_gcm import plot_gcm
from MOBPY import MonotonicBinner

# Fit binner
binner = MonotonicBinner(df, x='age', y='default')
binner.fit()

# Get PAVA results
groups_df = binner._pava.groups_
blocks = binner._pava.export_blocks(as_dict=True)

# Create GCM plot
ax = plot_gcm(groups_df, blocks)
plt.show()
```

### Custom Styling
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 7))

plot_gcm(
    groups_df, blocks,
    ax=ax,
    title="PAVA Result: Monotonic Fit",
    subtitle="Age vs Default Rate",
    point_color='navy',
    step_color='darkred',
    shade_color='gold',
    shade_alpha=0.2,
    line_width=3
)

ax.set_xlabel('Age (years)', fontsize=12)
ax.set_ylabel('Default Rate', fontsize=12)
plt.tight_layout()
plt.show()
```

### Without Shading
```python
# Clean plot showing only points and steps
ax = plot_gcm(
    groups_df, blocks,
    show_shaded=False,
    show_legend=True,
    title="GCM: Monotonic Step Function"
)
```

### Minimal Plot
```python
# Show only the step function
ax = plot_gcm(
    groups_df, blocks,
    show_points=False,
    show_shaded=False,
    show_legend=False,
    show_grid=False
)
```

## Visual Elements

### 1. Original Points
- Blue scatter points showing the mean of each unique x value
- Size indicates relative importance
- Connected by faint lines to show original trend

### 2. Step Function (GCM)
- Red horizontal lines showing the pooled monotonic values
- Vertical lines connect steps
- Width of each step corresponds to the pooled region

### 3. Shaded Regions
- Orange shaded areas indicate where pooling occurred
- Darker shading suggests more aggressive pooling
- No shading means the original point was already monotonic

## Interpretation Guide

### Reading the Plot
1. **Blue Points**: Original group means (potentially non-monotonic)
2. **Red Steps**: Final monotonic values after PAVA
3. **Shaded Areas**: Regions where violators were pooled

### What to Look For
- **Large Shaded Regions**: Indicate significant pooling was needed
- **Many Small Steps**: Data was already nearly monotonic
- **Few Large Steps**: Data had strong violations requiring aggressive pooling
- **Step Heights**: Show the final monotonic relationship

## Advanced Features

### Adding Annotations
```python
ax = plot_gcm(groups_df, blocks)

# Annotate specific blocks
for i, block in enumerate(blocks):
    if block['n'] > 100:  # Annotate large blocks
        ax.annotate(
            f"n={block['n']}", 
            xy=(block['right'], block['mean']),
            xytext=(5, 5), 
            textcoords='offset points'
        )
```

### Combining with Other Plots
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# GCM on left
plot_gcm(groups_df, blocks, ax=ax1, title="GCM")

# Original data on right
ax2.scatter(df['x'], df['y'], alpha=0.5)
ax2.set_title("Raw Data")

plt.tight_layout()
```

## Performance Notes
- Efficiently handles up to 1000 unique x values
- Automatically adjusts point density for large datasets
- Uses vectorized operations for speed

## Common Issues and Solutions

### Issue: Overlapping Points
```python
# Solution: Adjust point size or use transparency
plot_gcm(groups_df, blocks, point_size=30, point_alpha=0.7)
```

### Issue: Too Many Steps
```python
# Solution: Focus on regions of interest
ax = plot_gcm(groups_df, blocks)
ax.set_xlim(20, 60)  # Zoom to age 20-60
```

### Issue: Legend Covering Data
```python
# Solution: Move legend outside
ax = plot_gcm(groups_df, blocks)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
```

## Mathematical Background

The GCM represents the solution to:
```
minimize Σ(yi - μi)²
subject to μ1 ≤ μ2 ≤ ... ≤ μn
```

The plot visualizes this optimization by showing:
- Original values (potentially violating monotonicity)
- Final values (satisfying monotonicity)
- Pooled regions (where constraint was active)

## See Also
- [`plot_pava_comparison`](./plot_pava_comparison.md) - Side-by-side CSD and GCM
- [`PAVA`](../core/pava.md) - Algorithm details
- [`MonotonicBinner`](../binning/mob.md) - Main binning class
