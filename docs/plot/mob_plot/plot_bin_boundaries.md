# `plot_bin_boundaries` Function Documentation

## Overview
The `plot_bin_boundaries` function visualizes bin boundaries overlaid on the feature distribution, showing how the binning algorithm partitioned the continuous feature space. It helps understand the relationship between bin cuts and the underlying data density.

## Function Signature
```python
def plot_bin_boundaries(
    df: pd.DataFrame,
    binner: 'MonotonicBinner',
    *,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (14, 7),
    title: Optional[str] = None,
    hist_color: str = "#B3E5FC",
    boundary_color: str = "#E53935",
    boundary_style: str = "--",
    boundary_width: float = 2.0,
    hist_bins: int = 50,
    hist_alpha: float = 0.7,
    density: bool = True,
    kde: bool = True,
    kde_color: str = "#1565C0",
    show_stats: bool = True,
    show_labels: bool = True,
    label_position: str = "top",
    show_rug: bool = False,
    rug_alpha: float = 0.3,
    show_legend: bool = True,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_excluded: bool = True
) -> Axes
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **df** | `pd.DataFrame` | required | Original dataframe with feature data |
| **binner** | `MonotonicBinner` | required | Fitted MonotonicBinner instance |
| **ax** | `Optional[Axes]` | `None` | Matplotlib axes to plot on |
| **figsize** | `Tuple[float, float]` | `(14, 7)` | Figure size if creating new figure |
| **title** | `Optional[str]` | `None` | Plot title |
| **hist_color** | `str` | `"#B3E5FC"` | Color for histogram (light blue) |
| **boundary_color** | `str` | `"#E53935"` | Color for boundary lines (red) |
| **boundary_style** | `str` | `"--"` | Line style for boundaries |
| **boundary_width** | `float` | `2.0` | Width of boundary lines |
| **hist_bins** | `int` | `50` | Number of histogram bins |
| **hist_alpha** | `float` | `0.7` | Histogram transparency |
| **density** | `bool` | `True` | Show density instead of count |
| **kde** | `bool` | `True` | Overlay kernel density estimate |
| **kde_color** | `str` | `"#1565C0"` | Color for KDE curve (dark blue) |
| **show_stats** | `bool` | `True` | Display distribution statistics |
| **show_labels** | `bool` | `True` | Label boundary values |
| **label_position** | `str` | `"top"` | Position of labels ('top', 'bottom', 'alternate') |
| **show_rug** | `bool` | `False` | Show rug plot of actual values |
| **rug_alpha** | `float` | `0.3` | Rug plot transparency |
| **show_legend** | `bool` | `True` | Display legend |
| **xlabel** | `Optional[str]` | `None` | X-axis label |
| **ylabel** | `Optional[str]` | `None` | Y-axis label |
| **show_excluded** | `bool` | `True` | Show excluded values if present |

## Returns
- **Axes**: Matplotlib Axes object containing the plot

## Usage Examples

### Basic Usage
```python
from MOBPY.plot import plot_bin_boundaries

# After fitting binner
ax = plot_bin_boundaries(df, binner)
plt.show()
```

### With KDE and Rug Plot
```python
fig, ax = plt.subplots(figsize=(16, 8))

plot_bin_boundaries(
    df, binner,
    ax=ax,
    title="Feature Distribution with Optimal Bin Boundaries",
    kde=True,
    show_rug=True,
    rug_alpha=0.1,
    show_stats=True
)

plt.tight_layout()
plt.show()
```

### Highlighting Specific Regions
```python
ax = plot_bin_boundaries(df, binner)

# Highlight problematic regions
problem_regions = [(10, 20), (45, 50)]  # Feature value ranges

for start, end in problem_regions:
    ax.axvspan(start, end, alpha=0.2, color='red', 
               label='Problem Region' if start == problem_regions[0][0] else "")

ax.legend()
```

### Custom Boundary Styling
```python
ax = plot_bin_boundaries(
    df, binner,
    boundary_color='darkgreen',
    boundary_style='-',
    boundary_width=3,
    hist_color='lightgray',
    kde_color='darkblue'
)

# Add boundary annotations
bins_df = binner.bins_()
for i, row in bins_df.iterrows():
    if not np.isinf(row['right']):
        ax.annotate(f'Bin {i+1}', 
                   xy=(row['right'], 0),
                   xytext=(row['right'], ax.get_ylim()[1]*0.8),
                   arrowprops=dict(arrowstyle='->', color='gray'),
                   ha='center')
```

## Visual Components

### Histogram
- Shows empirical distribution of feature values
- Reveals data concentration and gaps
- Helps identify multimodality

### KDE Curve
- Smooth estimate of probability density
- Shows underlying distribution shape
- Helps identify local peaks/valleys

### Boundary Lines
- Vertical lines showing bin cutpoints
- First boundary is implicit at -∞
- Last boundary is implicit at +∞

### Rug Plot
- Individual data points along x-axis
- Shows actual data density
- Reveals gaps and clusters

## Advanced Features

### Multi-Dataset Comparison
```python
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Training data
plot_bin_boundaries(
    train_df, binner,
    ax=axes[0],
    title="Training Data Distribution",
    hist_color='lightblue'
)

# Test data with same boundaries
plot_bin_boundaries(
    test_df, binner,
    ax=axes[1],
    title="Test Data Distribution",
    hist_color='lightgreen'
)

# Check distribution shift
from scipy.stats import ks_2samp
ks_stat, p_value = ks_2samp(train_df[feature], test_df[feature])
fig.suptitle(f'Distribution Comparison (KS test p-value: {p_value:.3f})', 
            fontsize=14)

plt.tight_layout()
```

### Annotating with Bin Statistics
```python
ax = plot_bin_boundaries(df, binner, kde=False)

# Add bin statistics
bins_df = binner.bins_()
summary_df = binner.summary_()

for i, (bin_row, sum_row) in enumerate(zip(bins_df.iterrows(), summary_df.iterrows())):
    if not np.isinf(bin_row[1]['right']):
        x_pos = bin_row[1]['right']
        
        # Add vertical line with stats
        ax.axvline(x=x_pos, color='red', linestyle='--', alpha=0.7)
        
        # Annotate with count and event rate
        stats_text = f"n={sum_row[1]['count']}\nrate={sum_row[1]['mean']:.2%}"
        ax.text(x_pos, ax.get_ylim()[1]*0.9, stats_text,
               ha='center', fontsize=8, 
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='yellow', alpha=0.5))
```

### Density Heatmap Overlay
```python
import numpy as np
from scipy.stats import gaussian_kde

ax = plot_bin_boundaries(df, binner, kde=False, hist_alpha=0.3)

# Create density heatmap
x_feature = df[binner.x].dropna()
kde_func = gaussian_kde(x_feature)

# Create grid for heatmap
x_grid = np.linspace(x_feature.min(), x_feature.max(), 200)
density = kde_func(x_grid)

# Overlay heatmap
im = ax.imshow(density.reshape(1, -1), 
              extent=[x_grid.min(), x_grid.max(), 0, ax.get_ylim()[1]],
              aspect='auto', cmap='YlOrRd', alpha=0.6, origin='lower')

plt.colorbar(im, ax=ax, label='Density')
```

### Interactive Boundary Exploration
```python
from matplotlib.widgets import Slider, Button

fig, ax = plt.subplots(figsize=(14, 8))
plt.subplots_adjust(bottom=0.25)

# Initial plot
plot_bin_boundaries(df, binner, ax=ax)

# Store original boundaries
original_boundaries = binner.bins_()['right'].values[:-1]

# Add sliders for boundary adjustment
slider_axes = []
sliders = []

for i, boundary in enumerate(original_boundaries):
    if not np.isinf(boundary):
        ax_slider = plt.axes([0.1, 0.15 - i*0.03, 0.8, 0.02])
        slider = Slider(ax_slider, f'Boundary {i+1}', 
                       boundary - 10, boundary + 10, valinit=boundary)
        slider_axes.append(ax_slider)
        sliders.append(slider)

def update(val):
    ax.clear()
    # Redraw with new boundaries
    # (Implementation depends on specific requirements)
    fig.canvas.draw_idle()

for slider in sliders:
    slider.on_changed(update)

plt.show()
```

## Quality Assessment

### Identifying Good Boundaries
```python
def assess_boundary_quality(df, binner):
    """Assess quality of bin boundaries."""
    
    feature_values = df[binner.x].dropna()
    bins_df = binner.bins_()
    
    quality_metrics = []
    
    for i, row in bins_df.iterrows():
        left = row['left']
        right = row['right']
        
        # Get values in this bin
        mask = (feature_values >= left) & (feature_values < right)
        bin_values = feature_values[mask]
        
        if len(bin_values) > 1:
            # Calculate metrics
            metrics = {
                'bin': i,
                'density_gap': check_density_gap(bin_values),
                'homogeneity': bin_values.std() / bin_values.mean() if bin_values.mean() != 0 else 0,
                'separation': calculate_separation(bin_values, feature_values),
                'sample_adequacy': len(bin_values) > 30
            }
            quality_metrics.append(metrics)
    
    return pd.DataFrame(quality_metrics)

quality_df = assess_boundary_quality(df, binner)
print(quality_df)
```

### Visualizing Boundary Effectiveness
```python
ax = plot_bin_boundaries(df, binner)

# Color boundaries by effectiveness
bins_df = binner.bins_()
summary_df = binner.summary_()

for i, (bin_row, sum_row) in enumerate(zip(bins_df.iterrows(), summary_df.iterrows())):
    if not np.isinf(bin_row[1]['right']):
        x_pos = bin_row[1]['right']
        
        # Color based on IV contribution (for binary target)
        if 'iv' in sum_row[1]:
            iv_value = sum_row[1]['iv']
            if iv_value > 0.1:
                color = 'green'
                label = 'Strong'
            elif iv_value > 0.05:
                color = 'orange'
                label = 'Medium'
            else:
                color = 'red'
                label = 'Weak'
            
            ax.axvline(x=x_pos, color=color, linestyle='--', 
                      alpha=0.7, linewidth=2, label=f'{label} boundary')

# Remove duplicate labels
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())
```

## Common Issues and Solutions

### Issue: Boundaries Not Visible
```python
# Solution: Increase contrast
ax = plot_bin_boundaries(
    df, binner,
    hist_alpha=0.3,  # More transparent histogram
    boundary_color='black',  # High contrast
    boundary_width=3  # Thicker lines
)
```

### Issue: Overlapping Labels
```python
# Solution: Use alternating positions
ax = plot_bin_boundaries(df, binner, show_labels=False)

bins_df = binner.bins_()
for i, row in bins_df.iterrows():
    if not np.isinf(row['right']):
        y_pos = ax.get_ylim()[1] * (0.9 if i % 2 == 0 else 0.8)
        ax.text(row['right'], y_pos, f'{row["right"]:.2f}',
               ha='center', rotation=45, fontsize=8)
```

### Issue: Too Many Data Points
```python
# Solution: Sample for visualization
if len(df) > 10000:
    sample_df = df.sample(n=10000, random_state=42)
    ax = plot_bin_boundaries(sample_df, binner)
else:
    ax = plot_bin_boundaries(df, binner)
```

## Performance Optimization

### Large Datasets
```python
# Use fewer histogram bins and disable KDE
ax = plot_bin_boundaries(
    df, binner,
    hist_bins=30,  # Fewer bins
    kde=False,  # Disable expensive KDE
    show_rug=False  # No rug plot
)
```

### Caching KDE
```python
from functools import lru_cache

@lru_cache(maxsize=1)
def compute_kde(data_hash):
    from scipy.stats import gaussian_kde
    data = df[binner.x].dropna().values
    return gaussian_kde(data)

# Use cached KDE
data_hash = hash(tuple(df[binner.x].dropna().values[:100]))  # Sample for hash
kde_func = compute_kde(data_hash)
```

## See Also
- [`plot_sample_distribution`](./plot_sample_distribution.md) - Distribution without boundaries
- [`plot_bin_statistics`](./plot_bin_statistics.md) - Comprehensive view
- [`MonotonicBinner`](../binning/mob.md) - Main binning class
- [`PAVA`](../core/pava.md) - Algorithm creating boundaries