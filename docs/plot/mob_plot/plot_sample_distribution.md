# plot_sample_distribution Function Documentation

## Overview
The `plot_sample_distribution` function creates visualizations of how samples are distributed across bins, showing both absolute counts and percentages. It helps identify data concentration, sparse bins, and ensures adequate representation in each bin.

## Function Signature
```python
def plot_sample_distribution(
    summary_df: pd.DataFrame,
    *,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = None,
    plot_type: str = "bar",
    color: str = "#1E88E5",
    show_values: bool = True,
    value_type: str = "both",
    show_cumulative: bool = False,
    cumulative_color: str = "#E53935",
    show_grid: bool = True,
    grid_alpha: float = 0.3,
    orientation: str = "vertical",
    show_threshold: bool = True,
    threshold_value: float = 0.05,
    threshold_color: str = "#FFA726",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    rotation: int = 45,
    show_statistics: bool = True
) -> Axes
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **summary_df** | `pd.DataFrame` | required | Binning summary with count data |
| **ax** | `Optional[Axes]` | `None` | Matplotlib axes to plot on |
| **figsize** | `Tuple[float, float]` | `(12, 6)` | Figure size if creating new figure |
| **title** | `Optional[str]` | `None` | Plot title |
| **plot_type** | `str` | `"bar"` | Type of plot ('bar', 'hist', 'pie', 'area') |
| **color** | `str` | `"#1E88E5"` | Primary color for plot |
| **show_values** | `bool` | `True` | Display values on plot |
| **value_type** | `str` | `"both"` | Values to show ('count', 'percent', 'both') |
| **show_cumulative** | `bool` | `False` | Add cumulative distribution line |
| **cumulative_color** | `str` | `"#E53935"` | Color for cumulative line |
| **show_grid** | `bool` | `True` | Show grid lines |
| **grid_alpha** | `float` | `0.3` | Grid transparency |
| **orientation** | `str` | `"vertical"` | Bar orientation ('vertical', 'horizontal') |
| **show_threshold** | `bool` | `True` | Show minimum sample threshold |
| **threshold_value** | `float` | `0.05` | Minimum percentage threshold (5%) |
| **threshold_color** | `str` | `"#FFA726"` | Color for threshold line |
| **xlabel** | `Optional[str]` | `None` | X-axis label |
| **ylabel** | `Optional[str]` | `None` | Y-axis label |
| **rotation** | `int` | `45` | X-tick label rotation |
| **show_statistics** | `bool` | `True` | Display distribution statistics |

## Returns
- **Axes**: Matplotlib Axes object containing the plot

## Usage Examples

### Basic Bar Chart
```python
from MOBPY.plot import plot_sample_distribution

# After fitting binner
summary = binner.summary_()

ax = plot_sample_distribution(summary)
plt.show()
```

### With Cumulative Distribution
```python
fig, ax = plt.subplots(figsize=(14, 7))

plot_sample_distribution(
    summary,
    ax=ax,
    show_cumulative=True,
    title="Sample Distribution with Cumulative Percentage",
    value_type="both",
    show_threshold=True,
    threshold_value=0.05
)

plt.tight_layout()
plt.show()
```

### Horizontal Bar Chart
```python
ax = plot_sample_distribution(
    summary,
    orientation="horizontal",
    figsize=(8, 10),
    show_values=True,
    value_type="percent"
)

# Adjust layout for horizontal
ax.set_xlabel("Percentage of Samples")
ax.set_ylabel("Bins")
```

### Pie Chart Distribution
```python
ax = plot_sample_distribution(
    summary,
    plot_type="pie",
    figsize=(10, 10),
    title="Sample Distribution Across Bins"
)
```

## Plot Types

### Bar Chart (Default)
```python
ax = plot_sample_distribution(summary, plot_type="bar")
```
- Most common visualization
- Shows absolute or relative frequencies
- Easy comparison between bins

### Histogram
```python
ax = plot_sample_distribution(summary, plot_type="hist")
```
- Similar to bar but with connected bins
- Better for continuous feel
- Shows distribution shape

### Pie Chart
```python
ax = plot_sample_distribution(summary, plot_type="pie")
```
- Shows proportions clearly
- Good for reports
- Limited to <10 bins for clarity

### Area Chart
```python
ax = plot_sample_distribution(summary, plot_type="area")
```
- Shows distribution as filled area
- Good for highlighting concentration
- Works well with cumulative

## Advanced Features

### Highlighting Sparse Bins
```python
ax = plot_sample_distribution(summary)

# Highlight bins with low samples
min_samples = 30
for i, (idx, row) in enumerate(summary.iterrows()):
    if row['count'] < min_samples:
        if hasattr(ax, 'patches') and i < len(ax.patches):
            ax.patches[i].set_facecolor('red')
            ax.patches[i].set_alpha(0.5)
            
            # Add warning annotation
            ax.annotate('⚠ Low Sample', 
                       xy=(i, row['count']), 
                       xytext=(0, 5),
                       textcoords='offset points',
                       fontsize=8, 
                       color='red',
                       ha='center')
```

### Distribution Statistics
```python
ax = plot_sample_distribution(summary, show_statistics=True)

# Add statistical information
stats_text = f"""
Distribution Statistics:
• Mean: {summary['count'].mean():.0f}
• Median: {summary['count'].median():.0f}
• Std: {summary['count'].std():.0f}
• CV: {summary['count'].std()/summary['count'].mean():.2f}
• Gini: {calculate_gini(summary['count']):.3f}
"""

ax.text(0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
```

### Comparing Distributions
```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Training distribution
plot_sample_distribution(
    train_summary,
    ax=axes[0],
    title="Training Set Distribution",
    color='blue',
    show_cumulative=True
)

# Test distribution
plot_sample_distribution(
    test_summary,
    ax=axes[1],
    title="Test Set Distribution",
    color='green',
    show_cumulative=True
)

# Calculate similarity
from scipy.stats import wasserstein_distance
distance = wasserstein_distance(
    train_summary['count'], 
    test_summary['count']
)
fig.suptitle(f'Distribution Comparison (Wasserstein Distance: {distance:.3f})', 
            fontsize=14, fontweight='bold')

plt.tight_layout()
```

## Custom Styling

### Business Dashboard Style
```python
fig, ax = plt.subplots(figsize=(14, 7))

plot_sample_distribution(
    summary,
    ax=ax,
    plot_type="bar",
    color='#2E7D32',
    show_values=True,
    value_type="both",
    show_cumulative=True,
    cumulative_color='#1976D2',
    show_grid=False
)

# Add gradient effect
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

colors = ['#E8F5E9', '#2E7D32']
n_bins = len(summary)
cmap = LinearSegmentedColormap.from_list('gradient', colors, N=n_bins)

for i, patch in enumerate(ax.patches):
    patch.set_facecolor(cmap(i/n_bins))

ax.set_title('Customer Distribution Analysis', fontsize=16, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

### Publication Style
```python
fig, ax = plt.subplots(figsize=(10, 6))

plot_sample_distribution(
    summary,
    ax=ax,
    plot_type="bar",
    color='black',
    show_values=False,
    show_cumulative=True,
    cumulative_color='gray',
    show_grid=True,
    grid_alpha=0.2,
    show_threshold=False
)

# Add pattern fill
for i, patch in enumerate(ax.patches):
    if i % 2 == 0:
        patch.set_hatch('///')
    else:
        patch.set_hatch('\\\\\\')

ax.set_title('Figure 2: Sample Distribution', fontsize=12)
ax.set_xlabel('Bins', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.tick_params(labelsize=10)
```

## Interactive Features

### Hover Information
```python
import mplcursors

ax = plot_sample_distribution(summary)

cursor = mplcursors.cursor(ax.patches, hover=True)

@cursor.connect("add")
def on_add(sel):
    idx = sel.index
    row = summary.iloc[idx]
    sel.annotation.set_text(
        f"Bin: {row['bucket']}\n"
        f"Count: {row['count']:,}\n"
        f"Percentage: {row['count_pct']:.1f}%\n"
        f"Cumulative: {summary['count_pct'].iloc[:idx+1].sum():.1f}%"
    )
```

### Dynamic Threshold Adjustment
```python
from matplotlib.widgets import Slider

fig, ax = plt.subplots(figsize=(14, 8))
plt.subplots_adjust(bottom=0.2)

# Initial plot
plot = plot_sample_distribution(summary, ax=ax, show_threshold=True)

# Add slider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Threshold', 0.01, 0.2, valinit=0.05)

# Threshold line
threshold_line = ax.axhline(y=len(summary) * 0.05, 
                           color='red', linestyle='--', alpha=0.5)

def update(val):
    threshold = slider.val
    threshold_line.set_ydata([len(summary) * threshold])
    
    # Update bar colors
    total_count = summary['count'].sum()
    for i, patch in enumerate(ax.patches):
        if summary.iloc[i]['count'] / total_count < threshold:
            patch.set_facecolor('red')
            patch.set_alpha(0.5)
        else:
            patch.set_facecolor('#1E88E5')
            patch.set_alpha(1.0)
    
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()
```

## Quality Metrics

### Calculating Distribution Quality
```python
def assess_distribution_quality(summary_df):
    """Assess quality of sample distribution."""
    counts = summary_df['count'].values
    total = counts.sum()
    
    metrics = {
        'min_pct': counts.min() / total,
        'max_pct': counts.max() / total,
        'cv': counts.std() / counts.mean(),
        'gini': calculate_gini(counts),
        'sparse_bins': (counts < 30).sum(),
        'concentration': counts.max() / total
    }
    
    # Quality score
    quality_score = 100
    if metrics['min_pct'] < 0.02:
        quality_score -= 20  # Very sparse bin
    if metrics['cv'] > 1.0:
        quality_score -= 15  # High imbalance
    if metrics['sparse_bins'] > 0:
        quality_score -= 10 * metrics['sparse_bins']
    if metrics['concentration'] > 0.5:
        quality_score -= 25  # Over-concentrated
    
    return max(0, quality_score), metrics

score, metrics = assess_distribution_quality(summary)
print(f"Distribution Quality Score: {score}/100")
```

## Common Issues and Solutions

### Issue: Uneven Bar Widths
```python
# Solution: Use equal width bars
ax = plot_sample_distribution(summary)
for i, patch in enumerate(ax.patches):
    patch.set_width(0.8)  # Equal width
    patch.set_x(i - 0.4)  # Center bars
```

### Issue: Labels Overlapping
```python
# Solution 1: Rotate more
plot_sample_distribution(summary, rotation=90)

# Solution 2: Use horizontal
plot_sample_distribution(summary, orientation="horizontal")

# Solution 3: Abbreviate labels
summary_copy = summary.copy()
summary_copy['bucket'] = summary_copy['bucket'].apply(
    lambda x: x.replace('[-inf,', '<').replace(', +inf)', '+')
)
plot_sample_distribution(summary_copy)
```

### Issue: Values Not Visible
```python
# Solution: Adjust text position based on bar height
ax = plot_sample_distribution(summary, show_values=False)

for i, (idx, row) in enumerate(summary.iterrows()):
    height = row['count']
    
    # Place inside if bar is tall enough
    if height > summary['count'].max() * 0.1:
        va = 'center'
        y_pos = height / 2
        color = 'white'
    else:
        va = 'bottom'
        y_pos = height
        color = 'black'
    
    ax.text(i, y_pos, f'{height:,.0f}', 
           ha='center', va=va, color=color, fontweight='bold')
```

## See Also
- [`plot_event_rate`](./plot_event_rate.md) - Combined with event rate
- [`plot_bin_statistics`](./plot_bin_statistics.md) - Part of comprehensive view
- [`plot_bin_boundaries`](./plot_bin_boundaries.md) - Distribution with boundaries
- [`MonotonicBinner`](../binning/mob.md) - Main binning class