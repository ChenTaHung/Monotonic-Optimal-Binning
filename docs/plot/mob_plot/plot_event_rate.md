# `plot_event_rate` Function Documentation

## Overview
The `plot_event_rate` function creates a dual-axis visualization showing both the event rate (bad rate for binary classification) and sample distribution across bins. This helps identify bins with high risk and ensures adequate sample representation.

## Function Signature
```python
def plot_event_rate(
    summary_df: pd.DataFrame,
    *,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = None,
    bar_color: str = "#B3E5FC",
    line_color: str = "#E53935",
    point_color: str = "#C62828",
    bar_alpha: float = 0.7,
    line_width: float = 2.5,
    point_size: float = 80,
    show_values: bool = True,
    value_format: str = ".1%",
    show_grid: bool = True,
    grid_alpha: float = 0.3,
    show_legend: bool = True,
    xlabel: Optional[str] = None,
    y1_label: Optional[str] = None,
    y2_label: Optional[str] = None,
    rotation: int = 45,
    show_trend: bool = True
) -> Axes
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **summary_df** | `pd.DataFrame` | required | Binning summary with event rate and count data |
| **ax** | `Optional[Axes]` | `None` | Matplotlib axes to plot on |
| **figsize** | `Tuple[float, float]` | `(12, 6)` | Figure size if creating new figure |
| **title** | `Optional[str]` | `None` | Plot title |
| **bar_color** | `str` | `"#B3E5FC"` | Color for sample distribution bars (light blue) |
| **line_color** | `str` | `"#E53935"` | Color for event rate line (red) |
| **point_color** | `str` | `"#C62828"` | Color for event rate points (dark red) |
| **bar_alpha** | `float` | `0.7` | Transparency of bars |
| **line_width** | `float` | `2.5` | Width of event rate line |
| **point_size** | `float` | `80` | Size of event rate points |
| **show_values** | `bool` | `True` | Display values on plot |
| **value_format** | `str` | `".1%"` | Format for event rate values |
| **show_grid** | `bool` | `True` | Show grid lines |
| **grid_alpha** | `float` | `0.3` | Grid transparency |
| **show_legend** | `bool` | `True` | Display legend |
| **xlabel** | `Optional[str]` | `None` | X-axis label |
| **y1_label** | `Optional[str]` | `None` | Left y-axis label (count) |
| **y2_label** | `Optional[str]` | `None` | Right y-axis label (rate) |
| **rotation** | `int` | `45` | X-tick label rotation |
| **show_trend** | `bool` | `True` | Show trend line for event rate |

## Returns
- **Axes**: Primary axes object (secondary axes accessible via ax.right_ax)

## Usage Examples

### Basic Usage
```python
from MOBPY.plot import plot_event_rate

# After fitting binner
summary = binner.summary_()

ax = plot_event_rate(summary)
plt.show()
```

### Custom Styling
```python
fig, ax = plt.subplots(figsize=(14, 7))

plot_event_rate(
    summary,
    ax=ax,
    title="Risk Distribution: Default Rate and Sample Count by Age",
    bar_color='lightgray',
    line_color='darkred',
    point_color='red',
    bar_alpha=0.5,
    show_trend=True,
    value_format=".2%"
)

plt.tight_layout()
plt.show()
```

### Highlighting Risk Levels
```python
ax = plot_event_rate(summary)

# Add risk level zones
ax2 = ax.right_ax  # Access secondary axis
ax2.axhspan(0, 0.1, color='green', alpha=0.1, label='Low Risk')
ax2.axhspan(0.1, 0.3, color='yellow', alpha=0.1, label='Medium Risk')
ax2.axhspan(0.3, 1.0, color='red', alpha=0.1, label='High Risk')

ax2.legend(loc='upper left')
```

### Adding Statistical Information
```python
ax = plot_event_rate(summary)

# Add mean line
overall_rate = summary['mean'].mean()
ax2 = ax.right_ax
ax2.axhline(y=overall_rate, color='blue', linestyle='--', 
            alpha=0.7, label=f'Overall Rate: {overall_rate:.2%}')

# Add confidence bands
std = summary['mean'].std()
ax2.fill_between(range(len(summary)), 
                 overall_rate - std, overall_rate + std,
                 alpha=0.2, color='blue', label='±1 Std Dev')

ax2.legend(loc='upper right')
```

## Visual Components

### Primary Axis (Left)
- **Bars**: Sample count or percentage per bin
- **Scale**: Absolute count or percentage of total
- **Purpose**: Shows data distribution across bins

### Secondary Axis (Right)
- **Line**: Event rate (bad rate) across bins
- **Points**: Actual event rate values
- **Scale**: Rate from 0 to 1 (or 0% to 100%)
- **Purpose**: Shows risk level per bin

### Trend Indicators
- **Monotonic**: Line should be consistently increasing or decreasing
- **Volatility**: Smooth line indicates stable pattern
- **Outliers**: Sharp changes may indicate problematic bins

## Advanced Features

### Interactive Annotations
```python
ax = plot_event_rate(summary)

# Add hover information
for i, (idx, row) in enumerate(summary.iterrows()):
    # Annotate high-risk bins
    if row.get('mean', 0) > 0.3:
        ax.annotate(
            f"High Risk\n{row['count']} samples",
            xy=(i, row['count']),
            xytext=(5, 10),
            textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.3', 
                     facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', 
                          connectionstyle='arc3,rad=0')
        )
```

### Confidence Intervals
```python
import scipy.stats as stats

ax = plot_event_rate(summary)
ax2 = ax.right_ax

# Calculate confidence intervals for event rate
for i, (idx, row) in enumerate(summary.iterrows()):
    n = row['count']
    p = row['mean']
    
    # Wilson score interval
    ci_low, ci_high = stats.binomtest(
        int(p * n), n, p
    ).proportion_ci(confidence_level=0.95)
    
    # Plot error bars
    ax2.errorbar(i, p, yerr=[[p-ci_low], [ci_high-p]], 
                fmt='none', color='gray', alpha=0.5, capsize=3)
```

### Comparative Analysis
```python
fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Training data
plot_event_rate(train_summary, ax=axes[0])
axes[0].set_title('Training Set')

# Test data  
plot_event_rate(test_summary, ax=axes[1])
axes[1].set_title('Test Set')

plt.tight_layout()
```

## Customization Examples

### Business Presentation Style
```python
fig, ax = plt.subplots(figsize=(14, 7))

plot_event_rate(
    summary,
    ax=ax,
    bar_color='#E8E8E8',
    line_color='#D32F2F',
    point_color='#B71C1C',
    bar_alpha=0.8,
    show_values=True,
    show_grid=False
)

# Clean styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.right_ax.spines['top'].set_visible(False)

# Add title and labels
ax.set_title('Customer Risk Profile by Segment', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Customer Segment', fontsize=12)
ax.set_ylabel('Number of Customers', fontsize=12)
ax.right_ax.set_ylabel('Default Rate', fontsize=12)

plt.tight_layout()
```

### Academic Publication Style
```python
fig, ax = plt.subplots(figsize=(10, 6))

plot_event_rate(
    summary,
    ax=ax,
    bar_color='white',
    line_color='black',
    point_color='black',
    bar_alpha=1.0,
    line_width=1.5,
    point_size=40,
    show_values=False,
    show_grid=True,
    grid_alpha=0.2
)

# Add hatching to bars
for patch in ax.patches:
    patch.set_hatch('//')
    patch.set_edgecolor('black')
    patch.set_linewidth(0.5)

# Format for publication
ax.set_title('Figure 1: Event Rate Distribution', fontsize=12)
ax.tick_params(labelsize=10)
ax.right_ax.tick_params(labelsize=10)

plt.tight_layout()
```

## Common Patterns and Interpretation

### Ideal Pattern
- Monotonic event rate (consistently increasing/decreasing)
- Adequate sample size in each bin (>5% of total)
- Smooth transitions between bins

### Warning Signs
- **U-shaped curve**: Non-monotonic relationship
- **Sparse bins**: Very low sample count (<30 samples)
- **Rate jumps**: Large discontinuities in event rate
- **Edge effects**: Extreme rates in first/last bins

## Performance Considerations
- Dual-axis plotting may be slow for >50 bins
- Consider aggregating bins for large datasets
- Cache calculations for interactive updates

## Troubleshooting

### Issue: Overlapping Axes Labels
```python
# Solution: Adjust label positions
ax = plot_event_rate(summary)
ax.yaxis.set_label_coords(-0.1, 0.5)
ax.right_ax.yaxis.set_label_coords(1.1, 0.5)
```

### Issue: Event Rate Not Visible
```python
# Solution: Adjust secondary axis range
ax = plot_event_rate(summary)
ax.right_ax.set_ylim(0, max(summary['mean']) * 1.1)
```

### Issue: Legend Overlapping Data
```python
# Solution: Place legend outside
ax = plot_event_rate(summary)
ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
ax.right_ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
```

## Integration with Other Visualizations
```python
from MOBPY.plot import plot_event_rate, plot_woe_bars

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)

# Event rate on top
ax1 = fig.add_subplot(gs[0])
plot_event_rate(summary, ax=ax1)
ax1.set_title('Event Rate Analysis')

# WoE bars below
ax2 = fig.add_subplot(gs[1])
plot_woe_bars(summary, ax=ax2)
ax2.set_title('Weight of Evidence')

plt.tight_layout()
```

## See Also
- [`plot_woe_bars`](./plot_woe_bars.md) - WoE visualization
- [`plot_sample_distribution`](./plot_sample_distribution.md) - Detailed distribution analysis
- [`plot_bin_statistics`](./plot_bin_statistics.md) - Comprehensive statistics
- [`MonotonicBinner`](../binning/mob.md) - Main binning class