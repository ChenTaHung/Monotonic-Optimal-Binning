# `plot_woe_bars` Function Documentation

## Overview
The `plot_woe_bars` function creates a bar chart visualization of Weight of Evidence (WoE) values across bins for binary classification problems. It helps assess the predictive power of each bin and the monotonic relationship with the target variable.

## Function Signature
```python
def plot_woe_bars(
    summary_df: pd.DataFrame,
    *,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = None,
    bar_color: str = "#1E88E5",
    positive_color: str = "#43A047",
    negative_color: str = "#E53935",
    show_values: bool = True,
    value_format: str = ".3f",
    show_grid: bool = True,
    grid_alpha: float = 0.3,
    bar_width: float = 0.8,
    show_iv: bool = True,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    rotation: int = 45
) -> Axes
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **summary_df** | `pd.DataFrame` | required | Binning summary with 'bucket' and 'woe' columns |
| **ax** | `Optional[Axes]` | `None` | Matplotlib axes to plot on |
| **figsize** | `Tuple[float, float]` | `(12, 6)` | Figure size if creating new figure |
| **title** | `Optional[str]` | `None` | Plot title. Auto-generated if None |
| **bar_color** | `str` | `"#1E88E5"` | Default bar color (blue) |
| **positive_color** | `str` | `"#43A047"` | Color for positive WoE (green) |
| **negative_color** | `str` | `"#E53935"` | Color for negative WoE (red) |
| **show_values** | `bool` | `True` | Display WoE values on bars |
| **value_format** | `str` | `".3f"` | Format string for values |
| **show_grid** | `bool` | `True` | Show grid lines |
| **grid_alpha** | `float` | `0.3` | Grid transparency |
| **bar_width** | `float` | `0.8` | Width of bars (0-1) |
| **show_iv** | `bool` | `True` | Show total IV in title |
| **xlabel** | `Optional[str]` | `None` | X-axis label |
| **ylabel** | `Optional[str]` | `None` | Y-axis label |
| **rotation** | `int` | `45` | X-tick label rotation |

## Returns
- **Axes**: Matplotlib Axes object containing the plot

## Usage Examples

### Basic Usage
```python
from MOBPY.plot import plot_woe_bars

# After fitting binner on binary target
summary = binner.summary_()

ax = plot_woe_bars(summary)
plt.show()
```

### Custom Styling
```python
fig, ax = plt.subplots(figsize=(14, 7))

plot_woe_bars(
    summary,
    ax=ax,
    title="Credit Risk: Weight of Evidence by Age Group",
    positive_color='darkgreen',
    negative_color='darkred',
    bar_width=0.6,
    show_values=True,
    value_format=".2f"
)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()
```

### Highlighting Significant Bins
```python
ax = plot_woe_bars(summary)

# Highlight bins with high absolute WoE
for i, (idx, row) in enumerate(summary.iterrows()):
    if abs(row['woe']) > 0.5:
        ax.patches[i].set_edgecolor('black')
        ax.patches[i].set_linewidth(2)
        ax.annotate('High Impact', 
                   xy=(i, row['woe']), 
                   xytext=(5, 10),
                   textcoords='offset points',
                   fontsize=8, 
                   style='italic')
```

### Combined with IV Information
```python
ax = plot_woe_bars(summary, show_iv=True)

# Add IV contribution for each bin
for i, (idx, row) in enumerate(summary.iterrows()):
    if 'iv' in row:
        ax.text(i, row['woe']/2, f"IV: {row['iv']:.3f}", 
               ha='center', va='center', fontsize=8)
```

## Visual Interpretation

### WoE Values
- **Positive WoE** (green): Good rate > Bad rate (lower risk)
- **Negative WoE** (red): Good rate < Bad rate (higher risk)
- **Zero WoE**: Good rate = Bad rate (neutral)
- **Magnitude**: Larger absolute values indicate stronger predictive power

### Monotonicity Check
- Bars should show consistent trend (all increasing or all decreasing)
- Mixed directions suggest non-monotonic relationship

### Information Value
- Total IV shown in title indicates overall predictive power
- IV < 0.1: Weak predictor
- IV 0.1-0.3: Medium predictor
- IV > 0.3: Strong predictor

## Advanced Features

### Conditional Coloring
```python
# Color based on WoE magnitude
def get_color(woe):
    if abs(woe) < 0.1:
        return 'gray'
    elif woe > 0:
        return 'green'
    else:
        return 'red'

ax = plot_woe_bars(summary)
for i, (idx, row) in enumerate(summary.iterrows()):
    ax.patches[i].set_facecolor(get_color(row['woe']))
```

### Adding Reference Lines
```python
ax = plot_woe_bars(summary)

# Add significance thresholds
ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Strong positive')
ax.axhline(y=-0.3, color='red', linestyle='--', alpha=0.5, label='Strong negative')
ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Neutral')

ax.legend(loc='upper left')
```

### Annotating with Statistics
```python
ax = plot_woe_bars(summary)

# Add sample information
for i, (idx, row) in enumerate(summary.iterrows()):
    count_pct = row.get('count_pct', 0)
    ax.text(i, -0.5, f"{count_pct:.1f}%", 
           ha='center', fontsize=8, color='gray')

ax.text(0.5, -0.6, 'Sample %', transform=ax.transAxes, 
        ha='center', fontsize=9, color='gray')
```

## Customization Examples

### Publication-Ready
```python
fig, ax = plt.subplots(figsize=(10, 5))

plot_woe_bars(
    summary,
    ax=ax,
    bar_color='#2E4057',
    show_grid=True,
    grid_alpha=0.2,
    show_values=True,
    value_format=".2f",
    rotation=0  # Horizontal labels
)

ax.set_title('Weight of Evidence Analysis', fontsize=14, fontweight='bold')
ax.set_xlabel('Feature Bins', fontsize=12)
ax.set_ylabel('WoE', fontsize=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
```

### Interactive Tooltips (for Jupyter)
```python
import mplcursors

ax = plot_woe_bars(summary)

cursor = mplcursors.cursor(ax.patches, hover=True)

@cursor.connect("add")
def on_add(sel):
    idx = sel.index
    row = summary.iloc[idx]
    sel.annotation.set_text(
        f"Bin: {row['bucket']}\n"
        f"WoE: {row['woe']:.3f}\n"
        f"Count: {row.get('count', 'N/A')}\n"
        f"Bad Rate: {row.get('mean', 'N/A'):.2%}"
    )
```

## Common Issues and Solutions

### Issue: Overlapping X-axis Labels
```python
# Solution 1: Rotate labels more
plot_woe_bars(summary, rotation=90)

# Solution 2: Shorten labels
summary_copy = summary.copy()
summary_copy['bucket'] = summary_copy['bucket'].str.replace('[-inf,', '[<', regex=False)
plot_woe_bars(summary_copy)
```

### Issue: Values Not Visible
```python
# Solution: Adjust text position
ax = plot_woe_bars(summary, show_values=False)

for i, (idx, row) in enumerate(summary.iterrows()):
    y_pos = row['woe'] + (0.05 if row['woe'] > 0 else -0.05)
    ax.text(i, y_pos, f"{row['woe']:.2f}", 
           ha='center', va='bottom' if row['woe'] > 0 else 'top',
           fontsize=9)
```

## Performance Notes
- Handles up to 20 bins efficiently
- Automatically adjusts layout for many bins
- Caches color calculations for speed

## Integration with Other Plots
```python
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# WoE bars on top
plot_woe_bars(summary, ax=ax1)

# Sample distribution below
plot_sample_distribution(summary, ax=ax2)

plt.tight_layout()
```

## See Also
- [`plot_event_rate`](./plot_event_rate.md) - Event rate visualization
- [`plot_bin_statistics`](./plot_bin_statistics.md) - Comprehensive bin analysis
- [`MonotonicBinner`](../binning/mob.md) - Main binning class
- [`woe_iv`](../core/utils.md#woe_iv) - WoE/IV calculation details
