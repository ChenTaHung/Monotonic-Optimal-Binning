# Plot Module Documentation

## Overview
The plot module provides comprehensive visualization tools for monotonic optimal binning, including PAVA algorithm process visualization and binning result analysis.

## Module Location
`src/MOBPY/plot/`

## Quick Start

```python
from MOBPY import MonotonicBinner
from MOBPY.plot import plot_bin_statistics, plot_pava_comparison

# Fit binner
binner = MonotonicBinner(df, x='age', y='default')
binner.fit()

# Visualize PAVA process
fig1 = plot_pava_comparison(binner)

# Visualize final results
fig2 = plot_bin_statistics(binner)
plt.show()
```

## Available Functions

### PAVA Visualization (`csd_gcm.py`)
- `plot_gcm` - Greatest Convex Minorant showing monotonic fit
- `plot_pava_comparison` - Side-by-side CSD and GCM plots
- `plot_pava_process` - PAVA merging process visualization
- `plot_pava_animation` - Animated visualization of PAVA iterations

### MOB Result Visualization (`mob_plot.py`)
- `plot_woe_bars` - Weight of Evidence for binary targets
- `plot_event_rate` - Event rate and sample distribution
- `plot_bin_statistics` - Comprehensive multi-panel results
- `plot_sample_distribution` - Distribution of samples across bins
- `plot_bin_boundaries` - Bin cuts on feature distribution
- `plot_binning_stability` - Compare train vs test binning

For detailed documentation of each function, see:
- [PAVA Visualization Functions](./csd_gcm.md)
- [MOB Result Plot Functions](./mob_plot/)

## Common Parameters

Most plotting functions accept:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **ax** | `Axes` | `None` | Matplotlib axes to plot on |
| **figsize** | `tuple` | `(10, 6)` | Figure size if creating new |
| **title** | `str` | Auto | Plot title |
| **save_path** | `str` | `None` | Path to save figure |

## Customization Examples

### Custom Styling
```python
from MOBPY.plot import plot_woe_bars

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_facecolor('#f0f0f0')

plot_woe_bars(binner.summary_(), ax=ax, 
              bar_color='#2E86AB', line_color='#A23B72')

ax.set_title('Custom WoE Visualization', fontsize=16)
ax.grid(True, alpha=0.3)
```

### Multi-Panel Layout
```python
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
plot_woe_bars(summary, ax=ax1)

ax2 = fig.add_subplot(gs[0, 1])
plot_event_rate(summary, ax=ax2)

ax3 = fig.add_subplot(gs[1, :])
plot_sample_distribution(summary, ax=ax3)

fig.suptitle('Comprehensive Analysis', fontsize=16)
```

### High-Quality Export
```python
fig = plot_bin_statistics(binner)
fig.savefig('analysis.png', dpi=300, bbox_inches='tight')  # PNG
fig.savefig('analysis.svg', format='svg')  # Vector
fig.savefig('analysis.pdf')  # PDF