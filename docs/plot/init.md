# Plot Module Documentation

## Overview
The plot module provides comprehensive visualization tools for monotonic optimal binning. It includes functions for visualizing both the PAVA algorithm process and the final binning results, helping users understand how monotonicity is achieved and assess bin quality.

## Module Location
`src/MOBPY/plot/__init__.py`

## Module Structure
```
plot/
├── __init__.py       # Module initialization and exports
├── csd_gcm.py       # CSD/GCM plots for PAVA visualization
└── mob_plot.py      # MOB result plots for binning analysis
```

## Main Components

### PAVA Visualization (from `csd_gcm.py`)

| Function | Description |
|----------|-------------|
| **plot_gcm** | Greatest Convex Minorant showing monotonic fit |
| **plot_pava_comparison** | Side-by-side CSD and GCM plots |
| **plot_pava_process** | PAVA merging process visualization |
| **plot_pava_animation** | Animated visualization of PAVA iterations |

### MOB Result Visualization (from `mob_plot.py`)

| Function | Description |
|----------|-------------|
| **plot_woe_bars** | Weight of Evidence visualization for binary targets |
| **plot_event_rate** | Event rate and sample distribution by bin |
| **plot_bin_statistics** | Comprehensive multi-panel binning results |
| **plot_sample_distribution** | Distribution of samples across bins |
| **plot_bin_boundaries** | Bin cuts overlaid on feature distribution |
| **plot_binning_stability** | Compare binning on train vs test data |

## Quick Start

### Basic Usage
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

### Individual Plot Functions
```python
from MOBPY.plot import plot_woe_bars, plot_event_rate

# WoE bars for binary target
ax = plot_woe_bars(binner.summary_())

# Event rate with sample distribution
fig = plot_event_rate(binner.summary_())
```

## Plot Types

### 1. PAVA Process Visualization

#### CSD (Cumulative Sum Diagram)
Shows the cumulative sum of y values vs cumulative count, revealing the data's natural trend before monotonization.

#### GCM (Greatest Convex Minorant)
Displays the monotonic step function produced by PAVA, showing how adjacent violators were pooled.

### 2. Binning Results Visualization

#### WoE/IV Plots
For binary targets, visualizes Weight of Evidence and Information Value across bins.

#### Distribution Plots
Shows how samples are distributed across bins and the relationship with target variable.

#### Stability Plots
Compares binning results between training and test sets to assess robustness.

## Common Parameters

Most plotting functions accept these common parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **ax** | `Axes` | `None` | Matplotlib axes to plot on |
| **figsize** | `tuple` | `(10, 6)` | Figure size if creating new figure |
| **title** | `str` | Auto | Plot title |
| **colors** | `list/str` | Theme | Color scheme for plot elements |
| **show_legend** | `bool` | `True` | Whether to show legend |
| **save_path** | `str` | `None` | Path to save figure |

## Return Values

All plotting functions return matplotlib objects that can be further customized:

- Functions creating single plots return `Axes` objects
- Functions creating multi-panel plots return `Figure` objects
- All returned objects can be modified using standard matplotlib methods

## Customization Examples

### Custom Styling
```python
import matplotlib.pyplot as plt
from MOBPY.plot import plot_woe_bars

# Create custom axes with specific style
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_facecolor('#f0f0f0')

# Plot on custom axes
plot_woe_bars(binner.summary_(), ax=ax, 
              bar_color='#2E86AB', line_color='#A23B72')

# Further customize
ax.set_title('Custom WoE Visualization', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
```

### Multi-Panel Layout
```python
fig = plt.figure(figsize=(15, 10))

# Create custom layout
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Add different plots to each panel
ax1 = fig.add_subplot(gs[0, 0])
plot_woe_bars(summary, ax=ax1)

ax2 = fig.add_subplot(gs[0, 1])
plot_event_rate(summary, ax=ax2)

ax3 = fig.add_subplot(gs[1, :])
plot_sample_distribution(summary, ax=ax3)

fig.suptitle('Comprehensive Binning Analysis', fontsize=16)
plt.show()
```

### Saving Plots
```python
# High-resolution save for publication
fig = plot_bin_statistics(binner)
fig.savefig('binning_analysis.png', dpi=300, bbox_inches='tight')

# Vector format for editing
fig.savefig('binning_analysis.svg', format='svg')

# PDF for LaTeX inclusion
fig.savefig('binning_analysis.pdf', format='pdf')
```

## Color Schemes

The module uses carefully chosen color schemes for clarity:

### Default Colors
- **Primary**: `#1E88E5` (Vivid Blue)
- **Secondary**: `#E53935` (Vivid Red)
- **Success**: `#43A047` (Green)
- **Warning**: `#FB8C00` (Orange)
- **Info**: `#8E24AA` (Purple)

### Accessibility
All default color choices are:
- High contrast for readability
- Colorblind-friendly palettes available
- Print-friendly when needed

## Integration with Configuration

Plots respect global configuration settings:

```python
from MOBPY.config import set_config

# Set global plot style
set_config(plot_style='seaborn-v0_8-whitegrid')

# All subsequent plots use this style
fig = plot_bin_statistics(binner)
```

## Performance Considerations

- **Large Datasets**: Plots automatically downsample for performance when needed
- **Memory**: Figures are not cached; close them after use with `plt.close()`
- **Animations**: Use lower frame rates for complex visualizations

## Best Practices

1. **Always Label**: Ensure axes, title, and legend are present
2. **Choose Right Plot**: Use appropriate visualization for your analysis
3. **Consider Audience**: Academic vs business presentations need different styles
4. **Test on Data**: Verify plots work with your specific data characteristics
5. **Save Vectors**: Use SVG/PDF for publications, PNG for presentations

## Troubleshooting

### Common Issues

1. **Blank Plots**: Check data is not empty or all NaN
2. **Overlapping Labels**: Adjust figure size or use `plt.tight_layout()`
3. **Memory Issues**: Close figures after saving with `plt.close('all')`
4. **Import Errors**: Ensure matplotlib is installed

### Debug Mode
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for servers
```

## Dependencies
- matplotlib >= 3.5
- numpy
- pandas
- MOBPY.binning
- MOBPY.core

## See Also
- [`plot_csd_gcm`](./csd_gcm.md) - PAVA visualization details
- [`mob_plot`](./mob_plot.md) - Binning results visualization
- [`MonotonicBinner`](../binning/mob.md) - Main binning class
- [Matplotlib Documentation](https://matplotlib.org/)

## Future Enhancements
- Interactive plots with plotly
- 3D visualizations for multi-dimensional binning
- Real-time updating plots for streaming data
- Export to web-friendly formats