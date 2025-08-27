# `plot_pava_comparison` Function Documentation

## Overview
The `plot_pava_comparison` function creates a side-by-side visualization comparing the Cumulative Sum Diagram (CSD) with the Greatest Convex Minorant (GCM), providing a comprehensive view of how PAVA transforms the data.

## Function Signature
```python
def plot_pava_comparison(
    binner: 'MonotonicBinner',
    *,
    figsize: Tuple[float, float] = (16, 6),
    title: Optional[str] = None,
    csd_color: str = "#1E88E5",
    gcm_color: str = "#E53935",
    show_grid: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 100
) -> Figure
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **binner** | `MonotonicBinner` | required | Fitted MonotonicBinner instance |
| **figsize** | `Tuple[float, float]` | `(16, 6)` | Figure size (width, height) |
| **title** | `Optional[str]` | `None` | Main title. If None, auto-generates based on data |
| **csd_color** | `str` | `"#1E88E5"` | Color for CSD plot elements |
| **gcm_color** | `str` | `"#E53935"` | Color for GCM plot elements |
| **show_grid** | `bool` | `True` | Whether to show grid on both plots |
| **save_path** | `Optional[str]` | `None` | Path to save figure if provided |
| **dpi** | `int` | `100` | DPI for saved figure |

## Returns
- **Figure**: Matplotlib Figure object containing both subplots

## Usage Examples

### Basic Usage
```python
from MOBPY import MonotonicBinner
from MOBPY.plot import plot_pava_comparison

# Fit binner
binner = MonotonicBinner(df, x='age', y='default_rate')
binner.fit()

# Create comparison plot
fig = plot_pava_comparison(binner)
plt.show()
```

### Custom Styling
```python
fig = plot_pava_comparison(
    binner,
    figsize=(18, 7),
    title="PAVA Analysis: Credit Risk by Age",
    csd_color='darkblue',
    gcm_color='darkred'
)

# Further customize
fig.suptitle(fig._suptitle.get_text(), fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Save High-Resolution
```python
fig = plot_pava_comparison(
    binner,
    save_path='pava_comparison.png',
    dpi=300
)
```

## Visual Layout

### Left Panel: CSD (Cumulative Sum Diagram)
Shows the cumulative relationship between x and y:
- **X-axis**: Cumulative count of observations
- **Y-axis**: Cumulative sum of target values
- **Blue Line**: Actual cumulative path
- **Interpretation**: Reveals the underlying trend before monotonization

### Right Panel: GCM (Greatest Convex Minorant)
Shows the monotonic fit:
- **X-axis**: Feature values
- **Y-axis**: Fitted monotonic means
- **Red Steps**: Monotonic step function after PAVA
- **Shaded Regions**: Areas where pooling occurred

## Interpretation Guide

### What to Compare
1. **Smoothness**: CSD shows original variation, GCM shows smoothed result
2. **Violations**: CSD may show non-monotonic segments, GCM is always monotonic
3. **Pooling Effect**: Shaded regions in GCM correspond to sharp changes in CSD

### Key Insights
- **Steep CSD → Large GCM steps**: Strong signal in original data
- **Flat CSD → Pooled GCM regions**: Weak or noisy signal
- **Smooth CSD → Minimal GCM pooling**: Data already nearly monotonic

## Advanced Features

### Custom Panel Arrangement
```python
# Create figure with custom layout
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

# Get data from binner
groups_df = binner._pava.groups_
blocks = binner._pava.export_blocks(as_dict=True)

# Main comparison plots
ax1 = fig.add_subplot(gs[0, 0])
# Plot CSD manually

ax2 = fig.add_subplot(gs[0, 1])  
# Plot GCM manually

# Additional analysis below
ax3 = fig.add_subplot(gs[1, :])
# Add custom analysis

plt.tight_layout()
```

### Interactive Features
```python
from matplotlib.widgets import Button

fig = plot_pava_comparison(binner)

# Add reset zoom button
ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
btn = Button(ax_button, 'Reset')

def reset(event):
    for ax in fig.get_axes():
        ax.relim()
        ax.autoscale()
    plt.draw()

btn.on_clicked(reset)
plt.show()
```

## Styling Options

### Theme Variations
```python
# Light theme
fig = plot_pava_comparison(
    binner,
    csd_color='#0066CC',
    gcm_color='#CC0000'
)
for ax in fig.get_axes():
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.2)

# Dark theme  
fig = plot_pava_comparison(
    binner,
    csd_color='#64B5F6',
    gcm_color='#EF5350'
)
for ax in fig.get_axes():
    ax.set_facecolor('#1E1E1E')
    ax.grid(True, alpha=0.1, color='white')
```

## Performance Considerations
- Automatically downsamples if >1000 unique values
- Caches intermediate calculations
- Reuses binner's computed values

## Common Use Cases

### Model Validation
```python
# Compare train and test
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, (data, label) in enumerate([(train_df, 'Train'), (test_df, 'Test')]):
    binner = MonotonicBinner(data, x='feature', y='target')
    binner.fit()
    
    # Plot comparison
    # ... custom plotting logic
```

### Feature Engineering
```python
# Compare different transformations
transformations = {
    'Original': df['feature'],
    'Log': np.log1p(df['feature']),
    'Square Root': np.sqrt(np.abs(df['feature']))
}

fig, axes = plt.subplots(len(transformations), 2, figsize=(16, 18))

for idx, (name, feature) in enumerate(transformations.items()):
    temp_df = df.copy()
    temp_df['transformed'] = feature
    
    binner = MonotonicBinner(temp_df, x='transformed', y='target')
    binner.fit()
    
    # Plot each transformation
    # ... custom plotting logic
```

## Troubleshooting

### Issue: Plots Too Small
```python
# Solution: Increase figure size
fig = plot_pava_comparison(binner, figsize=(20, 8))
```

### Issue: Labels Overlapping
```python
# Solution: Rotate labels and adjust layout
fig = plot_pava_comparison(binner)
for ax in fig.get_axes():
    ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
```

## See Also
- [`plot_gcm`](./plot_gcm.md) - Detailed GCM visualization
- [`plot_pava_process`](./plot_pava_process.md) - Step-by-step PAVA process
- [`PAVA`](../core/pava.md) - Algorithm implementation
