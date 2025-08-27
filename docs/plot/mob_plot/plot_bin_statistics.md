# plot_bin_statistics Function Documentation

## Overview
The `plot_bin_statistics` function creates a comprehensive multi-panel visualization that combines multiple aspects of binning analysis into a single figure. It includes WoE bars, event rates, sample distribution, and statistical metrics, providing a complete picture of the binning results.

## Function Signature
```python
def plot_bin_statistics(
    binner: 'MonotonicBinner',
    *,
    figsize: Tuple[float, float] = (16, 12),
    title: Optional[str] = None,
    subplot_adjust: Dict[str, float] = None,
    color_scheme: str = "default",
    show_grid: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 100,
    show_table: bool = True,
    metrics_to_show: List[str] = None,
    custom_layout: Optional[Dict] = None
) -> Figure
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **binner** | `MonotonicBinner` | required | Fitted MonotonicBinner instance |
| **figsize** | `Tuple[float, float]` | `(16, 12)` | Overall figure size |
| **title** | `Optional[str]` | `None` | Main figure title |
| **subplot_adjust** | `Dict[str, float]` | `None` | Subplot spacing parameters |
| **color_scheme** | `str` | `"default"` | Color scheme ('default', 'business', 'academic', 'colorblind') |
| **show_grid** | `bool` | `True` | Show grid on all subplots |
| **save_path** | `Optional[str]` | `None` | Path to save figure |
| **dpi** | `int` | `100` | DPI for saved figure |
| **show_table** | `bool` | `True` | Include statistics table |
| **metrics_to_show** | `List[str]` | `None` | Specific metrics for table |
| **custom_layout** | `Optional[Dict]` | `None` | Custom subplot arrangement |

## Returns
- **Figure**: Matplotlib Figure object with multiple subplots

## Default Layout

The function creates a 2x2 grid (or 2x3 with table) containing:

1. **Top Left**: WoE bars (for binary) or mean values (for continuous)
2. **Top Right**: Event rate with sample distribution
3. **Bottom Left**: Sample distribution histogram
4. **Bottom Right**: Box plot of target distribution per bin
5. **Optional Bottom**: Statistics table

## Usage Examples

### Basic Usage
```python
from MOBPY.plot import plot_bin_statistics
from MOBPY import MonotonicBinner

# Fit binner
binner = MonotonicBinner(df, x='age', y='default')
binner.fit()

# Create comprehensive visualization
fig = plot_bin_statistics(binner)
plt.show()
```

### Custom Title and Save
```python
fig = plot_bin_statistics(
    binner,
    title="Credit Risk Analysis: Age-based Segmentation",
    figsize=(18, 14),
    save_path="risk_analysis.png",
    dpi=300
)
```

### Business Presentation
```python
fig = plot_bin_statistics(
    binner,
    color_scheme="business",
    show_table=True,
    metrics_to_show=['count', 'mean', 'woe', 'iv'],
    subplot_adjust={'hspace': 0.3, 'wspace': 0.25}
)

# Add company branding
fig.text(0.99, 0.01, '© Company Name', 
         ha='right', va='bottom', fontsize=8, alpha=0.5)
```

### Academic Publication
```python
fig = plot_bin_statistics(
    binner,
    color_scheme="academic",
    figsize=(10, 8),
    show_grid=True,
    show_table=False  # Separate table in paper
)

# Add figure caption
fig.text(0.5, 0.01, 
         'Figure 3: Comprehensive binning analysis showing (a) WoE distribution, '
         '(b) event rate trend, (c) sample distribution, and (d) target variability.',
         ha='center', va='bottom', fontsize=9, wrap=True)
```

## Color Schemes

### Default
```python
colors = {
    'primary': '#1E88E5',
    'secondary': '#E53935',
    'success': '#43A047',
    'warning': '#FB8C00',
    'info': '#8E24AA'
}
```

### Business
```python
colors = {
    'primary': '#003366',
    'secondary': '#CC0000',
    'success': '#006600',
    'warning': '#FF9900',
    'info': '#666666'
}
```

### Academic
```python
colors = {
    'primary': '#000000',
    'secondary': '#666666',
    'success': '#333333',
    'warning': '#999999',
    'info': '#CCCCCC'
}
```

### Colorblind-friendly
```python
colors = {
    'primary': '#0173B2',
    'secondary': '#DE8F05',
    'success': '#029E73',
    'warning': '#CC78BC',
    'info': '#ECE133'
}
```

## Customization Examples

### Custom Layout
```python
custom_layout = {
    'nrows': 3,
    'ncols': 2,
    'plots': {
        (0, 0): 'woe',
        (0, 1): 'event_rate',
        (1, 0): 'distribution',
        (1, 1): 'boxplot',
        (2, slice(None)): 'table'
    }
}

fig = plot_bin_statistics(
    binner,
    custom_layout=custom_layout,
    figsize=(14, 16)
)
```

### Selective Metrics
```python
# Show only essential metrics
fig = plot_bin_statistics(
    binner,
    metrics_to_show=['bucket', 'count', 'mean', 'woe'],
    show_table=True
)

# Highlight specific cells in table
for ax in fig.axes:
    if hasattr(ax, 'tables'):
        table = ax.tables[0]
        # Highlight high-risk bins
        for i, cell in enumerate(table.get_celld().values()):
            if i > 0 and cell.get_text().get_text() == 'High Risk':
                cell.set_facecolor('#FFE5E5')
```

### Interactive Features
```python
import matplotlib.patches as mpatches
from matplotlib.widgets import CheckButtons

fig = plot_bin_statistics(binner)

# Add toggle buttons for subplots
rax = plt.axes([0.01, 0.5, 0.08, 0.15])
labels = ['WoE', 'Event Rate', 'Distribution', 'Boxplot']
visibility = [True, True, True, True]
check = CheckButtons(rax, labels, visibility)

def toggle_visibility(label):
    index = labels.index(label)
    fig.axes[index].set_visible(not fig.axes[index].get_visible())
    plt.draw()

check.on_clicked(toggle_visibility)
plt.show()
```

## Advanced Features

### Adding Annotations
```python
fig = plot_bin_statistics(binner)

# Annotate each subplot
annotations = [
    (0, "Risk increases with age"),
    (1, "Sample concentration in middle bins"),
    (2, "Normal distribution of samples"),
    (3, "Increasing variance in older segments")
]

for idx, text in annotations:
    if idx < len(fig.axes):
        fig.axes[idx].text(0.02, 0.98, text,
                          transform=fig.axes[idx].transAxes,
                          fontsize=9, style='italic',
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', 
                                   facecolor='wheat', alpha=0.5))
```

### Comparative Analysis
```python
# Compare multiple models
models = {
    'Age': binner_age,
    'Income': binner_income,
    'Combined': binner_combined
}

fig, axes = plt.subplots(len(models), 4, figsize=(16, 4*len(models)))

for i, (name, binner) in enumerate(models.items()):
    # Create custom plots for each model
    plot_woe_bars(binner.summary_(), ax=axes[i, 0])
    plot_event_rate(binner.summary_(), ax=axes[i, 1])
    plot_sample_distribution(binner.summary_(), ax=axes[i, 2])
    
    axes[i, 0].set_ylabel(name, fontsize=12, fontweight='bold')

fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
```

### Export Components
```python
fig = plot_bin_statistics(binner)

# Export individual subplots
for i, ax in enumerate(fig.axes):
    extent = ax.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()
    )
    fig.savefig(f'subplot_{i}.png', bbox_inches=extent.expanded(1.1, 1.1))
```

## Statistics Table Details

The table includes (when applicable):
- **bucket**: Bin intervals
- **count**: Number of samples
- **count_pct**: Percentage of total
- **mean**: Target mean (event rate for binary)
- **std**: Standard deviation
- **min/max**: Target range
- **woe**: Weight of Evidence (binary only)
- **iv**: Information Value (binary only)

### Table Customization
```python
fig = plot_bin_statistics(binner)

# Find and customize table
for ax in fig.axes:
    for child in ax.get_children():
        if isinstance(child, plt.Table):
            # Style header row
            for (row, col), cell in child.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#E0E0E0')
                # Align numbers
                if col > 0 and row > 0:
                    cell.set_text_props(ha='right')
```

## Performance Optimization

### Large Datasets
```python
# Downsample for visualization
if len(df) > 10000:
    sample_df = df.sample(n=10000, random_state=42)
    binner_sample = MonotonicBinner(sample_df, x='feature', y='target')
    binner_sample.fit()
    fig = plot_bin_statistics(binner_sample)
else:
    fig = plot_bin_statistics(binner)
```

### Memory Management
```python
# Clear memory after saving
fig = plot_bin_statistics(binner, save_path='analysis.png', dpi=150)
plt.close(fig)
import gc
gc.collect()
```

## Common Issues and Solutions

### Issue: Overlapping Subplots
```python
# Solution: Adjust spacing
fig = plot_bin_statistics(
    binner,
    subplot_adjust={'hspace': 0.4, 'wspace': 0.3}
)
```

### Issue: Table Cut Off
```python
# Solution: Increase figure height
fig = plot_bin_statistics(
    binner,
    figsize=(16, 14),  # More height
    show_table=True
)
plt.subplots_adjust(bottom=0.1)  # More space at bottom
```

### Issue: Poor Resolution
```python
# Solution: Increase DPI
fig = plot_bin_statistics(
    binner,
    save_path='high_res.png',
    dpi=300  # Publication quality
)
```

## Integration with Reports

### Jupyter Notebook
```python
from IPython.display import display, HTML

fig = plot_bin_statistics(binner)
plt.show()

# Add summary text
display(HTML(f"""
<h3>Binning Analysis Summary</h3>
<ul>
    <li>Total bins: {len(binner.bins_())}</li>
    <li>Total IV: {binner.summary_()['iv'].sum():.3f}</li>
    <li>Monotonicity: {binner.resolved_sign_}</li>
</ul>
"""))
```

### PDF Report Generation
```python
from matplotlib.backends.backend_pdf import PdfPages

with PdfPages('binning_report.pdf') as pdf:
    # Main analysis
    fig = plot_bin_statistics(binner)
    pdf.savefig(fig)
    plt.close()
    
    # Additional pages...
    d = pdf.infodict()
    d['Title'] = 'Binning Analysis Report'
    d['Author'] = 'Data Science Team'
```

## See Also
- [`plot_woe_bars`](./plot_woe_bars.md) - WoE visualization details
- [`plot_event_rate`](./plot_event_rate.md) - Event rate analysis
- [`plot_sample_distribution`](./plot_sample_distribution.md) - Distribution analysis
- [`MonotonicBinner`](../binning/mob.md) - Main binning class