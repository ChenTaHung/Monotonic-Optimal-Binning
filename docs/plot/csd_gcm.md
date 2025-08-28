# CSD/GCM Visualization Module Documentation

## Overview
The csd_gcm module provides visualization tools for understanding the PAVA algorithm's behavior. It creates Cumulative Sum Diagrams (CSD) and Greatest Convex Minorant (GCM) plots to visualize how PAVA creates monotonic blocks from data.

## Module Location
`src/MOBPY/plot/csd_gcm.py`

## Main Functions

### `plot_gcm()`
Visualizes the Greatest Convex Minorant showing the monotonic fit produced by PAVA.

```python
def plot_gcm(
    groups_df: pd.DataFrame,
    blocks: List[Dict[str, Any]],
    *,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    show_points: bool = True,
    show_blocks: bool = True,
    point_color: str = "#1E88E5",
    block_color: str = "#E53935",
    block_alpha: float = 0.3,
    point_size: float = 60,
    line_width: float = 2.5,
    show_legend: bool = True,
    x_name: str = "x",
    y_name: str = "y"
) -> Axes
```

**Parameters:**
- **groups_df** (`pd.DataFrame`): DataFrame from PAVA.groups_ with grouped statistics
- **blocks** (`List[Dict]`): List of blocks from PAVA.export_blocks(as_dict=True)
- **ax** (`Optional[Axes]`): Matplotlib axes. If None, creates new figure
- **figsize** (`Tuple[float, float]`): Figure size if creating new figure
- **title** (`Optional[str]`): Plot title. If None, uses default
- **show_points** (`bool`): Whether to show original group points
- **show_blocks** (`bool`): Whether to show PAVA block regions
- **point_color** (`str`): Color for data points
- **block_color** (`str`): Color for block step function
- **block_alpha** (`float`): Transparency for block regions
- **point_size** (`float`): Size of scatter points
- **line_width** (`float`): Width of block lines
- **show_legend** (`bool`): Whether to show legend
- **x_name** (`str`): Name for x-axis label
- **y_name** (`str`): Name for y-axis label

**Returns:** Axes object with the plot

**Example:**
```python
from MOBPY.plot.csd_gcm import plot_gcm

# After fitting PAVA
pava = PAVA(df, x='age', y='default')
pava.fit()

# Plot GCM
ax = plot_gcm(
    groups_df=pava.groups_,
    blocks=pava.export_blocks(as_dict=True),
    title="PAVA Monotonic Fit",
    x_name="Age",
    y_name="Default Rate"
)
plt.show()
```

### `plot_csd()`
Plots the Cumulative Sum Diagram showing cumulative statistics.

```python
def plot_csd(
    groups_df: pd.DataFrame,
    *,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    color: str = "#1E88E5",
    line_width: float = 2,
    marker_size: float = 8,
    show_grid: bool = True,
    x_name: str = "x",
    y_name: str = "y"
) -> Axes
```

**Parameters:**
- **groups_df** (`pd.DataFrame`): DataFrame with cumulative statistics
- **ax** (`Optional[Axes]`): Matplotlib axes
- **figsize** (`Tuple[float, float]`): Figure size
- **title** (`Optional[str]`): Plot title
- **color** (`str`): Line and point color
- **line_width** (`float`): Line width
- **marker_size** (`float`): Marker size
- **show_grid** (`bool`): Whether to show grid
- **x_name** (`str`): X-axis label
- **y_name** (`str`): Y-axis label

**Returns:** Axes object with the plot

**Example:**
```python
from MOBPY.plot.csd_gcm import plot_csd

# Plot Cumulative Sum Diagram
ax = plot_csd(
    groups_df=pava.groups_,
    title="Cumulative Sum Diagram",
    show_grid=True
)
```

### `plot_pava_comparison()`
Creates side-by-side comparison of CSD and GCM plots.

```python
def plot_pava_comparison(
    binner: MonotonicBinner,
    *,
    figsize: Tuple[float, float] = (15, 6),
    title: Optional[str] = None,
    csd_color: str = "#1E88E5",
    gcm_point_color: str = "#43A047",
    gcm_block_color: str = "#E53935"
) -> Figure
```

**Parameters:**
- **binner** (`MonotonicBinner`): Fitted MonotonicBinner instance
- **figsize** (`Tuple[float, float]`): Figure size for combined plot
- **title** (`Optional[str]`): Overall figure title
- **csd_color** (`str`): Color for CSD plot
- **gcm_point_color** (`str`): Color for GCM data points
- **gcm_block_color** (`str`): Color for GCM blocks

**Returns:** Figure object with both plots

**Example:**
```python
from MOBPY.plot.csd_gcm import plot_pava_comparison
from MOBPY import MonotonicBinner

# Fit binner
binner = MonotonicBinner(df, x='age', y='default')
binner.fit()

# Create comparison plot
fig = plot_pava_comparison(
    binner,
    title="PAVA Algorithm Visualization"
)
plt.show()
```

### `plot_pava_process()`
Visualizes the step-by-step PAVA merging process.

```python
def plot_pava_process(
    groups_df: pd.DataFrame,
    blocks_history: List[List[Dict]],
    *,
    figsize: Tuple[float, float] = (12, 8),
    n_steps: int = 4,
    title: Optional[str] = None
) -> Figure
```

**Parameters:**
- **groups_df** (`pd.DataFrame`): Original grouped data
- **blocks_history** (`List[List[Dict]]`): History of blocks at each merge step
- **figsize** (`Tuple[float, float]`): Figure size
- **n_steps** (`int`): Number of steps to show
- **title** (`Optional[str]`): Figure title

**Returns:** Figure showing PAVA progression

**Example:**
```python
from MOBPY.plot.csd_gcm import plot_pava_process

# Track PAVA history (requires custom implementation)
history = []
# ... PAVA fitting with history tracking ...

fig = plot_pava_process(
    groups_df=pava.groups_,
    blocks_history=history,
    n_steps=4,
    title="PAVA Merging Process"
)
```

### `plot_pava_animation()`
Creates an animated visualization of PAVA execution.

```python
def plot_pava_animation(
    groups_df: pd.DataFrame,
    blocks_history: List[List[Dict]],
    *,
    filename: str = "pava_animation.gif",
    fps: int = 2,
    figsize: Tuple[float, float] = (10, 6)
) -> None
```

**Parameters:**
- **groups_df** (`pd.DataFrame`): Original grouped data
- **blocks_history** (`List[List[Dict]]`): History of blocks
- **filename** (`str`): Output filename for animation
- **fps** (`int`): Frames per second
- **figsize** (`Tuple[float, float]`): Figure size

**Note:** Requires matplotlib animation support and imagemagick/ffmpeg

**Example:**
```python
from MOBPY.plot.csd_gcm import plot_pava_animation

# Create animation (saves to file)
plot_pava_animation(
    groups_df=pava.groups_,
    blocks_history=history,
    filename="pava_process.gif",
    fps=2
)
```

## Visualization Components

### CSD (Cumulative Sum Diagram)
The CSD shows cumulative sums of the target variable against cumulative counts or x values. It reveals:
- Overall trend in the data
- Points where monotonicity is violated
- Natural groupings in the data

**Interpretation:**
- Upward slope: Positive relationship
- Downward slope: Negative relationship
- Changes in slope: Different regions in data

### GCM (Greatest Convex Minorant)
The GCM represents the monotonic step function produced by PAVA:
- Horizontal lines show constant mean within blocks
- Vertical jumps show transitions between blocks
- Width of steps indicates block size

**Interpretation:**
- Wider steps: More data pooled together
- Many narrow steps: Less pooling needed (already monotonic)
- Few wide steps: Heavy pooling (many violations)

## Styling and Customization

### Color Schemes
Default colors are chosen for clarity and accessibility:
```python
# Default color palette
COLORS = {
    'primary': '#1E88E5',   # Vivid Blue
    'secondary': '#E53935', # Vivid Red
    'success': '#43A047',   # Green
    'warning': '#FB8C00',   # Orange
    'info': '#8E24AA'       # Purple
}
```

### Custom Styling
```python
# Custom style example
ax = plot_gcm(
    groups_df=pava.groups_,
    blocks=blocks,
    point_color='navy',
    block_color='crimson',
    block_alpha=0.2,
    point_size=100,
    line_width=3
)

# Additional customization
ax.set_xlabel("Feature Value", fontsize=12)
ax.set_ylabel("Target Mean", fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_facecolor('#f0f0f0')
```

### Using with Matplotlib Styles
```python
import matplotlib.pyplot as plt

# Apply style before plotting
plt.style.use('seaborn-v0_8-darkgrid')

ax = plot_gcm(groups_df, blocks)
```

## Integration Examples

### Complete PAVA Visualization Pipeline
```python
from MOBPY.core.pava import PAVA
from MOBPY.plot.csd_gcm import plot_gcm, plot_csd, plot_pava_comparison
import matplotlib.pyplot as plt

def visualize_pava_complete(df, x, y):
    """Complete PAVA visualization pipeline."""
    
    # Fit PAVA
    pava = PAVA(df=df, x=x, y=y, sign='auto')
    pava.fit()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Original data
    axes[0].scatter(df[x], df[y], alpha=0.5, s=10)
    axes[0].set_title("Original Data")
    axes[0].set_xlabel(x)
    axes[0].set_ylabel(y)
    
    # 2. CSD
    plot_csd(
        groups_df=pava.groups_,
        ax=axes[1],
        title="Cumulative Sum Diagram"
    )
    
    # 3. GCM with blocks
    plot_gcm(
        groups_df=pava.groups_,
        blocks=pava.export_blocks(as_dict=True),
        ax=axes[2],
        title="PAVA Result (GCM)"
    )
    
    plt.tight_layout()
    return fig
```

### Comparing Different Monotonicity Directions
```python
def compare_monotonicity_directions(df, x, y):
    """Compare PAVA with different monotonicity assumptions."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, sign, title in zip(
        axes, 
        ['auto', '+', '-'],
        ['Auto-detected', 'Forced Increasing', 'Forced Decreasing']
    ):
        pava = PAVA(df=df, x=x, y=y, sign=sign)
        pava.fit()
        
        plot_gcm(
            groups_df=pava.groups_,
            blocks=pava.export_blocks(as_dict=True),
            ax=ax,
            title=f"{title} (sign={pava.resolved_sign_})",
            show_legend=False
        )
    
    plt.suptitle("PAVA with Different Monotonicity Constraints")
    plt.tight_layout()
    return fig
```

## Best Practices

1. **Use High-Contrast Colors**: Ensure visibility in different environments
2. **Include Legends**: Always label different components
3. **Show Original Data**: Display points behind the fitted blocks
4. **Add Grid**: Helps in reading values
5. **Label Axes**: Clear axis labels improve interpretability
6. **Save High Resolution**: Use dpi=300 for publication quality

## Performance Considerations

- **Large Datasets**: Automatically downsamples points for display if > 10000
- **Memory**: Matplotlib figures can consume significant memory
- **Animation**: Keep frame count reasonable (< 100 frames)

## Saving Plots

### Static Images
```python
# High-quality PNG
ax = plot_gcm(groups_df, blocks)
plt.savefig('pava_gcm.png', dpi=300, bbox_inches='tight')

# Vector format for publications
plt.savefig('pava_gcm.pdf', format='pdf', bbox_inches='tight')

# SVG for web
plt.savefig('pava_gcm.svg', format='svg', bbox_inches='tight')
```

### Interactive Plots
```python
# Save as interactive HTML (requires plotly conversion)
import plotly.tools as tls
import plotly.offline as py

mpl_fig = plt.gcf()
plotly_fig = tls.mpl_to_plotly(mpl_fig)
py.plot(plotly_fig, filename='pava_interactive.html')
```

## Troubleshooting

### Common Issues

1. **Empty Plot**: Check that groups_df is not empty
2. **Misaligned Blocks**: Ensure blocks match the groups data
3. **Memory Error**: Reduce figure size or downsample data
4. **No Animation**: Install imagemagick or ffmpeg

### Debug Mode
```python
# Enable debug information
import logging
logging.basicConfig(level=logging.DEBUG)

# Plot with debug info
ax = plot_gcm(groups_df, blocks)
print(f"Groups: {len(groups_df)}")
print(f"Blocks: {len(blocks)}")
print(f"X range: [{groups_df['x'].min()}, {groups_df['x'].max()}]")
```

## Dependencies
- matplotlib >= 3.5
- numpy
- pandas
- MOBPY.core.pava (for type hints)
- MOBPY.exceptions
- MOBPY.config

## See Also
- [`PAVA`](../core/pava.md) - Algorithm being visualized
- [`MonotonicBinner`](../binning/mob.md) - Main binning class
- [`mob_plot`](./mob_plot.md) - Binning result visualizations
- [Matplotlib Documentation](https://matplotlib.org/)