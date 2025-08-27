
# `plot_pava_process` Function Documentation

## Overview
The `plot_pava_process` function visualizes the step-by-step process of the PAVA algorithm, showing how adjacent violators are progressively pooled to achieve monotonicity. This educational visualization helps understand the algorithm's mechanics.

## Function Signature
```python
def plot_pava_process(
    groups_df: pd.DataFrame,
    blocks: List[Dict[str, Any]],
    *,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (14, 8),
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    fallback_color: str = "#FFA726",
    anchor_color: str = "#E53935",
    initial_color: str = "#BDBDBD",
    line_alpha: float = 0.3,
    point_size: float = 80,
    anchor_size: float = 120,
    show_legend: bool = True,
    x_column: str = 'x',
    y_column: str = 'mean',
    show_annotations: bool = True,
    animation_delay: Optional[float] = None
) -> Axes
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **groups_df** | `pd.DataFrame` | required | DataFrame with grouped statistics from PAVA |
| **blocks** | `List[Dict]` | required | Final blocks after PAVA completion |
| **ax** | `Optional[Axes]` | `None` | Axes to plot on, creates new if None |
| **figsize** | `Tuple` | `(14, 8)` | Figure size for new plots |
| **title** | `Optional[str]` | `None` | Main title |
| **subtitle** | `Optional[str]` | `None` | Subtitle with additional context |
| **fallback_color** | `str` | `"#FFA726"` | Color for intermediate merge points (orange) |
| **anchor_color** | `str` | `"#E53935"` | Color for final anchor points (red) |
| **initial_color** | `str` | `"#BDBDBD"` | Color for initial points (gray) |
| **line_alpha** | `float` | `0.3` | Transparency for connecting lines |
| **point_size** | `float` | `80` | Size of intermediate points |
| **anchor_size** | `float` | `120` | Size of anchor points |
| **show_legend** | `bool` | `True` | Whether to show legend |
| **x_column** | `str` | `'x'` | Name of x column |
| **y_column** | `str` | `'mean'` | Name of y column |
| **show_annotations** | `bool` | `True` | Whether to annotate merge operations |
| **animation_delay** | `Optional[float]` | `None` | Delay between steps if animating |

## Returns
- **Axes**: Matplotlib Axes object with the process visualization

## Usage Examples

### Basic Usage
```python
from MOBPY.plot import plot_pava_process

# After fitting binner
groups_df = binner._pava.groups_
blocks = binner._pava.export_blocks(as_dict=True)

ax = plot_pava_process(groups_df, blocks)
plt.show()
```

### Detailed Process View
```python
fig, ax = plt.subplots(figsize=(16, 10))

plot_pava_process(
    groups_df, blocks,
    ax=ax,
    title="PAVA Algorithm: Step-by-Step Pooling",
    subtitle=f"Merging {len(groups_df)} groups into {len(blocks)} blocks",
    show_annotations=True,
    anchor_size=150
)

# Add step numbers
for i, block in enumerate(blocks):
    ax.text(block['right'], block['mean'], f"Block {i+1}", 
            fontsize=10, ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

### Educational Visualization
```python
# Show the process with clear stages
ax = plot_pava_process(
    groups_df, blocks,
    figsize=(16, 10),
    title="Understanding PAVA: From Violation to Monotonicity",
    initial_color='lightgray',
    fallback_color='gold',
    anchor_color='darkred',
    show_annotations=True
)

# Add educational annotations
ax.text(0.02, 0.98, "Gray: Original values", 
        transform=ax.transAxes, fontsize=10, va='top')
ax.text(0.02, 0.94, "Gold: Intermediate pooling", 
        transform=ax.transAxes, fontsize=10, va='top')
ax.text(0.02, 0.90, "Red: Final monotonic values", 
        transform=ax.transAxes, fontsize=10, va='top')
```

## Visual Elements

### Point Types
1. **Gray Points**: Original group means before any pooling
2. **Gold Points**: Intermediate stages showing ongoing pooling
3. **Red Anchors**: Final monotonic values after all pooling

### Connecting Lines
- **Faint Lines**: Show the pooling path
- **Solid Lines**: Connect final monotonic blocks

### Annotations
- **Merge Indicators**: Arrows showing which points were pooled
- **Block Labels**: Final block identifiers
- **Statistics**: Sample sizes or variance reduction

## Process Stages

### Stage 1: Initial State
All points shown in gray, potentially violating monotonicity

### Stage 2: Violation Detection
Highlights pairs that violate monotonicity constraint

### Stage 3: Pooling Operations
Shows progressive merging of violators

### Stage 4: Final Result
Red anchors showing the monotonic solution

## Advanced Features

### Step-by-Step Animation
```python
import matplotlib.animation as animation

def animate_pava_process(groups_df, blocks):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Animation function
    def update(frame):
        ax.clear()
        # Show process up to frame
        plot_pava_process(
            groups_df, blocks[:frame+1],
            ax=ax,
            title=f"PAVA Step {frame+1}/{len(blocks)}"
        )
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(blocks),
        interval=500, repeat=True
    )
    
    return anim

# Create and save animation
anim = animate_pava_process(groups_df, blocks)
anim.save('pava_process.gif', writer='pillow', fps=2)
```

### Highlighting Specific Merges
```python
ax = plot_pava_process(groups_df, blocks)

# Highlight large merges
for i, block in enumerate(blocks):
    if block.get('merge_count', 0) > 3:
        circle = plt.Circle(
            (block['right'], block['mean']), 
            0.5, color='red', fill=False, 
            linewidth=2, linestyle='--'
        )
        ax.add_patch(circle)
        ax.annotate(f"Major merge\n({block['merge_count']} groups)",
                   xy=(block['right'], block['mean']),
                   xytext=(10, 10), textcoords='offset points')
```

## Interpretation Guide

### Reading the Visualization
1. **Starting Point**: Gray points show original non-monotonic data
2. **Pooling Process**: Gold points appear where violations are resolved
3. **End Result**: Red anchors form the final monotonic function

### Key Patterns
- **Many Gold Points**: Extensive pooling was needed
- **Few Red Anchors**: Data compressed into few monotonic blocks
- **Steep Transitions**: Strong signal changes in the data

## Performance Notes
- Efficiently handles up to 100 blocks
- Automatically simplifies display for >50 blocks
- Uses caching for repeated renders

## Customization

### Color Schemes
```python
# High contrast scheme
plot_pava_process(
    groups_df, blocks,
    initial_color='#000000',
    fallback_color='#FFD700',
    anchor_color='#FF0000'
)

# Colorblind-friendly
plot_pava_process(
    groups_df, blocks,
    initial_color='#999999',
    fallback_color='#0173B2',
    anchor_color='#DE8F05'
)
```

### Focus on Region
```python
ax = plot_pava_process(groups_df, blocks)

# Zoom to interesting region
ax.set_xlim(20, 40)
ax.set_ylim(0.1, 0.3)

# Add inset for full view
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axins = inset_axes(ax, width="30%", height="30%", loc='upper right')
plot_pava_process(groups_df, blocks, ax=axins, 
                  show_legend=False, show_annotations=False)
```

## Common Issues

### Issue: Too Many Points
```python
# Solution: Show only key stages
key_blocks = [b for b in blocks if b.get('merge_count', 0) > 2]
plot_pava_process(groups_df, key_blocks)
```

### Issue: Overlapping Annotations
```python
# Solution: Selective annotation
plot_pava_process(
    groups_df, blocks,
    show_annotations=False  # Disable auto annotations
)
# Add manual annotations for clarity
```

## See Also
- [`plot_pava_animation`](./plot_pava_animation.md) - Animated version
- [`plot_gcm`](./plot_gcm.md) - Final result visualization
- [`PAVA`](../core/pava.md) - Algorithm details
```
