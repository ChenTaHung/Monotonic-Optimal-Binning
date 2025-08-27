
# `plot_pava_animation` Function Documentation

## Overview
The `plot_pava_animation` function creates an animated visualization of the PAVA algorithm execution, showing how violations are progressively detected and resolved through pooling operations. This provides an intuitive understanding of the algorithm's dynamics.

## Function Signature
```python
def plot_pava_animation(
    groups_df: pd.DataFrame,
    blocks: List[Dict[str, Any]],
    *,
    figsize: Tuple[float, float] = (14, 8),
    interval: int = 500,
    repeat: bool = True,
    fps: int = 2,
    save_path: Optional[str] = None,
    writer: str = 'pillow',
    title_template: str = "PAVA Animation - Step {step}/{total}",
    colors: Dict[str, str] = None,
    show_progress: bool = True
) -> animation.FuncAnimation
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **groups_df** | `pd.DataFrame` | required | DataFrame with grouped statistics |
| **blocks** | `List[Dict]` | required | Blocks showing algorithm progression |
| **figsize** | `Tuple` | `(14, 8)` | Figure size for animation |
| **interval** | `int` | `500` | Milliseconds between frames |
| **repeat** | `bool` | `True` | Whether animation loops |
| **fps** | `int` | `2` | Frames per second for saved file |
| **save_path** | `Optional[str]` | `None` | Path to save animation (gif/mp4) |
| **writer** | `str` | `'pillow'` | Animation writer ('pillow', 'ffmpeg', 'imagemagick') |
| **title_template** | `str` | template | Title format with {step} and {total} placeholders |
| **colors** | `Dict` | `None` | Custom color scheme |
| **show_progress** | `bool` | `True` | Show progress bar while generating |

## Returns
- **FuncAnimation**: Matplotlib animation object

## Usage Examples

### Basic Animation
```python
from MOBPY.plot import plot_pava_animation

# Create animation
anim = plot_pava_animation(
    groups_df=binner._pava.groups_,
    blocks=binner._pava.export_blocks(as_dict=True)
)

# Display in Jupyter notebook
from IPython.display import HTML
HTML(anim.to_jshtml())
```

### Save as GIF
```python
anim = plot_pava_animation(
    groups_df, blocks,
    save_path='pava_process.gif',
    fps=2,
    interval=600  # Slower animation
)
```

### Save as MP4
```python
anim = plot_pava_animation(
    groups_df, blocks,
    save_path='pava_process.mp4',
    writer='ffmpeg',
    fps=10,
    interval=100  # Faster for video
)
```

### Custom Color Scheme
```python
colors = {
    'initial': '#666666',
    'violation': '#FF6B6B',
    'pooling': '#4ECDC4',
    'final': '#45B7D1'
}

anim = plot_pava_animation(
    groups_df, blocks,
    colors=colors,
    title_template="Monotonic Optimization: Frame {step} of {total}"
)
```

## Animation Stages

### Frame Sequence
1. **Initial State**: Shows original data points
2. **Violation Detection**: Highlights monotonicity violations
3. **Pooling Decision**: Indicates which points will merge
4. **Merge Operation**: Shows the pooling animation
5. **Update**: Display new monotonic state
6. **Repeat**: Continue until all violations resolved

### Visual Indicators
- **Red Arrows**: Indicate violations
- **Yellow Highlights**: Points being pooled
- **Green Checkmarks**: Resolved monotonic segments
- **Progress Bar**: Shows algorithm completion

## Advanced Features

### Interactive Controls
```python
from matplotlib.widgets import Button, Slider

fig, ax = plt.subplots(figsize=(14, 10))

# Create animation
anim = plot_pava_animation(groups_df, blocks)

# Add play/pause button
ax_button = plt.axes([0.45, 0.02, 0.1, 0.04])
button = Button(ax_button, 'Pause')

def toggle(event):
    if button.label.get_text() == 'Pause':
        anim.pause()
        button.label.set_text('Play')
    else:
        anim.resume()
        button.label.set_text('Pause')

button.on_clicked(toggle)

# Add speed control
ax_slider = plt.axes([0.2, 0.02, 0.2, 0.04])
slider = Slider(ax_slider, 'Speed', 0.5, 2.0, valinit=1.0)

def update_speed(val):
    anim.interval = 500 / val

slider.on_changed(update_speed)

plt.show()
```

### Multi-Panel Animation
```python
def create_multi_panel_animation(groups_df, blocks):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        # Left: Current state
        plot_current_state(ax1, frame)
        
        # Right: Cumulative changes
        plot_cumulative(ax2, frame)
        
        fig.suptitle(f'PAVA Progress: Step {frame+1}/{len(blocks)}')
    
    return animation.FuncAnimation(
        fig, update, frames=len(blocks),
        interval=500, repeat=True
    )
```

### Educational Annotations
```python
def educational_animation(groups_df, blocks):
    fig, ax = plt.subplots(figsize=(14, 10))
    
    explanations = [
        "Starting with non-monotonic data",
        "Detecting first violation",
        "Pooling adjacent violators",
        "Checking for new violations",
        "Continuing until monotonic",
        "Final monotonic solution"
    ]
    
    def update(frame):
        ax.clear()
        
        # Plot current state
        plot_state(ax, frame)
        
        # Add educational text
        explanation_idx = min(frame, len(explanations)-1)
        ax.text(0.02, 0.98, explanations[explanation_idx],
                transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor="yellow", alpha=0.5))
    
    return animation.FuncAnimation(
        fig, update, frames=len(blocks),
        interval=800, repeat=True
    )
```

## Performance Optimization

### Large Datasets
```python
# Downsample for smooth animation
if len(groups_df) > 100:
    # Sample key frames
    key_frames = np.linspace(0, len(blocks)-1, 20, dtype=int)
    sampled_blocks = [blocks[i] for i in key_frames]
    
    anim = plot_pava_animation(
        groups_df, sampled_blocks,
        interval=300  # Faster since fewer frames
    )
```

### Memory Management
```python
# Clear cache between frames
def memory_efficient_animation(groups_df, blocks):
    import gc
    
    def update_with_cleanup(frame):
        # Update plot
        update_plot(frame)
        
        # Clear memory
        if frame % 10 == 0:
            gc.collect()
    
    return animation.FuncAnimation(
        fig, update_with_cleanup, 
        frames=len(blocks),
        cache_frame_data=False  # Don't cache
    )
```

## Export Options

### High-Quality GIF
```python
anim = plot_pava_animation(
    groups_df, blocks,
    figsize=(16, 9),  # HD aspect ratio
    save_path='pava_hd.gif',
    fps=5,
    writer='pillow'
)
```

### Web-Optimized
```python
# Smaller file size for web
anim = plot_pava_animation(
    groups_df, blocks,
    figsize=(10, 6),
    save_path='pava_web.gif',
    fps=2,
    interval=400
)

# Further optimize with external tools
# gifsicle -O3 --colors 128 pava_web.gif > pava_optimized.gif
```

### Presentation Format
```python
# For PowerPoint/Keynote
anim = plot_pava_animation(
    groups_df, blocks,
    figsize=(16, 9),
    save_path='pava_presentation.mp4',
    writer='ffmpeg',
    fps=15,
    interval=100
)
```

## Troubleshooting

### Issue: Animation Not Playing
```python
# Solution 1: Check backend
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend

# Solution 2: Use JavaScript HTML
from IPython.display import HTML
HTML(anim.to_jshtml())
```

### Issue: Slow Performance
```python
# Solution: Reduce quality
anim = plot_pava_animation(
    groups_df, blocks,
    figsize=(10, 6),  # Smaller size
    interval=200,      # Faster frames
    show_progress=False  # Skip progress bar
)
```

### Issue: Large File Size
```python
# Solution: Optimize parameters
anim = plot_pava_animation(
    groups_df, blocks,
    save_path='pava_small.gif',
    fps=1,  # Lower framerate
    figsize=(8, 5)  # Smaller dimensions
)
```

## Dependencies
- matplotlib.animation
- pillow (for GIF)`
- ffmpeg (for MP4, optional)
- imagemagick (alternative writer, optional)

## See Also
- [`plot_pava_process`](./plot_pava_process.md) - Static process view
- [`plot_gcm`](./plot_gcm.md) - Final result
- [`PAVA`](../core/pava.md) - Algorithm implementation
