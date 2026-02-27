# `plot_event_rate` Function Documentation

## Overview

The `plot_event_rate` function creates a dual-axis visualization showing the event rate (mean of y) as a line overlaid on sample counts as bars. It helps assess both the risk pattern and the statistical reliability of each bin.

## Function Signature

```python
def plot_event_rate(
    summary_df: pd.DataFrame,
    *,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    bar_color: str = "#64B5F6",
    line_color: str = "#E53935",
    show_counts: bool = True,
    show_rate_values: bool = True,
    rotation: int = 45,
    y_format: str = "percentage",
    tick_labels: Optional[Union[List[str], str]] = None,
) -> Axes
```

All parameters after `summary_df` are keyword-only.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **summary_df** | `pd.DataFrame` | required | Summary from `binner.summary_()` — must have `bucket`, `mean`, `count` columns |
| **ax** | `Optional[Axes]` | `None` | Matplotlib axes to plot on; creates new figure if `None` |
| **figsize** | `Tuple[float, float]` | `(10, 6)` | Figure size used when `ax` is `None` |
| **title** | `Optional[str]` | `None` | Plot title; auto-generated if `None` |
| **bar_color** | `str` | `"#64B5F6"` | Colour for sample count bars (light blue) |
| **line_color** | `str` | `"#E53935"` | Colour for the event rate line (red) |
| **show_counts** | `bool` | `True` | Show sample counts as bars on the primary y-axis |
| **show_rate_values** | `bool` | `True` | Annotate the event rate line with numeric values |
| **rotation** | `int` | `45` | X-tick label rotation angle |
| **y_format** | `str` | `"percentage"` | Rate scale: `"percentage"` (0–100 %) or `"decimal"` (0–1) |
| **tick_labels** | `Optional[Union[List[str], str]]` | `None` | X-axis tick label override (see below) |

### `tick_labels` Options

| Value | Behaviour |
|-------|-----------|
| `None` (default) | Use `bucket` column verbatim — fully backward-compatible |
| `list[str]` | Use the provided strings directly (one per row) |
| `'auto'` | Use `bucket` verbatim for numeric bins; generate compact `"Bin N\n(XX.X%)"` labels (0-based) when any label starts with `'{'` (categorical set labels) |

## Returns

`Axes` — the primary (left) axes. The secondary right axes (event rate scale) is accessible via `ax.right_ax` if needed for further customization.

## Raises

`DataError` — if `summary_df` is missing any of the required columns (`bucket`, `mean`, `count`).

## Visual Components

### Primary axis (left)

When `show_counts=True`, sample counts per bin are drawn as bars. The y-axis label reads `"Sample Count"`.

### Secondary axis (right)

The event rate (mean of y) is drawn as a line with circular markers. The y-axis is formatted as a percentage (`y_format="percentage"`) or decimal (`y_format="decimal"`).

## Usage Examples

### Basic numeric binning

```python
from MOBPY.plot import plot_event_rate

binner = MonotonicBinner(df, x='age', y='default')
binner.fit()
summary = binner.summary_()

ax = plot_event_rate(summary, show_counts=True)
plt.show()
```

### Categorical binning with compact labels

```python
binner = MonotonicBinner(df, x='merchant', y='is_fraud',
                         x_type='categorical')
binner.fit()
summary = binner.summary_()

ax = plot_event_rate(summary, tick_labels='auto', show_counts=True)
plt.show()
```

### Rate-only view (no count bars)

```python
ax = plot_event_rate(summary, show_counts=False, y_format='decimal')
```

### Custom styling

```python
fig, ax = plt.subplots(figsize=(14, 7))

plot_event_rate(
    summary,
    ax=ax,
    title="Default Rate by Income Bin",
    bar_color='lightgray',
    line_color='darkred',
    show_counts=True,
    show_rate_values=True,
    y_format='percentage',
)
plt.tight_layout()
plt.show()
```

### Side-by-side WoE and event rate

```python
from MOBPY.plot import plot_woe_bars, plot_event_rate

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

plot_woe_bars(summary, ax=ax1, tick_labels='auto')
plot_event_rate(summary, ax=ax2, tick_labels='auto', show_counts=True)

plt.tight_layout()
plt.show()
```

### Accessing the secondary axis

```python
ax = plot_event_rate(summary)
ax2 = ax.right_ax   # secondary axis for event rate

# Add an overall mean reference line
overall_rate = summary['mean'].mean() * 100
ax2.axhline(y=overall_rate, color='blue', linestyle='--', alpha=0.7,
            label=f'Overall rate: {overall_rate:.1f}%')
ax2.legend(loc='upper right')
```

## Interpretation Guide

### Ideal pattern

- Event rate increases (or decreases) monotonically across bins.
- Bins have comparable sample counts — no extremely small bins.

### Warning signs

- **Non-monotonic line**: suggests the binning did not achieve the desired monotonicity.
- **Very sparse bins**: sample counts near zero reduce the reliability of the event rate estimate.
- **Extreme rates**: bins at 0% or 100% event rate indicate class-count constraint issues.

## See Also

- [`plot_woe_bars`](./plot_woe_bars.md) — WoE visualization with identical `tick_labels` support
- [`plot_categorical_merge`](./plot_categorical_merge.md) — category merge visualization
- [`plot_bin_statistics`](./plot_bin_statistics.md) — comprehensive multi-panel view
- [`MonotonicBinner`](../../binning/mob.md) — main binning class
