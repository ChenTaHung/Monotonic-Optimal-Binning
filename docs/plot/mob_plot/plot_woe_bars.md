# `plot_woe_bars` Function Documentation

## Overview

The `plot_woe_bars` function creates a bar chart of Weight of Evidence (WoE) values across bins for binary classification problems. It helps assess the predictive power of each bin and the monotonic relationship with the target variable.

## Function Signature

```python
def plot_woe_bars(
    summary_df: pd.DataFrame,
    *,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10, 6),
    title: Optional[str] = None,
    bar_color: str = "#1976D2",
    positive_color: str = "#388E3C",
    negative_color: str = "#D32F2F",
    show_values: bool = True,
    show_iv: bool = True,
    rotation: int = 45,
    bar_width: float = 0.8,
    tick_labels: Optional[Union[List[str], str]] = None,
) -> Axes
```

All parameters after `summary_df` are keyword-only.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **summary_df** | `pd.DataFrame` | required | Summary from `binner.summary_()` — must have `bucket` and `woe` columns |
| **ax** | `Optional[Axes]` | `None` | Matplotlib axes to plot on; creates new figure if `None` |
| **figsize** | `Tuple[float, float]` | `(10, 6)` | Figure size used when `ax` is `None` |
| **title** | `Optional[str]` | `None` | Plot title; auto-generated if `None` |
| **bar_color** | `str` | `"#1976D2"` | Fallback bar colour when positive/negative colours are not used |
| **positive_color** | `str` | `"#388E3C"` | Colour for positive WoE bars (green) |
| **negative_color** | `str` | `"#D32F2F"` | Colour for negative WoE bars (red) |
| **show_values** | `bool` | `True` | Display WoE values as text on each bar |
| **show_iv** | `bool` | `True` | Append total IV to the plot title |
| **rotation** | `int` | `45` | X-tick label rotation angle |
| **bar_width** | `float` | `0.8` | Bar width (0–1) |
| **tick_labels** | `Optional[Union[List[str], str]]` | `None` | X-axis tick label override (see below) |

### `tick_labels` Options

| Value | Behaviour |
|-------|-----------|
| `None` (default) | Use `bucket` column verbatim — fully backward-compatible for numeric binning |
| `list[str]` | Use the provided strings directly (one per non-NaN WoE row) |
| `'auto'` | Use `bucket` verbatim when labels are plain numeric intervals; generate compact `"Bin N\n(XX.X%)"` labels (0-based) when any label starts with `'{'` (categorical set labels) |

Detection for `'auto'` is structural — only labels starting with `{` trigger compact output. Plain numeric labels like `(-inf, 25.5)` are never modified.

## Returns

`Axes` — the matplotlib Axes containing the plot.

## Raises

`DataError` — if `summary_df` is missing the `woe` or `bucket` column.

## Usage Examples

### Numeric binning (default)

```python
from MOBPY.plot import plot_woe_bars

binner = MonotonicBinner(df, x='age', y='default')
binner.fit()
summary = binner.summary_()

ax = plot_woe_bars(summary)
plt.show()
```

### Categorical binning with compact labels

```python
from MOBPY.plot import plot_woe_bars

binner = MonotonicBinner(df, x='merchant', y='is_fraud',
                         x_type='categorical')
binner.fit()
summary = binner.summary_()

# tick_labels='auto' detects categorical labels and generates "Bin 0\n(4.5%)" style
ax = plot_woe_bars(summary, tick_labels='auto', show_iv=True)
plt.show()
```

### Explicit label override

```python
labels = [f"Group {i}" for i in range(len(summary))]
ax = plot_woe_bars(summary, tick_labels=labels)
```

### Custom styling

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
    rotation=0,
)
plt.tight_layout()
plt.show()
```

## Visual Interpretation

### WoE Values

- **Positive WoE** (green): Event rate below average — lower risk bin
- **Negative WoE** (red): Event rate above average — higher risk bin
- **Zero WoE**: Neutral — event rate equals population average
- **Magnitude**: Larger absolute values indicate stronger predictive power

### Information Value

Total IV shown in the title when `show_iv=True`:

| IV Range | Interpretation |
|----------|----------------|
| < 0.1 | Weak predictor |
| 0.1 – 0.3 | Medium predictor |
| > 0.3 | Strong predictor |

## Side-by-side with `plot_categorical_merge`

```python
from MOBPY.plot import plot_woe_bars, plot_categorical_merge

fig, axes = plt.subplots(1, 2, figsize=(18, 5))

plot_woe_bars(binner.summary_(), ax=axes[0], tick_labels='auto', show_iv=True)
plot_categorical_merge(binner, ax=axes[1], show_counts=False)

plt.tight_layout()
plt.show()
```

## See Also

- [`plot_event_rate`](./plot_event_rate.md) — event rate visualization with optional `tick_labels`
- [`plot_categorical_merge`](./plot_categorical_merge.md) — category merge visualization
- [`plot_bin_statistics`](./plot_bin_statistics.md) — comprehensive multi-panel view
- [`MonotonicBinner`](../../binning/mob.md) — main binning class
