# `plot_categorical_merge` Function Documentation

## Overview

`plot_categorical_merge` visualizes how original categories were merged into final bins after fitting a categorical `MonotonicBinner`. It is the categorical analogue of the PAVA process plot for numeric binning.

Each bar represents one original category. Bars belonging to the same final bin are grouped together (with a visible gap between groups), coloured identically, and have a dashed horizontal line drawn at the bin's pooled event rate. An overall mean dotted line provides a reference.

## Function Signature

```python
def plot_categorical_merge(
    binner,
    *,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (12, 5),
    title: Optional[str] = None,
    show_counts: bool = True,
) -> Axes
```

All parameters after `binner` are keyword-only.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **binner** | `MonotonicBinner` | required | A fitted binner with `x_type='categorical'` |
| **ax** | `Optional[Axes]` | `None` | Axes to draw on; creates new figure if `None` |
| **figsize** | `Tuple[float, float]` | `(12, 5)` | Figure size used when `ax` is `None` |
| **title** | `Optional[str]` | `None` | Chart title; defaults to `"Category Merge: N cats → K bins  [column_name]"` |
| **show_counts** | `bool` | `True` | Annotate each bar with its sample count (`n=...`) |

## Returns

`Axes` — the matplotlib Axes containing the chart.

## Raises

- `NotFittedError`: If `binner.fit()` has not been called.
- `ValueError`: If the binner was not fitted with `x_type='categorical'`.

## Chart Elements

| Element | Description |
|---------|-------------|
| Bars | One bar per original category, coloured by final bin assignment |
| Group shading | Light `axvspan` background per bin group (6% alpha) |
| Group header | `"Bin N  (XX.X%)\nn=K"` label above each group; N is 0-based and matches `bins_()` row index |
| Dashed hline | Per-bin pooled event rate spanning its group |
| Dotted hline | Overall mean event rate across all data |
| Legend | One patch per bin (0-based index, labelled with event rate) + overall mean line |

Within each group, bars are sorted by ascending individual event rate. Groups are ordered by ascending pooled event rate.

## Bin Index Convention

The bin index in group headers and legend labels is **0-based** and matches the row index of `binner.bins_()` and `binner.summary_()` exactly:

```python
ax = plot_categorical_merge(binner)

# "Bin 2 (12.5%)" in the chart → bins_().loc[2]
binner.bins_().loc[2]
binner.summary_().loc[2]

# Also cross-reference via bin_assignment()
ba = binner.bin_assignment()
ba[ba == 2].index.tolist()   # which categories are in Bin 2
```

## Import

```python
from MOBPY.plot import plot_categorical_merge
```

## Usage Examples

### Basic usage

```python
from MOBPY import MonotonicBinner, BinningConstraints
from MOBPY.plot import plot_categorical_merge
import matplotlib.pyplot as plt

binner = MonotonicBinner(
    df, x='merchant_category', y='is_fraud',
    x_type='categorical',
    constraints=BinningConstraints(max_bins=6, min_bins=2, min_samples=30),
)
binner.fit()

ax = plot_categorical_merge(binner)
plt.tight_layout()
plt.show()
```

### Without count annotations

```python
ax = plot_categorical_merge(binner, show_counts=False)
```

### Custom figure size and title

```python
ax = plot_categorical_merge(
    binner,
    figsize=(16, 6),
    title="Merchant Category Binning — Fraud Rate",
)
```

### Side-by-side with WoE bars

```python
from MOBPY.plot import plot_woe_bars, plot_categorical_merge

fig, axes = plt.subplots(1, 2, figsize=(18, 5))

plot_woe_bars(binner.summary_(), ax=axes[0],
              tick_labels='auto', show_iv=True)
plot_categorical_merge(binner, ax=axes[1], show_counts=False)

plt.tight_layout()
plt.show()
```

### Embedding in a larger layout

```python
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 2)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

plot_woe_bars(binner.summary_(), ax=ax1, tick_labels='auto')
plot_event_rate(binner.summary_(), ax=ax2, tick_labels='auto')
plot_categorical_merge(binner, ax=ax3)

fig.suptitle('Categorical Binning Report', fontsize=16)
plt.tight_layout()
plt.show()
```

## Interpretation Guide

- **Bars within a group** show the within-bin heterogeneity. A wide spread of event rates inside a group means the bin groups categories with different individual behaviour — which may be acceptable if the overall chi-square p-value is not significant.
- **Dashed line per group** is the pooled event rate for that bin — the value reported in `summary_()['mean']`.
- **Dotted overall mean** helps assess which bins are above / below the population average.
- **Legend entries** are labelled with 0-based bin indices matching `bins_()`, making cross-referencing easy.

## Accessing Raw Statistics After Plotting

```python
# Print category → bin mapping
ba = binner.bin_assignment()
for bin_idx in sorted(ba.unique()):
    cats = sorted(ba[ba == bin_idx].index)
    rate = binner.bins_().loc[bin_idx, 'mean']
    print(f"Bin {bin_idx} ({rate:.1%}): {cats}")

# Get full summary with WoE
print(binner.summary_())
```

## See Also

- [`plot_woe_bars`](./plot_woe_bars.md) — WoE bars with `tick_labels='auto'` for categorical
- [`plot_event_rate`](./plot_event_rate.md) — event rate with `tick_labels='auto'` for categorical
- [`MonotonicBinner.bin_assignment()`](../../binning/mob.md#bin_assignment) — category → bin index mapping
- [Categorical Merge Module](../../core/categorical_merge.md) — algorithm details
