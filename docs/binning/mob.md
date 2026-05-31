# MonotonicBinner Class Documentation

## Overview

The MonotonicBinner class is the main orchestrator for monotonic optimal binning. It supports two distinct fitting pipelines:

- **Numeric path** (`x_type='numeric'`, default): PAVA + adjacent merging via Welch's t-test.
- **Categorical path** (`x_type='categorical'`): chi-square merging with multiple comparison correction.

Both paths share the same constraint system, WoE/IV calculation, and summary output format.

## Module Location

`src/MOBPY/binning/mob.py`

## Class Definition

```python
class MonotonicBinner:
    def __init__(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        *,
        metric: Literal["mean"] = "mean",
        sign: Literal["+", "-", "auto"] = "auto",
        strict: bool = True,
        constraints: Optional[BinningConstraints] = None,
        exclude_values: Optional[Iterable] = None,
        sort_kind: Optional[str] = "quicksort",
        merge_strategy: Union[MergeStrategy, str] = MergeStrategy.HIGHEST_PVALUE,
        x_type: Literal["auto", "numeric", "categorical"] = "numeric",
        categorical_alpha: float = 0.05,
        categorical_correction: Literal["bonferroni", "holm", "fdr_bh"] = "holm",
        unseen_categories: Literal["unknown", "error"] = "error",
        max_label_cats: Optional[int] = None,
    )
```

All parameters after `y` are keyword-only.

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **df** | `pd.DataFrame` | required | Input DataFrame |
| **x** | `str` | required | Feature column name |
| **y** | `str` | required | Target column name |
| **metric** | `Literal["mean"]` | `"mean"` | Aggregation metric (only `"mean"` supported) |
| **sign** | `Literal["+", "-", "auto"]` | `"auto"` | Monotonicity direction (numeric path only) |
| **strict** | `bool` | `True` | Enforce strict monotonicity (numeric path only) |
| **constraints** | `BinningConstraints` | `None` | Binning constraints; defaults to `BinningConstraints()` |
| **exclude_values** | `Iterable` | `None` | Feature values excluded from binning (reported separately) |
| **sort_kind** | `str` | `"quicksort"` | Pandas sort algorithm (numeric path only) |
| **merge_strategy** | `MergeStrategy` | `HIGHEST_PVALUE` | Block selection strategy (numeric path only) |
| **x_type** | `Literal["auto","numeric","categorical"]` | `"numeric"` | Routing: `'numeric'` always uses PAVA; `'categorical'` uses chi-square; `'auto'` detects from dtype |
| **categorical_alpha** | `float` | `0.05` | Significance level for categorical merging — pairs with adjusted p ≥ alpha are merged |
| **categorical_correction** | `Literal["bonferroni","holm","fdr_bh"]` | `"holm"` | Multiple comparison correction for categorical merging |
| **unseen_categories** | `Literal["unknown","error"]` | `"error"` | Behaviour for categories not seen during fit: `'error'` (default) raises `ValueError`; `'unknown'` returns `"Unknown"` / NaN WoE |
| **max_label_cats** | `Optional[int]` | `None` | Max category names in a bin label — excess truncated as `{A, B, C, ...+N}` |

## Key Methods

### fit()

Runs the complete binning pipeline. Automatically routes to numeric or categorical path based on `x_type`.

**Numeric pipeline steps:**

1. Partition data into clean / missing / excluded subsets.
2. Validate and resolve constraints.
3. Run PAVA to create initial monotonic blocks.
4. Merge adjacent blocks to satisfy constraints.
5. Build bins DataFrame and summary with WoE/IV.

**Categorical pipeline steps:**

1. Partition data into clean / missing / excluded subsets.
2. Validate that y is binary (required for chi-square).
3. Resolve constraints.
4. Build one `CategoryBlock` per unique category value.
5. Merge blocks using chi-square + multiple comparison correction.
6. Build categorical bins DataFrame and summary with WoE/IV.

**Returns:** Self (for method chaining)

**Raises:**

- `DataError`: No clean rows, wrong dtype, or non-binary y on the categorical path.
- `FittingError`: PAVA convergence failure or zero bins after merging.

### bins_()

Returns the fitted bins as a DataFrame. Missing and Excluded rows are not included; use `summary_()` for those.

**Numeric x** — one row per bin, 0-based index:

| Column | Description |
|--------|-------------|
| `left` | Left bin edge (`-inf` for the first bin) |
| `right` | Right bin edge (`+inf` for the last bin) |
| `n` | Number of samples |
| `sum` | Sum of y values |
| `mean` | Mean of y |
| `std` | Standard deviation |
| `min` / `max` | Range of y values |

**Categorical x** — one row per merged category group, 0-based index:

| Column | Description |
|--------|-------------|
| `categories` | Sorted list of category values in this bin |
| `n` | Number of samples |
| `sum` | Sum of y values |
| `mean` | Mean of y (event rate) |
| `std` | Standard deviation |
| `min` / `max` | Range of y values |

**Raises:** `NotFittedError` if called before `fit()`.

### summary_()

Returns the full binning summary including WoE/IV for binary targets.

Includes separate rows for Missing and Excluded values when present. WoE and IV are calculated for **all** bins including Missing and Excluded.

**Returns:** DataFrame with columns:

| Column | Description |
|--------|-------------|
| `bucket` | Bin label (interval string or category group label) |
| `count` | Number of samples |
| `count_pct` | Percentage of total |
| `sum` | Sum of y values |
| `mean` | Mean of y (event rate for binary) |
| `std` | Standard deviation |
| `min` / `max` | Range of y values |
| `woe` | Weight of Evidence (binary targets only) |
| `iv` | Information Value contribution (binary targets only) |

**Raises:** `NotFittedError` if called before `fit()`.

### transform(x_values, assign="interval")

Transforms raw x values to bin assignments.

**Parameters:**

| Parameter  | Type        | Default      | Description         |
|------------|-------------|--------------|---------------------|
| `x_values` | `pd.Series` | required     | Values to transform |
| `assign`   | `str`       | `"interval"` | Assignment type     |

**Assignment types:**

- `"interval"`: Bin label string (interval for numeric; category group label for categorical).
- `"left"` / `"right"`: Numeric bin edges (numeric path only; raises `ValueError` for categorical).
- `"woe"`: Weight of Evidence value (binary targets only).

Missing values map to `"Missing"` (interval) or their WoE. Excluded values map to `"Excluded:<value>"` (interval) or their WoE. Unseen categories (categorical path) are handled by the `unseen_categories` constructor parameter.

**Raises:**

- `NotFittedError`: If called before `fit()`.
- `ValueError`: If `assign='woe'` on a non-binary target, or if `assign='left'/'right'` on a categorical binner.

### get_diagnostics()

Returns diagnostic information from the fitting process.

**Numeric path returns:**

| Key | Description |
|-----|-------------|
| `x_type` | `"numeric"` |
| `partition_summary` | Counts for clean / missing / excluded |
| `is_binary` | Whether target was binary |
| `resolved_sign` | Final monotonicity direction (`"+"` or `"-"`) |
| `pava_diagnostics` | PAVA algorithm metrics |
| `n_pava_blocks` | Number of blocks after PAVA |
| `n_final_bins` | Number of bins after merging |
| `constraints_satisfied` | Dict of constraint satisfaction booleans |

**Categorical path returns:**

| Key | Description |
|-----|-------------|
| `x_type` | `"categorical"` |
| `partition_summary` | Counts for clean / missing / excluded |
| `is_binary` | `True` (always — categorical path requires binary y) |
| `n_initial_categories` | Number of unique categories before merging |
| `n_final_bins` | Number of bins after merging |
| `constraints_satisfied` | Dict of constraint satisfaction booleans |

**`constraints_satisfied` keys (both paths):**

```python
cs = binner.get_diagnostics()['constraints_satisfied']
cs['max_bins']      # bool
cs['min_bins']      # bool
cs['min_samples']   # bool
cs['min_positives'] # bool (binary only)
cs['min_negatives'] # bool (binary only)
```

**Raises:** `NotFittedError` if called before `fit()`.

### bin_assignment()

Returns a Series mapping every original category to its 0-based bin index.

**Categorical path only.** The bin index directly corresponds to the row index of `bins_()` and `summary_()`.

**Returns:** `pd.Series` with `name='bin'`, dtype `int`, and index named `'category'`.

**Example:**

```python
ba = binner.bin_assignment()

# Which categories are in bin 2?
ba[ba == 2].index.tolist()

# Cross-reference with bins_() and summary_()
binner.bins_().loc[2]      # bin 2 statistics
binner.summary_().loc[2]   # bin 2 WoE/IV
```

**Raises:**

- `NotFittedError`: If called before `fit()`.
- `ValueError`: If `x_type != 'categorical'`.

### pava_blocks_(as_dict=True)

Returns the raw PAVA blocks before merging. **Numeric path only.**

Useful for inspecting the initial monotonic structure before constraint-based merging.

**Parameters:**

- `as_dict` (`bool`, default `True`): Return list of dicts (`True`) or `Block` objects (`False`).

**Returns:** List of blocks from PAVA (before `merge_adjacent`).

**Raises:** `NotFittedError` if called before `fit()` or if `x_type='categorical'`.

### pava_groups_()

Returns the grouped statistics used by PAVA before pooling. **Numeric path only.**

**Returns:** DataFrame with columns `x, count, sum, sum2, ymin, ymax, cum_count, cum_sum, cum_mean, group_mean`.

**Raises:** `NotFittedError` if called before `fit()` or if `x_type='categorical'`.

## Usage Examples

### Numeric Binning (default)

```python
import pandas as pd
from MOBPY import MonotonicBinner, BinningConstraints

df = pd.read_csv('credit_data.csv')

constraints = BinningConstraints(
    max_bins=6,
    min_bins=2,
    min_samples=0.05,
    min_positives=0.02,
    min_negatives=0.02
)

binner = MonotonicBinner(
    df=df,
    x='credit_amount',
    y='default',
    constraints=constraints
)
binner.fit()

summary = binner.summary_()
print(f"Total IV: {summary['iv'].sum():.4f}")
```

### Categorical Binning

```python
import pandas as pd
from MOBPY import MonotonicBinner, BinningConstraints

df = pd.read_csv('transactions.csv')

binner = MonotonicBinner(
    df=df,
    x='merchant_category',
    y='is_fraud',
    x_type='categorical',          # REQUIRED for categorical path
    categorical_alpha=0.05,
    categorical_correction='holm',
    constraints=BinningConstraints(max_bins=10, min_bins=2, min_samples=30),
    max_label_cats=3,              # truncate long bin labels
)
binner.fit()

diag = binner.get_diagnostics()
print(f"{diag['n_initial_categories']} categories → {diag['n_final_bins']} bins")

# Category → bin mapping
ba = binner.bin_assignment()
for bin_idx in sorted(ba.unique()):
    print(f"Bin {bin_idx} ({binner.bins_().loc[bin_idx, 'mean']:.1%}):",
          sorted(ba[ba == bin_idx].index))

# Transform
df['bin_label'] = binner.transform(df['merchant_category'], assign='interval')
df['woe_score'] = binner.transform(df['merchant_category'], assign='woe')
```

### Auto-detection

```python
# x_type='auto' routes based on column dtype
binner = MonotonicBinner(df, x='feature', y='target', x_type='auto')
binner.fit()
print(binner.get_diagnostics()['x_type'])  # 'numeric' or 'categorical'
```

## Constraint Satisfaction Diagnostics

```python
binner.fit()
diag = binner.get_diagnostics()
print("Constraints satisfied:")
for k, v in diag['constraints_satisfied'].items():
    status = "✓" if v else "✗"
    print(f"  {status} {k}")
```

## Edge Convention (Numeric)

- Bins use half-open intervals `[left, right)`.
- First bin: `(-∞, right)`.
- Last bin: `[left, +∞)`.
- Ensures complete coverage for any future x value.

## Performance Characteristics

| Path | Sorting | Main step | Merging |
|-------------|------------|-----------|----------------------|
| Numeric | O(n log n) | O(n) PAVA | O(k²) adjacent |
| Categorical | — | — | O(k²) chi-square |

Both paths handle 10² to 10⁶ samples efficiently.

## Integration Points

- **BinningConstraints**: [Constraint configuration](../core/constraints.md)
- **PAVA**: [Core monotonization algorithm (numeric)](../core/pava.md)
- **merge_adjacent**: [Numeric block merging](../core/merge.md)
- **merge_categorical**: [Categorical block merging](../core/categorical_merge.md)
- **Plotting**: [Visualization utilities](../plot/init.md)

## See Also

- [BinningConstraints](../core/constraints.md)
- [Categorical Merge Module](../core/categorical_merge.md)
- [Plot functions](../plot/init.md)
- [plot_categorical_merge](../plot/mob_plot/plot_categorical_merge.md)
