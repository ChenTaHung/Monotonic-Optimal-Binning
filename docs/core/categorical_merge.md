# Categorical Merge Module Documentation

## Overview

The categorical merge module implements chi-square-based block merging for categorical feature binning. It is the categorical counterpart to `merge_adjacent` in `core.merge`.

Key design choices:

- **Pair-result caching with local invalidation**: O(k²) total chi-square calls instead of the naive O(k³), where k is the initial number of categories.
- **Three merging phases** mirroring the numeric path: statistical merging → min_samples enforcement → class-count enforcement.
- **Multiple comparison correction** applied at every iteration to the current active pair set, controlling false merges.

## Module Location

`src/MOBPY/core/categorical_merge.py`

## Data Structures

### `CategoryBlock`

A block representing one or more merged category values with aggregated y statistics.

Inherits sufficient statistics from `_BlockStatsBase` (shared with the numeric `Block`).

```python
@dataclass
class CategoryBlock(_BlockStatsBase):
    categories: FrozenSet[Any]          # original category values in this block
    merge_history: List[...]            # (left_cats, right_cats) tuples from each merge
    pvalue_history: List[float]         # adjusted p-values recorded at each merge step
```

**Inherited properties (from `_BlockStatsBase`):**

| Property | Type | Description |
|----------|------|-------------|
| `n` | `int` | Total sample count |
| `sum` | `float` | Sum of y values |
| `sum2` | `float` | Sum of squared y values |
| `ymin` / `ymax` | `float` | Range of y values |
| `mean` | `float` | Mean of y (`sum / n`) |
| `var` | `float` | Unbiased sample variance |
| `std` | `float` | Standard deviation |
| `positives` | `float` | Count of y=1 (`sum`) |
| `negatives` | `float` | Count of y=0 (`n - sum`) |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `merge_with(other)` | `CategoryBlock` | Pool statistics and take union of categories |
| `as_dict()` | `dict` | Export all fields as a flat dictionary |

### `ChiResult`

Result of a chi-square test between two `CategoryBlock` objects.

```python
@dataclass
class ChiResult:
    chi2: float    # chi-square statistic
    pvalue: float  # raw p-value (higher = more similar = merge candidate)
    dof: int       # degrees of freedom (always 1 for 2×2 binary table)
    n_obs: int     # total observations in both blocks
```

## Main Function

### `merge_categorical()`

```python
def merge_categorical(
    blocks: List[CategoryBlock],
    constraints: BinningConstraints,
    is_binary_y: bool,
    *,
    alpha: float = 0.05,
    correction: str = "holm",
    history: Optional[List[List[Dict]]] = None,
) -> List[CategoryBlock]
```

Merges categorical blocks using chi-square tests with multiple comparison correction.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **blocks** | `List[CategoryBlock]` | required | Initial blocks — one per unique category value |
| **constraints** | `BinningConstraints` | required | Resolved binning constraints |
| **is_binary_y** | `bool` | required | Must be `True`; chi-square requires binary target |
| **alpha** | `float` | `0.05` | Significance level — pairs with adjusted p ≥ alpha are merged |
| **correction** | `str` | `"holm"` | Multiple comparison correction: `'holm'`, `'bonferroni'`, or `'fdr_bh'` |
| **history** | `Optional[List[List[Dict]]]` | `None` | Append merge snapshots here for debugging |

**Returns:** `List[CategoryBlock]` — merged blocks satisfying all constraints.

**Raises:**

- `ValueError`: If `is_binary_y=False`.
- `FittingError`: If merging produces zero blocks or exceeds `max_bins`.

## Algorithm

### Phase 1 — Statistical merging

1. Assign integer IDs to each initial block.
2. Compute chi-square p-values for all C(k, 2) pairs upfront → `pair_cache`.
3. Apply multiple comparison correction across all active pairs.
4. Find the pair with the highest adjusted p-value (most statistically similar).
5. If `adj_p ≥ alpha` AND `n_bins > min_bins`, merge the pair:
   - Remove cached results touching either merged block (local invalidation).
   - Recompute only the O(k) new pairs for the merged block.
6. Repeat until no pair qualifies or the `min_bins` floor is reached.

**Complexity**: O(k²) total chi-square calls (vs naive O(k³) for full recompute).

### Phase 2 — Enforce min_samples

Find the smallest undersized block and merge it with its most similar partner (by chi-square p-value). Stop at `min_bins` floor.

### Phase 3 — Enforce min_positives / min_negatives

For binary targets, find the first bin violating either class-count constraint and merge it with its most similar partner. Stop at `min_bins` floor.

```
┌─────────────────────────────────────────┐
│ Phase 1: Statistical merging            │
│ (chi-square + multiple comparison corr) │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│ Phase 2: Enforce min_samples            │
│ (stop at min_bins)                      │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│ Phase 3: Enforce min_positives AND      │
│          min_negatives (binary only)    │
│ (stop at min_bins)                      │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│ Validation & Warnings                   │
└─────────────────────────────────────────┘
```

## Multiple Comparison Correction

### Available methods

| Method | Description |
|--------|-------------|
| `'holm'` (default) | Holm-Bonferroni step-down — controls FWER, strictly more powerful than Bonferroni |
| `'bonferroni'` | Multiply each p-value by the number of active pairs |
| `'fdr_bh'` | Benjamini-Hochberg — controls FDR, more permissive than Holm |

Correction is reapplied at every iteration to the **current active pair set**, so the number of comparisons shrinks as bins are merged. This means the correction becomes progressively less conservative as the algorithm converges.

### Chi-square test details

For each pair of blocks (a, b), a 2×2 contingency table is built:

```
          y=0          y=1
    a:  a.negatives  a.positives
    b:  b.negatives  b.positives
```

Yates' correction is **not** applied (correction is handled by Holm/Bonferroni). Degenerate tables (all-zero rows or columns) return `pvalue=1.0`.

## Usage Example

The `merge_categorical` function is called automatically by `MonotonicBinner.fit()` when `x_type='categorical'`. Direct use is an advanced pattern:

```python
from MOBPY.core.categorical_merge import CategoryBlock, merge_categorical
from MOBPY.core.constraints import BinningConstraints

# Build initial blocks (one per unique category)
blocks = [
    CategoryBlock(n=100, sum=10.0, sum2=10.0, ymin=0.0, ymax=1.0,
                  categories=frozenset(['electronics'])),
    CategoryBlock(n=120, sum=12.0, sum2=12.0, ymin=0.0, ymax=1.0,
                  categories=frozenset(['food'])),
    CategoryBlock(n=80,  sum=30.0, sum2=30.0, ymin=0.0, ymax=1.0,
                  categories=frozenset(['travel'])),
]

constraints = BinningConstraints(max_bins=4, min_bins=2, min_samples=50)
constraints.resolve(total_n=300, total_pos=52)

merged = merge_categorical(
    blocks=blocks,
    constraints=constraints,
    is_binary_y=True,
    alpha=0.05,
    correction='holm',
)

for b in merged:
    print(f"{sorted(b.categories)}: n={b.n}, event_rate={b.mean:.1%}")
```

## Validation and Warnings

After merging, the `_validate_cat_merge_result` helper checks:

| Condition | Action |
|-----------|--------|
| `n_bins > max_bins` | Raises `FittingError` |
| Any bin below `min_samples` AND above `min_bins` floor | Logger warning |
| Any bin below `min_samples` AND at `min_bins` floor | `UserWarning` (relaxation) |
| Any bin below `min_positives` (binary) | Warning (severity based on `min_bins` floor) |
| Any bin below `min_negatives` (binary) | Warning (severity based on `min_bins` floor) |

## Module Exports

```python
from MOBPY.core.categorical_merge import (
    CategoryBlock,
    ChiResult,
    merge_categorical,
)
```

`CategoryBlock` and `merge_categorical` are also re-exported from `MOBPY.core`:

```python
from MOBPY.core import CategoryBlock, merge_categorical
```

## See Also

- [MonotonicBinner](../binning/mob.md) — calls `merge_categorical` internally
- [BinningConstraints](./constraints.md) — supplies resolved constraints
- [Merge Module (numeric)](./merge.md) — numeric counterpart using adjacent merging
