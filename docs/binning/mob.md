# MonotonicBinner Class Documentation

## Overview
The `MonotonicBinner` class is the main orchestrator for monotonic optimal binning. It implements a complete pipeline from data partitioning through PAVA (Pool-Adjacent-Violators Algorithm) fitting and constraint-based merging to final bin creation with WoE/IV calculations for binary targets.

## Module Location
`src/MOBPY/binning/mob.py`

## Class Definition

```python
class MonotonicBinner(
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
    merge_strategy: Union[MergeStrategy, str] = MergeStrategy.HIGHEST_PVALUE
)
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **df** | `pd.DataFrame` | required | Input dataframe containing feature and target |
| **x** | `str` | required | Name of the feature column to bin |
| **y** | `str` | required | Name of the target column |
| **metric** | `Literal["mean"]` | `"mean"` | Metric to monotonize (currently only "mean" supported) |
| **sign** | `Literal["+", "-", "auto"]` | `"auto"` | Monotonicity direction: "+" (increasing), "-" (decreasing), or "auto" (infer from data) |
| **strict** | `bool` | `True` | If True, enforce strict monotonicity (no plateaus) |
| **constraints** | `BinningConstraints` | `None` | Binning constraints. If None, uses defaults |
| **exclude_values** | `Iterable` | `None` | Feature values to exclude from binning (e.g., special codes like -999) |
| **sort_kind** | `str` | `"quicksort"` | Pandas sorting algorithm for PAVA |
| **merge_strategy** | `MergeStrategy` | `HIGHEST_PVALUE` | Strategy for selecting adjacent blocks to merge |

## Attributes

### Public Attributes
| Attribute | Type | Description |
|-----------|------|-------------|
| **resolved_sign_** | `Literal["+", "-"]` | Actual monotonicity direction used after inference |

### Private Attributes (Internal Use)
| Attribute | Type | Description |
|-----------|------|-------------|
| **_is_fitted** | `bool` | Flag indicating if model has been fitted |
| **_is_binary_y** | `bool` | Whether target is binary (0/1) |
| **_parts** | `Parts` | Data partitions (clean/missing/excluded) |
| **_pava** | `PAVA` | Fitted PAVA instance |
| **_merged_blocks** | `List[Block]` | Final merged blocks |
| **_bins_df** | `pd.DataFrame` | Clean numeric bins |
| **_full_summary_df** | `pd.DataFrame` | Complete summary with special values |
| **_fit_diagnostics** | `Dict` | Diagnostic information from fitting |

## Main Methods

### `fit() -> MonotonicBinner`
Fits the monotonic binner to the data.

**Pipeline Steps:**
1. Partition data by x values (clean/missing/excluded)
2. Check if y is binary on clean partition
3. Resolve constraints based on actual data size
4. Run PAVA to create initial monotonic blocks
5. Merge adjacent blocks to satisfy constraints
6. Build final bins and summary DataFrame

**Returns:** Self for method chaining

**Raises:**
- `DataError`: If data has issues (e.g., no clean rows)
- `FittingError`: If fitting fails (e.g., PAVA convergence)

**Example:**
```python
binner = MonotonicBinner(df, x='age', y='default')
binner.fit()
```

### `bins_() -> pd.DataFrame`
Returns the fitted bins as a DataFrame.

**Returns:** DataFrame with columns:
- `left`: Left bin edge (first is -inf)
- `right`: Right bin edge (last is +inf)
- `n`: Number of samples
- `sum`: Sum of y values
- `mean`: Mean of y values
- `std`: Standard deviation
- `min`: Minimum y value
- `max`: Maximum y value

**Raises:** `NotFittedError` if called before fit()

**Example:**
```python
bins = binner.bins_()
print(bins[['left', 'right', 'n', 'mean']])
```

### `summary_() -> pd.DataFrame`
Returns the full binning summary including WoE/IV for binary targets.

**Returns:** DataFrame with columns:
- `bucket`: Bin label (e.g., "[-inf, 25.5)", "Missing", "Excluded:-999")
- `count`: Number of samples
- `count_pct`: Percentage of total samples
- `sum`: Sum of y values (events for binary)
- `mean`: Mean of y (event rate for binary)
- `std`: Standard deviation
- `min/max`: Range of y values
- `woe`: Weight of Evidence (binary only)
- `iv`: Information Value contribution (binary only)

**Raises:** `NotFittedError` if called before fit()

**Example:**
```python
summary = binner.summary_()
print(f"Total IV: {summary['iv'].sum():.4f}")
```

### `transform(x_values, assign="interval") -> pd.Series`
Transforms raw x values to bin assignments.

**Parameters:**
- **x_values** (`pd.Series`): Values to transform
- **assign** (`Literal["interval", "left", "right", "woe"]`): Type of assignment
  - `"interval"`: Bin label like "[10, 20)" or "(-inf, 5)"
  - `"left"`: Left edge of the bin
  - `"right"`: Right edge of the bin
  - `"woe"`: Weight of Evidence (binary targets only)

**Returns:** Series with assigned values

**Example:**
```python
# Get interval labels
labels = binner.transform(df['x'], assign='interval')

# Get left edges for numerical operations
left_edges = binner.transform(df['x'], assign='left')

# Get WoE values for scoring (binary targets)
woe_values = binner.transform(df['x'], assign='woe')
```

## Usage Examples

### Basic Usage with Binary Target
```python
from MOBPY.binning import MonotonicBinner

binner = MonotonicBinner(df, x='age', y='default')
binner.fit()

bins = binner.bins_()
summary = binner.summary_()  # Includes WoE/IV
```

### Custom Constraints
```python
from MOBPY.core import BinningConstraints

constraints = BinningConstraints(
    max_bins=5,
    min_samples=0.05,     # 5% of data per bin
    min_positives=0.01    # 1% of positives per bin
)

binner = MonotonicBinner(
    df, x='income', y='approved',
    constraints=constraints,
    exclude_values=[-999, -1]  # Special codes
)
binner.fit()
```

### Continuous Target
```python
# Works with continuous targets too
binner = MonotonicBinner(
    df, x='sqft', y='price',
    sign='+'  # Expect positive correlation
)
binner.fit()

# Summary won't have WoE/IV for continuous targets
summary = binner.summary_()
```

## Edge Convention
- Bins use half-open intervals `[left, right)`
- First bin: `(-∞, right)` 
- Last bin: `[left, +∞)`
- This ensures complete coverage for any future x value

## Error Handling

The class includes comprehensive error handling:

1. **Data Validation**: Checks for missing columns, invalid data types
2. **Constraint Validation**: Ensures constraints are feasible
3. **Fitting Errors**: Handles PAVA convergence issues
4. **Transform Errors**: Validates input for transformation

## Integration with Other Components

- **`BinningConstraints`**: Configuration for constraints
- **`PAVA`**: Core monotonization algorithm
- **`merge_adjacent`**: Block merging logic
- **`partition_df`**: Data partitioning utility
- **`compute_woe_iv`**: WoE/IV calculation for binary targets

## Performance Characteristics

- **Time Complexity**: O(n log n) for sorting + O(n) for PAVA + O(k²) for merging
- **Space Complexity**: O(n) for data storage
- **Scalability**: Handles 10² to 10⁶ samples efficiently

## Dependencies
- pandas
- numpy
- scipy
- MOBPY.core modules
- MOBPY.exceptions

## See Also
- [`BinningConstraints`](../core/constraints.md) - Constraint configuration
- [`PAVA`](../core/pava.md) - Core PAVA algorithm
- [`merge_adjacent`](../core/merge.md) - Block merging strategies
- [`MOBPlot`](../plot/mob_plot.md) - Visualization utilities
