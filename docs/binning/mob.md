# MonotonicBinner Class Documentation

## Overview
The MonotonicBinner class is the main orchestrator for monotonic optimal binning. It manages the complete pipeline from data preprocessing through PAVA and merging to final bin creation with WoE/IV calculation.

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
        metric: Literal["mean"] = "mean",
        sign: Literal["+", "-", "auto"] = "auto",
        strict: bool = True,
        constraints: Optional[BinningConstraints] = None,
        exclude_values: Optional[Iterable] = None,
        merge_strategy: Union[MergeStrategy, str] = MergeStrategy.HIGHEST_PVALUE,
        sort_kind: Optional[str] = "quicksort"
    )
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **df** | `pd.DataFrame` | required | Input data |
| **x** | `str` | required | Feature column name |
| **y** | `str` | required | Target column name |
| **metric** | `Literal["mean"]` | `"mean"` | Aggregation metric |
| **sign** | `Literal["+", "-", "auto"]` | `"auto"` | Monotonicity direction |
| **strict** | `bool` | `True` | Enforce strict monotonicity |
| **constraints** | `BinningConstraints` | `None` | Binning constraints |
| **exclude_values** | `Iterable` | `None` | Values to exclude from binning |
| **merge_strategy** | `MergeStrategy` | `HIGHEST_PVALUE` | Merge selection strategy |
| **sort_kind** | `str` | `"quicksort"` | Sorting algorithm |

## Key Methods

### fit()
Runs the complete binning pipeline.

**Pipeline Steps:**
1. Partition data by x values (clean/missing/excluded)
2. Check if y is binary on clean partition
3. Resolve constraints based on actual data size
4. Run PAVA to create initial monotonic blocks
5. Merge adjacent blocks to satisfy constraints (including class count constraints)
6. Build final bins and summary DataFrame

**Returns:** Self for method chaining

**Raises:**
- `DataError`: If data has issues
- `FittingError`: If fitting fails

### bins_()
Returns the fitted bins as a DataFrame.

**Returns:** DataFrame with columns:
- `left`: Left bin edge (first is -inf)
- `right`: Right bin edge (last is +inf)
- `n`: Number of samples
- `sum`: Sum of y values
- `mean`: Mean of y values
- `std`: Standard deviation
- `min/max`: Range of y values

### summary_()
Returns full binning summary including WoE/IV for binary targets.

**Returns:** DataFrame with columns:
- `bucket`: Bin label
- `count`: Number of samples
- `count_pct`: Percentage of total
- `sum`: Sum of y values
- `mean`: Mean of y
- `std`: Standard deviation
- `min/max`: Range
- `woe`: Weight of Evidence (binary only)
- `iv`: Information Value (binary only)

### transform(x_values, assign="interval")
Transforms raw x values to bin assignments.

**Parameters:**
- `x_values`: Values to transform
- `assign`: Type of assignment ("interval", "left", "right", "woe")

**Returns:** Series with assigned values

### get_diagnostics()
Returns diagnostic information about the fitting process.

**Returns:** Dictionary with keys:
- `partition_summary`: Data partition statistics
- `is_binary`: Whether target is binary
- `resolved_sign`: Detected monotonicity direction
- `pava_diagnostics`: Diagnostic dict from the PAVA algorithm (see `PAVA.get_diagnostics()`)
- `n_pava_blocks`: Number of blocks from PAVA
- `n_final_bins`: Number of final bins
- `constraints_satisfied`: Dictionary of constraint satisfaction status

## Constraint Satisfaction Diagnostics (Updated in v2.2.0)

The `constraints_satisfied` dictionary now includes:

```python
diagnostics = binner.get_diagnostics()
cs = diagnostics['constraints_satisfied']

# Available keys:
cs['max_bins']       # bool: Did we stay within max_bins?
cs['min_bins']       # bool: Did we maintain min_bins?
cs['min_samples']    # bool: Do all bins meet min_samples?
cs['min_positives']  # bool: Do all bins meet min_positives? (binary only)
cs['min_negatives']  # bool: Do all bins meet min_negatives? (binary only, NEW in v2.2.0)
```

**Example:**
```python
binner = MonotonicBinner(
    df=df, x='amount', y='default',
    constraints=BinningConstraints(
        max_bins=6,
        min_positives=10,
        min_negatives=20
    )
)
binner.fit()

diag = binner.get_diagnostics()
print(f"Min positives satisfied: {diag['constraints_satisfied']['min_positives']}")
print(f"Min negatives satisfied: {diag['constraints_satisfied']['min_negatives']}")
```

## Usage Example with Class Count Constraints

```python
import pandas as pd
from MOBPY import MonotonicBinner
from MOBPY.core.constraints import BinningConstraints

# Load data
df = pd.read_csv('credit_data.csv')

# Create constraints for stable WoE calculations
constraints = BinningConstraints(
    max_bins=6,
    min_bins=2,
    min_samples=0.05,     # 5% of data per bin
    min_positives=0.02,   # 2% of defaults per bin
    min_negatives=0.02    # 2% of non-defaults per bin
)

# Create and fit binner
binner = MonotonicBinner(
    df=df,
    x='credit_amount',
    y='default',
    constraints=constraints
)
binner.fit()

# Check if constraints were satisfied
diag = binner.get_diagnostics()
print("Constraints satisfied:")
for k, v in diag['constraints_satisfied'].items():
    status = "✓" if v else "✗"
    print(f"  {status} {k}")

# Get summary with WoE/IV
summary = binner.summary_()
print(summary)
```

## Edge Convention
- Bins use half-open intervals `[left, right)`
- First bin: `(-∞, right)` 
- Last bin: `[left, +∞)`
- Ensures complete coverage for any future x value

## Error Handling

The class includes comprehensive error handling for:
1. **Data Validation**: Missing columns, invalid data types
2. **Constraint Validation**: Feasible constraints
3. **Fitting Errors**: PAVA convergence issues
4. **Transform Errors**: Input validation

## Performance Characteristics

- **Time Complexity**: O(n log n) sorting + O(n) PAVA + O(k²) merging
- **Space Complexity**: O(n) for data storage
- **Scalability**: Handles 10² to 10⁶ samples efficiently

## WoE Stability (v2.2.0)

For binary classification, stable Weight of Evidence requires non-zero counts of both classes in each bin:

$$WoE_i = \ln\left(\frac{\text{Distribution of Goods}_i}{\text{Distribution of Bads}_i}\right)$$

With `min_positives` and `min_negatives` constraints, you can ensure:
- No division by zero in WoE calculation
- No `log(0)` errors
- Statistically meaningful bin statistics

**Recommended settings for credit scoring:**
```python
constraints = BinningConstraints(
    max_bins=6,
    min_samples=0.05,
    min_positives=20,   # Or 0.02 for 2% of events
    min_negatives=50    # Or 0.02 for 2% of non-events
)
```

## Integration Points

- **BinningConstraints**: [Constraint configuration](../core/constraints.md)
- **PAVA**: [Core monotonization algorithm](../core/pava.md)
- **merge_adjacent**: [Block merging logic](../core/merge.md)
- **Plotting**: [Visualization utilities](../plot/init.md)

## See Also
- [Complete workflow example](../MOBPY-Overview.md#complete-workflow-example)
- [BinningConstraints](../core/constraints.md)
- [Plot functions](../plot/init.md)