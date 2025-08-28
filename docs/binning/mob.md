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
        sort_kind: Optional[str] = None
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
| **sort_kind** | `str` | `None` | Sorting algorithm |

## Key Methods

### fit()
Runs the complete binning pipeline.

**Pipeline Steps:**
1. Partition data by x values (clean/missing/excluded)
2. Check if y is binary on clean partition
3. Resolve constraints based on actual data size
4. Run PAVA to create initial monotonic blocks
5. Merge adjacent blocks to satisfy constraints
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

## Integration Points

- **BinningConstraints**: [Constraint configuration](../core/constraints.md)
- **PAVA**: [Core monotonization algorithm](../core/pava.md)
- **merge_adjacent**: [Block merging logic](../core/merge.md)
- **Plotting**: [Visualization utilities](../plot/init.md)

## See Also
- [Complete workflow example](../MOBPY-Overview.md#complete-workflow-example)
- [BinningConstraints](../core/constraints.md)
- [Plot functions](../plot/init.md)