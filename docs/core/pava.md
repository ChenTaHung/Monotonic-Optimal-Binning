# PAVA (Pool-Adjacent-Violators Algorithm) Documentation

## Overview
The PAVA class implements the Pool-Adjacent-Violators Algorithm for isotonic regression on grouped data. It creates monotone (isotonic) step functions by pooling adjacent groups that violate monotonicity constraints. This implementation is optimized for monotonic binning use cases with O(n) complexity using a stack-based approach.

## Module Location
`src/MOBPY/core/pava.py`

## Class Definition

```python
class PAVA:
    def __init__(
        self,
        *,
        df: pd.DataFrame,
        x: str,
        y: str,
        metric: str = "mean",
        sign: str = "auto",
        strict: bool = True,
        sort_kind: Optional[str] = "quicksort"
    )
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **df** | `pd.DataFrame` | required | Input DataFrame containing x and y columns |
| **x** | `str` | required | Name of the feature column to group by |
| **y** | `str` | required | Name of the target column to aggregate |
| **metric** | `str` | `"mean"` | Aggregation metric (only 'mean' currently supported) |
| **sign** | `str` | `"auto"` | Monotonicity direction: '+' (increasing), '-' (decreasing), or 'auto' (infer from data) |
| **strict** | `bool` | `True` | If True, merge equal-mean blocks to ensure strict monotonicity (no plateaus) |
| **sort_kind** | `Optional[str]` | `"quicksort"` | Pandas sorting algorithm. Options: None, "quicksort", "mergesort", "heapsort", "stable" |

## Attributes

### Public Attributes (After Fitting)
| Attribute | Type | Description |
|-----------|------|-------------|
| **blocks_** | `List[_Block]` | List of monotone blocks after fitting |
| **groups_** | `pd.DataFrame` | DataFrame of grouped statistics with cumulative columns |
| **resolved_sign_** | `Literal["+", "-"]` | Actual monotonicity direction used ('+' or '-') |

### Private Attributes
| Attribute | Type | Description |
|-----------|------|-------------|
| **_n_merges** | `int` | Number of merges performed during PAVA |
| **_n_initial_groups** | `int` | Number of unique x values before pooling |

## Internal Classes

### `_Block` Dataclass
Internal representation of a monotone block with sufficient statistics:

```python
@dataclass
class _Block:
    left: float          # Left edge (inclusive)
    right: float         # Right edge (exclusive)
    n: int               # Number of observations
    sum: float           # Sum of y values
    sum2: float          # Sum of y² values
    ymin: float          # Minimum y value
    ymax: float          # Maximum y value
    merge_count: int     # Number of merges (tracking)
    original_groups: List[float]  # Original x values
```

**Properties:**
- `mean`: Block mean (sum/n)
- `var`: Sample variance using Welford's method
- `std`: Standard deviation

**Methods:**
- `merge_with(other)`: Merge with another block, pooling statistics
- `as_dict()`: Export as dictionary for serialization

## Main Methods

### `fit() -> PAVA`
Runs PAVA to create monotone blocks.

**Algorithm Steps:**
1. Group data by x and compute statistics
2. Determine monotonicity direction (if auto)
3. Apply PAVA using stack-based pooling
4. Optionally merge equal-mean blocks (strict mode)
5. Store results with cumulative statistics

**Returns:** Self for method chaining

**Raises:**
- `DataError`: If x/y columns missing or invalid
- `FittingError`: If PAVA fails to converge

**Example:**
```python
pava = PAVA(df=data, x='feature', y='target', sign='auto')
pava.fit()
```

### `export_blocks(as_dict=True) -> List`
Exports fitted blocks in specified format.

**Parameters:**
- **as_dict** (`bool`): If True, return list of dicts. If False, return list of tuples

**Returns:** List of blocks in requested format

**Example:**
```python
# Export as dictionaries
blocks = pava.export_blocks(as_dict=True)
# Each block: {'left': ..., 'right': ..., 'n': ..., 'mean': ..., ...}

# Export as tuples (backward compatibility)
blocks = pava.export_blocks(as_dict=False)
# Each block: (left, right, n, sum, sum2, ymin, ymax, mean, std, var)
```

## Algorithm Details

### Stack-Based PAVA Implementation
The algorithm uses a stack to achieve O(n) complexity:

```python
1. Initialize empty stack
2. For each block in sorted order:
   a. Push block onto stack
   b. While top two blocks violate monotonicity:
      - Merge top two blocks
      - Replace with merged block
   c. Continue to next block
3. Return stack contents
```

**Complexity:**
- Time: O(n) - each block pushed/popped at most once
- Space: O(n) - stack storage

### Monotonicity Detection (Auto Mode)
When `sign="auto"`, the direction is inferred by:
1. Computing correlation between x and y means
2. Positive correlation → increasing ('+')
3. Negative correlation → decreasing ('-')

### Strict Monotonicity Enforcement
When `strict=True`, blocks with equal means are merged:
- Removes plateaus in the step function
- Ensures each block has a unique mean
- Useful for creating distinct bins

## Edge Handling

**Critical:** PAVA ensures full real-line coverage:
- First block: `left = -∞`
- Last block: `right = +∞`
- Guarantees any future x value can be assigned to a block

## Usage Examples

### Basic Usage
```python
from MOBPY.core.pava import PAVA
import pandas as pd

# Create sample data
df = pd.DataFrame({
    'feature': [1, 2, 3, 4, 5],
    'target': [0.1, 0.3, 0.2, 0.5, 0.6]
})

# Run PAVA with auto-detected direction
pava = PAVA(df=df, x='feature', y='target', sign='auto')
pava.fit()

print(f"Direction: {pava.resolved_sign_}")
print(f"Created {len(pava.blocks_)} blocks")
```

### Enforcing Direction
```python
# Force increasing monotonicity
pava = PAVA(df=df, x='age', y='default_rate', sign='+')
pava.fit()

# Force decreasing monotonicity
pava = PAVA(df=df, x='credit_score', y='risk', sign='-')
pava.fit()
```

### Non-Strict Monotonicity
```python
# Allow plateaus (blocks with equal means)
pava = PAVA(
    df=df, 
    x='feature', 
    y='target', 
    sign='auto',
    strict=False  # Allow equal means
)
pava.fit()
```

### Accessing Results
```python
# Get grouped statistics
groups = pava.groups_
print(groups[['x', 'count', 'mean', 'cumsum']])

# Get blocks for further processing
blocks = pava.export_blocks(as_dict=True)
for block in blocks:
    print(f"Range: [{block['left']}, {block['right']}), "
          f"Mean: {block['mean']:.3f}")
```

## Numerical Stability

The implementation includes several numerical stability features:

1. **Welford's Method**: For variance calculation
2. **Epsilon Comparisons**: Uses configurable epsilon for floating-point comparisons
3. **Non-negative Variance**: Ensures variance is never negative due to numerical errors
4. **Protected Division**: Safe handling of zero-division cases

## Performance Characteristics

- **Grouping**: O(n log n) for sorting, O(n) for aggregation
- **PAVA**: O(k) where k is number of unique x values
- **Memory**: O(k) for storing blocks
- **Scalability**: Handles 10⁶+ samples efficiently through grouping

## Integration with Other Components

- **MonotonicBinner**: Uses PAVA as first step in binning pipeline
- **merge_adjacent**: Takes PAVA blocks as input for constraint-based merging
- **plot_csd_gcm**: Visualizes PAVA results with CSD/GCM plots

## Configuration

PAVA respects global configuration from `MOBPYConfig`:
- `epsilon`: Tolerance for floating-point comparisons
- `validate_inputs`: Whether to validate data types
- `enable_progress_bar`: Show progress during fitting

## Error Handling

Common errors and their causes:

1. **DataError**: Missing columns, invalid data types
2. **FittingError**: Convergence issues, numerical instabilities
3. **ValueError**: Invalid metric or sort_kind

## Best Practices

1. **Use Auto Sign**: Let PAVA detect monotonicity direction unless you have domain knowledge
2. **Keep Strict=True**: For binning, strict monotonicity usually produces better bins
3. **Check Groups**: Examine `groups_` DataFrame to understand data distribution
4. **Handle Edge Cases**: Be aware of behavior with constant y values

## Dependencies
- numpy
- pandas
- scipy
- MOBPY.exceptions
- MOBPY.config
- MOBPY.logging_utils

## See Also
- [`MonotonicBinner`](../binning/mob.md) - Main binning orchestrator
- [`merge_adjacent`](./merge.md) - Block merging after PAVA
- [`BinningConstraints`](./constraints.md) - Constraint specifications
- [`plot_csd_gcm`](../plot/csd_gcm.md) - Visualization of PAVA results
