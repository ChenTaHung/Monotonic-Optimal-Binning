# PAVA (Pool-Adjacent-Violators Algorithm) Documentation

## Overview
The PAVA class implements the Pool-Adjacent-Violators Algorithm for isotonic regression. It creates monotonic blocks from data by pooling adjacent violators, ensuring the mean values follow a specified monotonic direction (increasing or decreasing).

## Module Location
`src/MOBPY/core/pava.py`

## Class Definition

```python
class PAVA:
    def __init__(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        *,
        metric: Literal["mean"] = "mean",
        sign: Literal["+", "-", "auto"] = "auto",
        strict: bool = True,
        sort_kind: Optional[str] = "quicksort"
    )
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **df** | `pd.DataFrame` | required | Input DataFrame with feature and target |
| **x** | `str` | required | Name of feature column to group by |
| **y** | `str` | required | Name of target column to aggregate |
| **metric** | `Literal["mean"]` | `"mean"` | Aggregation metric (only "mean" currently supported) |
| **sign** | `Literal["+", "-", "auto"]` | `"auto"` | Monotonicity direction: "+" (increasing), "-" (decreasing), or "auto" (infer from data) |
| **strict** | `bool` | `True` | If True, enforce strict monotonicity (merge equal-mean blocks) |
| **sort_kind** | `Optional[str]` | `"quicksort"` | Pandas sorting algorithm. None uses pandas default |

## Attributes

### Public Attributes (after fitting)
| Attribute | Type | Description |
|-----------|------|-------------|
| **groups_** | `pd.DataFrame` | Grouped statistics for each unique x value |
| **resolved_sign_** | `Literal["+", "-"]` | Actual monotonicity direction used |
| **n_iterations_** | `int` | Number of PAVA iterations performed |
| **is_fitted_** | `bool` | Whether fit() has been called |

### Groups DataFrame Columns
After fitting, `groups_` contains:
- **x**: Unique x values (sorted)
- **count**: Number of samples per x value
- **sum**: Sum of y values
- **sum2**: Sum of y² values
- **ymin**: Minimum y value
- **ymax**: Maximum y value
- **cum_count**: Cumulative count
- **cum_sum**: Cumulative sum
- **cum_mean**: Cumulative mean
- **group_mean**: Mean y value per group

## Main Methods

### `fit() -> PAVA`
Runs the PAVA algorithm to create monotonic blocks.

**Algorithm Steps:**
1. Groups data by unique x values
2. Computes aggregated statistics per group
3. Infers sign if set to "auto"
4. Applies stack-based PAVA for O(n) complexity
5. Optionally enforces strict monotonicity
6. Stores results in internal blocks

**Returns:** Self for method chaining

**Raises:**
- `DataError`: If columns are missing or data is invalid
- `FittingError`: If PAVA fails to converge

**Example:**
```python
from MOBPY.core.pava import PAVA

pava = PAVA(df=data, x='age', y='default', sign='auto')
pava.fit()

print(f"Resolved sign: {pava.resolved_sign_}")
print(f"Iterations: {pava.n_iterations_}")
```

### `export_blocks(as_dict: bool = True) -> List[Union[Dict, Block]]`
Exports the monotonic blocks created by PAVA.

**Parameters:**
- **as_dict** (`bool`): If True, return list of dictionaries. If False, return Block objects

**Returns:** List of blocks with statistics

**Block Structure (as dict):**
```python
{
    'left': float,    # Left boundary (inclusive)
    'right': float,   # Right boundary (exclusive)
    'n': int,         # Number of samples
    'sum': float,     # Sum of y values
    'sum2': float,    # Sum of y² values
    'mean': float,    # Mean of y values
    'std': float,     # Standard deviation
    'ymin': float,    # Minimum y value
    'ymax': float     # Maximum y value
}
```

**Example:**
```python
# Get blocks as dictionaries
blocks = pava.export_blocks(as_dict=True)
for block in blocks:
    print(f"[{block['left']}, {block['right']}): mean={block['mean']:.4f}")

# Get blocks as Block objects
blocks = pava.export_blocks(as_dict=False)
for block in blocks:
    print(f"Block mean: {block.mean}, n={block.n}")
```

### `validate_monotonicity(tolerance: float = 1e-10) -> bool`
Validates that the fitted blocks are monotonic.

**Parameters:**
- **tolerance** (`float`): Numerical tolerance for comparisons

**Returns:** True if blocks satisfy monotonicity constraint

**Example:**
```python
if pava.validate_monotonicity():
    print("✓ Blocks are monotonic")
else:
    print("✗ Monotonicity violation detected")
```

### `get_diagnostics() -> Dict[str, Any]`
Returns diagnostic information about the fitting process.

**Returns:** Dictionary with diagnostics:
```python
{
    'n_unique_x': int,           # Number of unique x values
    'n_blocks': int,             # Number of final blocks
    'n_iterations': int,         # PAVA iterations
    'resolved_sign': str,        # '+' or '-'
    'compression_ratio': float,  # n_unique_x / n_blocks
    'strict_applied': bool       # Whether strict monotonicity was enforced
}
```

**Example:**
```python
diag = pava.get_diagnostics()
print(f"Compression: {diag['n_unique_x']} -> {diag['n_blocks']} blocks")
print(f"Ratio: {diag['compression_ratio']:.2f}x")
```

## Internal Classes

### `_Block`
Internal class for maintaining block statistics during PAVA.

**Attributes:**
- **indices**: List of group indices in this block
- **n**: Total sample count
- **sum**: Sum of y values
- **sum2**: Sum of y² values
- **ymin**: Minimum y value
- **ymax**: Maximum y value

**Methods:**
- **mean**: Property returning block mean
- **merge_with(other)**: Merge with another block, pooling statistics

## Algorithm Details

### Stack-Based PAVA Implementation
```python
# Pseudocode for the core algorithm
stack = []
for group in groups:
    block = create_block(group)
    stack.append(block)
    
    # Pool violators
    while len(stack) >= 2:
        if violates_monotonicity(stack[-2], stack[-1], sign):
            merged = stack[-2].merge_with(stack[-1])
            stack = stack[:-2] + [merged]
        else:
            break
```

**Complexity:**
- Time: O(n) where n is number of unique x values
- Space: O(n) for the stack
- Guaranteed to converge in one pass

### Sign Inference
When `sign="auto"`, PAVA infers monotonicity from data:
```python
correlation = calculate_correlation(x_values, y_means)
if correlation >= 0:
    sign = "+"  # Non-decreasing
else:
    sign = "-"  # Non-increasing
```

### Strict Monotonicity
When `strict=True`, blocks with equal means are merged:
```python
if abs(block1.mean - block2.mean) < epsilon:
    merge_blocks(block1, block2)
```

## Usage Examples

### Basic PAVA Fitting
```python
from MOBPY.core.pava import PAVA
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
df = pd.DataFrame({
    'x': np.random.uniform(0, 100, 1000),
    'y': np.random.binomial(1, 0.3, 1000)
})

# Fit PAVA
pava = PAVA(df=df, x='x', y='y', sign='auto')
pava.fit()

# Get results
blocks = pava.export_blocks(as_dict=True)
print(f"Created {len(blocks)} monotonic blocks")
```

### Manual Sign Control
```python
# Force increasing monotonicity
pava_inc = PAVA(df=df, x='x', y='y', sign='+')
pava_inc.fit()

# Force decreasing monotonicity  
pava_dec = PAVA(df=df, x='x', y='y', sign='-')
pava_dec.fit()
```

### Working with Groups
```python
# Access grouped data
pava = PAVA(df=df, x='x', y='y')
pava.fit()

# Examine groups before pooling
groups = pava.groups_
print(f"Unique x values: {len(groups)}")
print(f"Mean range: [{groups['group_mean'].min():.4f}, {groups['group_mean'].max():.4f}]")

# Compare to blocks after pooling
blocks = pava.export_blocks()
print(f"Blocks after PAVA: {len(blocks)}")
```

### Validation and Diagnostics
```python
# Comprehensive validation
pava = PAVA(df=df, x='feature', y='target')
pava.fit()

# Check monotonicity
assert pava.validate_monotonicity(), "Monotonicity violated!"

# Get detailed diagnostics
diag = pava.get_diagnostics()
print(f"""
PAVA Diagnostics:
  Unique x values: {diag['n_unique_x']}
  Final blocks: {diag['n_blocks']}
  Compression: {diag['compression_ratio']:.2f}x
  Iterations: {diag['n_iterations']}
  Direction: {diag['resolved_sign']}
  Strict monotonicity: {diag['strict_applied']}
""")
```

## Numerical Stability

### Variance Calculation
Uses Welford's stable algorithm:
```python
# Instead of: var = E[X²] - E[X]²
# Use: var = (sum2 - sum²/n) / (n-1)
```

### Epsilon Comparisons
Uses configurable epsilon for floating-point comparisons:
```python
from MOBPY.config import get_config
config = get_config()

# For monotonicity checks
if sign == '+':
    is_monotonic = all(b1.mean <= b2.mean + config.epsilon 
                      for b1, b2 in zip(blocks[:-1], blocks[1:]))
```

### Protected Operations
- Division by zero protection in mean calculations
- Non-negative variance guarantee
- Infinity handling in boundaries

## Performance Characteristics

- **Grouping**: O(n log n) for sorting, O(n) for aggregation
- **PAVA**: O(k) where k is number of unique x values
- **Memory**: O(k) for storing blocks
- **Scalability**: Handles 10⁶+ samples efficiently through grouping

### Optimization Techniques
1. **Pre-grouping**: Aggregates by unique x before PAVA
2. **Stack-based**: Single-pass algorithm
3. **Sufficient Statistics**: Maintains sum/sum2 for O(1) merges
4. **Vectorized Operations**: Uses pandas/numpy where possible

## Edge Cases

### Constant Y Values
```python
# All y values are the same
df = pd.DataFrame({'x': [1, 2, 3], 'y': [0.5, 0.5, 0.5]})
pava = PAVA(df=df, x='x', y='y')
pava.fit()
# Results in single block with mean=0.5
```

### Single Unique X
```python
# Only one unique x value
df = pd.DataFrame({'x': [1, 1, 1], 'y': [0, 1, 0]})
pava = PAVA(df=df, x='x', y='y')
pava.fit()
# Results in single block
```

### Perfect Monotonicity
```python
# Already monotonic data
df = pd.DataFrame({'x': [1, 2, 3], 'y': [0.1, 0.5, 0.9]})
pava = PAVA(df=df, x='x', y='y', sign='+')
pava.fit()
# Each group becomes its own block (no pooling needed)
```

## Integration with Other Components

- **MonotonicBinner**: Uses PAVA as first step in binning pipeline
- **merge_adjacent**: Takes PAVA blocks as input for constraint-based merging
- **plot_csd_gcm**: Visualizes PAVA results with CSD/GCM plots

## Configuration

PAVA respects global configuration from `MOBPYConfig`:
```python
from MOBPY.config import get_config, set_config

# Set custom epsilon
set_config(epsilon=1e-10)

# PAVA will use this for comparisons
pava = PAVA(df=df, x='x', y='y')
pava.fit()
```

## Error Handling

Common errors and their causes:

1. **DataError: Column not found**
```python
# Wrong column name
pava = PAVA(df, x='nonexistent', y='y')  # Raises DataError
```

2. **DataError: Non-numeric data**
```python
# Non-numeric x or y
df = pd.DataFrame({'x': ['a', 'b'], 'y': [1, 2]})
pava = PAVA(df, x='x', y='y')  # Raises DataError during fit
```

3. **ValueError: Invalid metric**
```python
# Currently only 'mean' is supported
pava = PAVA(df, x='x', y='y', metric='median')  # Raises ValueError
```

## Best Practices

1. **Use Auto Sign**: Let PAVA detect monotonicity direction unless you have domain knowledge
2. **Keep Strict=True**: For binning, strict monotonicity usually produces better bins
3. **Check Groups**: Examine `groups_` DataFrame to understand data distribution
4. **Validate Results**: Always check monotonicity after fitting
5. **Handle Edge Cases**: Be aware of behavior with constant y values or single x values

## Thread Safety

PAVA instances are not thread-safe during fitting. For parallel processing:
- Create separate PAVA instances per thread
- Or use appropriate locking mechanisms

## Dependencies
- numpy
- pandas
- scipy (for correlation calculation)
- MOBPY.exceptions
- MOBPY.config
- MOBPY.logging_utils
- MOBPY.core.utils (for calculate_correlation)

## See Also
- [`MonotonicBinner`](../binning/mob.md) - Main binning orchestrator using PAVA
- [`merge_adjacent`](./merge.md) - Block merging after PAVA
- [`BinningConstraints`](./constraints.md) - Constraint specifications
- [`plot_csd_gcm`](../plot/csd_gcm.md) - Visualization of PAVA results
- [`calculate_correlation`](./utils.md#calculate_correlation) - Used for sign inference