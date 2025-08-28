# `BinningConstraints` Class Documentation

## Overview
The BinningConstraints class defines and manages constraints for the binning process. It supports both fractional (percentage-based) and absolute constraints, automatically resolving them based on the actual data size during fitting.

## Module Location
`src/MOBPY/core/constraints.py`

## Class Definition

```python
@dataclass
class BinningConstraints:
    max_bins: int = 6
    min_bins: int = 4
    max_samples: Optional[Union[float, int]] = None
    min_samples: Optional[Union[float, int]] = None
    min_positives: Optional[Union[float, int]] = None
    initial_pvalue: float = 0.4
    maximize_bins: bool = True
```

## Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **max_bins** | `int` | `6` | Maximum number of bins to create |
| **min_bins** | `int` | `4` | Minimum number of bins to maintain |
| **max_samples** | `Optional[Union[float, int]]` | `None` | Maximum samples per bin. If float in (0,1], treated as fraction |
| **min_samples** | `Optional[Union[float, int]]` | `None` | Minimum samples per bin. If float in (0,1], treated as fraction |
| **min_positives** | `Optional[Union[float, int]]` | `None` | Minimum positive samples per bin (binary targets only) |
| **initial_pvalue** | `float` | `0.4` | Initial p-value threshold for merging (annealed during search) |
| **maximize_bins** | `bool` | `True` | If True, create as many bins as possible within constraints |

## Resolved Attributes

After calling `resolve()`, these attributes become available:

| Attribute | Type | Description |
|-----------|------|-------------|
| **abs_max_samples** | `Optional[int]` | Absolute maximum samples per bin |
| **abs_min_samples** | `int` | Absolute minimum samples per bin |
| **abs_min_positives** | `int` | Absolute minimum positives per bin |

## Main Methods

### `resolve(total_n: int, total_pos: Optional[int] = None) -> None`
Converts fractional constraints to absolute values based on data size.

**Parameters:**
- **total_n** (`int`): Total number of samples in clean data
- **total_pos** (`Optional[int]`): Total number of positive samples (for binary targets)

**Behavior:**
- Fractional values (0 < value ≤ 1) are converted to absolute counts
- Absolute values (> 1) remain unchanged
- Sets resolved attributes (abs_min_samples, abs_max_samples, abs_min_positives)
- Validates that min ≤ max for all constraints

**Raises:**
- `ConstraintError`: If constraints are invalid or contradictory

**Example:**
```python
constraints = BinningConstraints(
    min_samples=0.05,    # 5% of data
    max_samples=0.30,    # 30% of data
    min_positives=0.01   # 1% of positives
)

# Resolve for actual data size
constraints.resolve(total_n=1000, total_pos=200)

# Now we have absolute values
print(f"Min samples: {constraints.abs_min_samples}")  # 50
print(f"Max samples: {constraints.abs_max_samples}")  # 300
print(f"Min positives: {constraints.abs_min_positives}")  # 2
```

### `validate() -> None`
Validates that all constraints are consistent and feasible.

**Checks:**
- All constraint values are non-negative
- `initial_pvalue` is in (0, 1]
- `min_bins ≤ max_bins`
- `min_samples ≤ max_samples` (after resolution)

**Raises:**
- `ConstraintError`: If validation fails

**Example:**
```python
constraints = BinningConstraints(
    min_bins=2,
    max_bins=6,
    min_samples=0.05,
    initial_pvalue=0.4
)

# Validate on creation
constraints.validate()  # Passes

# Invalid constraints
bad_constraints = BinningConstraints(
    min_bins=10,
    max_bins=5  # min > max!
)
bad_constraints.validate()  # Raises ConstraintError
```

### `copy() -> BinningConstraints`
Creates a deep copy of the constraints.

**Returns:**
- `BinningConstraints`: New instance with same values

**Example:**
```python
original = BinningConstraints(max_bins=5, min_samples=0.1)
copy = original.copy()

# Modify copy without affecting original
copy.max_bins = 10
assert original.max_bins == 5
```

### `is_resolved() -> bool`
Property indicating whether constraints have been resolved.

**Returns:**
- `bool`: True if `resolve()` has been called, False otherwise

**Example:**
```python
constraints = BinningConstraints()
assert not constraints.is_resolved

constraints.resolve(total_n=1000)
assert constraints.is_resolved
```

### `__str__() -> str`
Returns a human-readable string representation.

**Example:**
```python
constraints = BinningConstraints(max_bins=5, min_samples=0.1)
print(constraints)
# Output: BinningConstraints(max_bins=5, min_bins=4, min_samples=0.1, ...)
```

## Usage Patterns

### Basic Usage with Fractions
```python
from MOBPY.core import BinningConstraints

# Use fractions for adaptive constraints
constraints = BinningConstraints(
    max_bins=6,
    min_samples=0.05,    # Each bin gets at least 5% of data
    min_positives=0.01   # Each bin gets at least 1% of positives
)

# Resolution happens automatically during fit
# Or manually:
constraints.resolve(total_n=1000, total_pos=200)
print(f"Min samples per bin: {constraints.abs_min_samples}")  # 50
print(f"Min positives per bin: {constraints.abs_min_positives}")  # 2
```

### Absolute Values
```python
# Use absolute values for fixed constraints
constraints = BinningConstraints(
    max_bins=5,
    min_samples=100,     # Each bin needs at least 100 samples
    max_samples=1000     # No bin can exceed 1000 samples
)

# Absolute values remain unchanged after resolution
constraints.resolve(total_n=5000, total_pos=1000)
print(f"Min samples: {constraints.abs_min_samples}")  # 100
print(f"Max samples: {constraints.abs_max_samples}")  # 1000
```

### Mixed Fractional and Absolute
```python
# Mix fractional and absolute constraints
constraints = BinningConstraints(
    min_samples=0.1,     # Fractional: 10% of data
    max_samples=200,     # Absolute: max 200 samples
    min_positives=0.05   # Fractional: 5% of positives
)

constraints.resolve(total_n=1000, total_pos=400)
# abs_min_samples = 100 (10% of 1000)
# abs_max_samples = 200 (unchanged)
# abs_min_positives = 20 (5% of 400)
```

### Maximize vs Minimize Strategy
```python
# Maximize bins: Try to create as many bins as possible
constraints_max = BinningConstraints(
    max_bins=6,
    min_bins=2,
    maximize_bins=True   # Default
)

# Minimize bins: Try to create as few bins as possible
constraints_min = BinningConstraints(
    max_bins=6,
    min_bins=2,
    maximize_bins=False
)
```

## Constraint Interaction

### With Merging Process
- **maximize_bins=True**: Merge stops when reaching `max_bins` or when constraints prevent further merging
- **maximize_bins=False**: Merge continues until reaching `min_bins` or constraints force stopping

### With PAVA
- Constraints don't affect PAVA directly
- PAVA creates initial monotonic blocks
- Constraints guide subsequent merging

### Priority Order
When constraints conflict, the priority is:
1. **min_samples**: Ensures statistical reliability
2. **min_positives**: Ensures class representation (binary only)
3. **max_bins**: Limits complexity
4. **min_bins**: Ensures sufficient granularity
5. **max_samples**: Prevents oversized bins

## Error Handling

### Common Errors

1. **Negative Values**
```python
# Raises ConstraintError
BinningConstraints(min_samples=-0.1)
```

2. **Invalid p-value**
```python
# Raises ConstraintError (must be in (0, 1])
BinningConstraints(initial_pvalue=0.0)
BinningConstraints(initial_pvalue=1.5)
```

3. **Contradictory Constraints**
```python
constraints = BinningConstraints(
    min_samples=200,    # Absolute
    max_samples=0.1     # 10% fractional
)
# Raises ConstraintError during resolve if min > max
constraints.resolve(total_n=1000)  # min=200 > max=100
```

4. **Infeasible Combinations**
```python
constraints = BinningConstraints(
    min_bins=5,
    min_samples=0.3  # 30% each = 150% total!
)
# Will fail during merging if data size makes it impossible
```

## Advanced Features

### Dynamic P-value Annealing
The `initial_pvalue` is automatically adjusted during merging:
```python
# Starts at initial_pvalue
current_pvalue = constraints.initial_pvalue

# If no valid merges found, reduces by factor
while no_valid_merges and current_pvalue > 0.01:
    current_pvalue *= 0.9  # Anneal
    # Retry merge selection
```

### Constraint Relaxation
When strict constraints cannot be satisfied:
```python
# System may relax constraints with warnings
if cannot_satisfy_min_samples:
    logger.warning(
        "Cannot satisfy min_samples=%d, relaxing to %d",
        original_min, relaxed_min
    )
```

## Best Practices

1. **Use Fractions for Portability**: Fractional constraints adapt to different dataset sizes
2. **Set Reasonable Defaults**: Start with default values and adjust based on results
3. **Consider Data Size**: Ensure constraints are feasible for your data size
4. **Binary vs Continuous**: Use `min_positives` only for binary targets
5. **Validate Early**: Constraints are validated at initialization and resolution
6. **Monitor Warnings**: Pay attention to constraint relaxation warnings

## Integration Examples

### With MonotonicBinner
```python
from MOBPY import MonotonicBinner, BinningConstraints

constraints = BinningConstraints(
    max_bins=5,
    min_samples=0.05,
    min_positives=0.01
)

binner = MonotonicBinner(
    df, x='age', y='default',
    constraints=constraints  # Automatically resolved during fit
)
binner.fit()
```

### Manual Resolution
```python
# For testing or debugging
constraints = BinningConstraints(min_samples=0.1)

# Check before resolution
assert not constraints.is_resolved

# Manually resolve
n_clean = len(df)
n_positives = df['y'].sum()
constraints.resolve(total_n=n_clean, total_pos=n_positives)

# Check after resolution
assert constraints.is_resolved
assert constraints.abs_min_samples == int(0.1 * n_clean)
```

## Performance Notes

- **Resolution**: O(1) - simple arithmetic operations
- **Validation**: O(1) - direct comparisons
- **Copy**: O(1) - shallow copy of primitive values
- **No impact on binning algorithm performance**

## Thread Safety

BinningConstraints instances are thread-safe for read operations after resolution. For concurrent modifications, use separate instances or appropriate locking.

## Configuration Interaction

Constraints respect global configuration:
```python
from MOBPY.config import get_config

config = get_config()
# May use config.epsilon for numerical comparisons
# May use config.warn_on_small_bins for warnings
```

## Dependencies
- Python dataclasses
- typing module
- MOBPY.exceptions

## See Also
- [`MonotonicBinner`](../binning/mob.md) - Main binning orchestrator using constraints
- [`merge_adjacent`](./merge.md) - Block merging that respects constraints
- [`PAVA`](./pava.md) - Initial monotonic block creation
- [`Configuration`](../config.md) - Global configuration settings