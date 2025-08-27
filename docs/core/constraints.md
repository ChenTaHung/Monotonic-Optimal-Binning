# BinningConstraints Class Documentation

## Overview
The `BinningConstraints` class encapsulates all user-defined constraints for the binning process. It provides validation and automatic resolution of constraints from fractional to absolute values based on the actual data size at fit time.

## Module Location
`src/MOBPY/core/constraints.py`

## Class Definition

```python
@dataclass
class BinningConstraints:
    max_bins: int = 6
    min_bins: int = 4
    max_samples: Optional[float] = None
    min_samples: Optional[float] = None
    min_positives: Optional[float] = None
    initial_pvalue: float = 0.4
    maximize_bins: bool = True
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **max_bins** | `int` | `6` | Maximum number of bins allowed. Must be ≥ 1 |
| **min_bins** | `int` | `4` | Minimum number of bins to maintain. Must be ≥ 1 and ≤ max_bins |
| **max_samples** | `Optional[float]` | `None` | Maximum samples per bin. If in (0,1], treated as fraction. If > 1, treated as absolute count. None means no upper limit |
| **min_samples** | `Optional[float]` | `None` | Minimum samples per bin. If in (0,1], treated as fraction. If > 1, treated as absolute count. None defaults to 0 |
| **min_positives** | `Optional[float]` | `None` | Minimum positive samples per bin (binary targets only). If in (0,1], treated as fraction of total positives. If > 1, treated as absolute count. None defaults to 0 |
| **initial_pvalue** | `float` | `0.4` | Initial p-value threshold for merge decisions. Higher values make merging more aggressive. Must be in (0, 1] |
| **maximize_bins** | `bool` | `True` | If True, prioritize staying at/below max_bins. If False, prioritize staying at/above min_bins |

## Resolved Attributes

After calling `resolve()`, these attributes contain absolute values:

| Attribute | Type | Description |
|-----------|------|-------------|
| **abs_max_samples** | `Optional[int]` | Resolved absolute maximum samples per bin |
| **abs_min_samples** | `int` | Resolved absolute minimum samples per bin |
| **abs_min_positives** | `int` | Resolved absolute minimum positives per bin |
| **_resolved** | `bool` | Internal flag tracking resolution state |

## Methods

### `__post_init__() -> None`
Validates constraints immediately after initialization.

**Raises:**
- `ConstraintError`: If initial constraints are invalid (e.g., negative values, invalid p-value range)

### `resolve(total_n: int, total_pos: Optional[int] = None) -> None`
Resolves fractional constraints to absolute values based on data size.

**Parameters:**
- **total_n** (`int`): Total number of clean samples
- **total_pos** (`Optional[int]`): Total number of positive samples (for binary targets)

**Resolution Rules:**
- Values in (0, 1] are treated as fractions and converted using floor: `int(value * total)`
- Values > 1 are treated as absolute counts and kept unchanged
- None values for minimums become 0
- None values for maximums remain None (no limit)

**Validation After Resolution:**
- Ensures `min_samples ≤ max_samples` (if both specified)
- Ensures `min_bins ≤ max_bins`
- Ensures all resolved values are non-negative

**Raises:**
- `ConstraintError`: If resolved constraints are contradictory

**Example:**
```python
constraints = BinningConstraints(
    min_samples=0.1,    # 10% of data
    min_positives=0.05  # 5% of positives
)
constraints.resolve(total_n=1000, total_pos=200)
# Now: abs_min_samples = 100, abs_min_positives = 10
```

### `validate() -> None`
Internal validation method called during initialization and after resolution.

**Checks:**
- All constraint values are non-negative
- `initial_pvalue` is in (0, 1]
- `min_bins ≤ max_bins`
- `min_samples ≤ max_samples` (after resolution)

### `is_resolved() -> bool`
Property indicating whether constraints have been resolved.

**Returns:** `True` if `resolve()` has been called, `False` otherwise

## Usage Examples

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

## Best Practices

1. **Use Fractions for Portability**: Fractional constraints adapt to different dataset sizes
2. **Set Reasonable Defaults**: Start with default values and adjust based on results
3. **Consider Data Size**: Ensure constraints are feasible for your data size
4. **Binary vs Continuous**: Use `min_positives` only for binary targets
5. **Validate Early**: Constraints are validated at initialization and resolution

## Performance Notes

- Resolution is O(1) - simple arithmetic operations
- Validation is O(1) - direct comparisons
- No impact on binning algorithm performance

## Integration Notes

- Used by `MonotonicBinner` during fit process
- Resolution happens automatically on clean data partition
- Guides the `merge_adjacent` function behavior
- Works with `MergeStrategy` enum for merge selection

## Dependencies
- Python dataclasses
- typing module
- MOBPY.exceptions

## See Also
- [`MonotonicBinner`](../binning/mob.md) - Main binning orchestrator
- [`merge_adjacent`](./merge.md) - Block merging using constraints
- [`PAVA`](./pava.md) - Initial monotonic block creation
