# BinningConstraints Class Documentation

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

## Methods

### resolve(total_n: int, total_pos: Optional[int] = None)
Converts fractional constraints to absolute values based on data size.

**Parameters:**
- `total_n`: Total number of samples in clean data
- `total_pos`: Total number of positive samples (for binary targets)

**Example:**
```python
constraints = BinningConstraints(
    min_samples=0.05,    # 5% of data
    max_samples=0.30,    # 30% of data
    min_positives=0.01   # 1% of positives
)

constraints.resolve(total_n=1000, total_pos=200)
print(f"Min samples: {constraints.abs_min_samples}")  # 50
print(f"Max samples: {constraints.abs_max_samples}")  # 300
```

### validate()
Validates that all constraints are consistent and feasible.

### copy()
Creates a deep copy of the constraints.

### is_resolved (property)
Returns True if constraints have been resolved to absolute values.

## Usage Patterns

### Fractional Constraints (Adaptive)
```python
# Use fractions for adaptive constraints
constraints = BinningConstraints(
    max_bins=6,
    min_samples=0.05,    # Each bin gets at least 5% of data
    min_positives=0.01   # Each bin gets at least 1% of positives
)
```

### Absolute Constraints (Fixed)
```python
# Use absolute values for fixed constraints
constraints = BinningConstraints(
    max_bins=5,
    min_samples=100,     # Each bin needs at least 100 samples
    max_samples=1000     # No bin can exceed 1000 samples
)
```

### Mixed Constraints
```python
# Mix fractional and absolute constraints
constraints = BinningConstraints(
    min_samples=0.1,     # Fractional: 10% of data
    max_samples=200,     # Absolute: max 200 samples
    min_positives=0.05   # Fractional: 5% of positives
)
```

## Constraint Priority

When constraints conflict, the priority is:
1. **min_samples**: Ensures statistical reliability
2. **min_positives**: Ensures class representation (binary only)
3. **max_bins**: Limits complexity
4. **min_bins**: Ensures sufficient granularity
5. **max_samples**: Prevents oversized bins

## Advanced Features

### Dynamic P-value Annealing
The `initial_pvalue` is automatically adjusted during merging if no valid merges are found, reducing by a factor until reaching 0.01.

### Constraint Relaxation
When strict constraints cannot be satisfied, the system may relax constraints with warnings.

## Best Practices

1. **Use Fractions for Portability**: Fractional constraints adapt to different dataset sizes
2. **Set Reasonable Defaults**: Start with default values and adjust based on results
3. **Consider Data Size**: Ensure constraints are feasible for your data size
4. **Binary vs Continuous**: Use `min_positives` only for binary targets
5. **Validate Early**: Constraints are validated at initialization and resolution

## Error Handling

Common errors and their solutions:

- **Negative Values**: Raises `ConstraintError` at initialization
- **Invalid p-value**: Must be in (0, 1], zero is not allowed
- **Contradictory Constraints**: Detected during resolution when min > max
- **Infeasible Combinations**: May fail during merging if data size makes constraints impossible

See also: [MonotonicBinner](../binning/mob.md) for usage within the binning pipeline