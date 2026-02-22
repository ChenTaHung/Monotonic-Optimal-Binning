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
    min_negatives: Optional[Union[float, int]] = None  # NEW in v2.2.0
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
| **min_positives** | `Optional[Union[float, int]]` | `None` | Minimum positive samples (y=1) per bin (binary targets only) |
| **min_negatives** | `Optional[Union[float, int]]` | `None` | Minimum negative samples (y=0) per bin (binary targets only). *New in v2.2.0* |
| **initial_pvalue** | `float` | `0.4` | Initial p-value threshold for merging (annealed during search) |
| **maximize_bins** | `bool` | `True` | If True, create as many bins as possible within constraints |

## Resolved Attributes

After calling `resolve()`, the following absolute values are available:

| Attribute | Type | Description |
|-----------|------|-------------|
| **abs_max_samples** | `Optional[int]` | Resolved absolute maximum samples |
| **abs_min_samples** | `int` | Resolved absolute minimum samples |
| **abs_min_positives** | `int` | Resolved absolute minimum positives |
| **abs_min_negatives** | `int` | Resolved absolute minimum negatives. *New in v2.2.0* |

## Methods

### resolve(*, total_n: int, total_pos: int = 0)
Converts fractional constraints to absolute values based on data size.

**Parameters:**
- `total_n`: Total number of samples in clean data
- `total_pos`: Total number of positive samples (for binary targets). Defaults to `0` for non-binary targets.

**Feasibility Warnings (New in v2.2.0):**
The method now issues warnings when constraints are mathematically infeasible:
- When `min_samples` makes `min_bins` impossible
- When `min_positives` cannot be satisfied with available positives
- When `min_negatives` cannot be satisfied with available negatives

**Example:**
```python
constraints = BinningConstraints(
    min_samples=0.05,    # 5% of data
    max_samples=0.30,    # 30% of data
    min_positives=0.01,  # 1% of positives
    min_negatives=0.01   # 1% of negatives (NEW)
)

constraints.resolve(total_n=1000, total_pos=200)
print(f"Min samples: {constraints.abs_min_samples}")    # 50
print(f"Max samples: {constraints.abs_max_samples}")    # 300
print(f"Min positives: {constraints.abs_min_positives}")  # 2
print(f"Min negatives: {constraints.abs_min_negatives}")  # 8 (1% of 800)
```

### validate()
Validates that all constraints are consistent and feasible.

### copy()
Creates a deep copy of the constraints (includes `min_negatives`).

### is_resolved() -> bool
Returns True if constraints have been resolved to absolute values.

## Usage Patterns

### Fractional Constraints (Adaptive)
```python
# Use fractions for adaptive constraints
constraints = BinningConstraints(
    max_bins=6,
    min_samples=0.05,    # Each bin gets at least 5% of data
    min_positives=0.01,  # Each bin gets at least 1% of positives
    min_negatives=0.01   # Each bin gets at least 1% of negatives
)
```

### Absolute Constraints (Fixed)
```python
# Use absolute values for fixed constraints
constraints = BinningConstraints(
    max_bins=5,
    min_samples=100,     # Each bin needs at least 100 samples
    max_samples=1000,    # No bin can exceed 1000 samples
    min_positives=10,    # At least 10 positives per bin
    min_negatives=10     # At least 10 negatives per bin
)
```

### Mixed Constraints
```python
# Mix fractional and absolute constraints
constraints = BinningConstraints(
    min_samples=0.1,     # Fractional: 10% of data
    max_samples=200,     # Absolute: max 200 samples
    min_positives=0.05,  # Fractional: 5% of positives
    min_negatives=20     # Absolute: at least 20 negatives
)
```

### WoE Stability Constraints (Recommended for Credit Scoring)
```python
# Ensure stable Weight of Evidence calculations
# Both positives and negatives needed to avoid log(0) or division by zero
constraints = BinningConstraints(
    max_bins=6,
    min_bins=2,
    min_samples=0.05,
    min_positives=0.02,  # At least 2% of events per bin
    min_negatives=0.02   # At least 2% of non-events per bin
)
```

## Constraint Priority

When constraints conflict, the priority is:
1. **min_bins**: Hard floor - merging stops when this is reached
2. **min_samples**: Ensures statistical reliability
3. **min_positives**: Ensures positive class representation (binary only)
4. **min_negatives**: Ensures negative class representation (binary only)
5. **max_bins**: Limits complexity
6. **max_samples**: Prevents oversized bins

## Enforcement Behavior (v2.2.0)

### Active Enforcement
The following constraints are now **actively enforced** through merging:
- `min_samples`: Bins below threshold are merged with neighbors
- `min_positives`: Bins with insufficient positives are merged (binary targets)
- `min_negatives`: Bins with insufficient negatives are merged (binary targets)

### Hard Floor
`min_bins` acts as a hard floor - enforcement stops when this limit is reached, even if other constraints are not satisfied. Warnings are issued when constraints cannot be fully satisfied.

### Soft Penalties
During the statistical merge phase, the merge scorer applies bonuses (1.4x score multiplier) for merging bins that violate class count constraints, encouraging their merge even before the enforcement phase.

## Advanced Features

### Dynamic P-value Annealing
The `initial_pvalue` is automatically adjusted during merging if no valid merges are found, reducing by a factor until reaching 0.01.

### Constraint Relaxation
When strict constraints cannot be satisfied due to `min_bins` floor, the system issues warnings but continues with the best possible result.

### Feasibility Warnings
During `resolve()`, warnings are issued if:
```python
# Example warning scenarios:
# "With min_samples=100, only 3 bins are possible, but min_bins=5"
# "With min_positives=50 and total_pos=100, only 2 bins can satisfy the constraint"
# "With min_negatives=200 and total_neg=500, only 2 bins can satisfy the constraint"
```

## Best Practices

1. **Use Fractions for Portability**: Fractional constraints adapt to different dataset sizes
2. **Set Reasonable Defaults**: Start with default values and adjust based on results
3. **Consider Data Size**: Ensure constraints are feasible for your data size
4. **Binary Targets**: Use both `min_positives` AND `min_negatives` for stable WoE
5. **Validate Early**: Constraints are validated at initialization and resolution
6. **Check Diagnostics**: After fitting, check `get_diagnostics()['constraints_satisfied']`

## Error Handling

Common errors and their solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| **Negative Values** | Negative constraint value | Use positive values only |
| **Invalid p-value** | p-value ≤ 0 or > 1 | Use value in (0, 1] |
| **Contradictory Constraints** | min > max after resolution | Adjust min/max relationship |
| **Infeasible Combinations** | Not enough data for constraints | Reduce min_bins or constraint values |

## See Also
- [MonotonicBinner](../binning/mob.md) for usage within the binning pipeline
- [Merge Module](./merge.md) for constraint enforcement details