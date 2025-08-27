# Block Merging Module Documentation

## Overview
The merge module provides algorithms for merging adjacent blocks based on statistical tests and constraints. It implements multiple merge strategies and handles constraint satisfaction including minimum samples, maximum bins, and minimum positives for binary targets.

## Module Location
`src/MOBPY/core/merge.py`

## Main Classes

### `Block`
Data structure representing a contiguous block with sufficient statistics.

```python
@dataclass
class Block:
    left: float         # Left boundary (inclusive)
    right: float        # Right boundary (exclusive)
    n: int              # Number of samples
    sum: float          # Sum of y values
    sum2: float         # Sum of y² values
    ymin: float         # Minimum y value
    ymax: float         # Maximum y value
    merge_history: List[Tuple[float, float]] = field(default_factory=list)
```

**Properties:**
| Property | Type | Description |
|----------|------|-------------|
| **mean** | `float` | Sample mean (sum/n) |
| **var** | `float` | Sample variance using stable computation |
| **std** | `float` | Sample standard deviation |
| **cv** | `float` | Coefficient of variation (std/mean) |

**Methods:**
- `merge_with(other: Block) -> Block`: Merge with another block, pooling statistics
- `as_dict() -> Dict`: Export block as dictionary

### `MergeStrategy` (Enum)
Enumeration defining merge selection strategies.

```python
class MergeStrategy(Enum):
    HIGHEST_PVALUE = "highest_pvalue"      # Merge most similar (default)
    SMALLEST_LOSS = "smallest_loss"        # Minimize information loss
    BALANCED_SIZE = "balanced_size"        # Balance block sizes
```

### `MergeScorer`
Scoring system for evaluating potential merges.

```python
class MergeScorer:
    def __init__(
        self,
        constraints: BinningConstraints,
        is_binary_y: bool,
        strategy: MergeStrategy
    )
```

**Methods:**
- `score_pair(block1: Block, block2: Block) -> float`: Score a potential merge

## Main Functions

### `merge_adjacent()`
Main function for merging adjacent blocks with constraints.

```python
def merge_adjacent(
    blocks: Union[List[Block], List[Dict]],
    constraints: BinningConstraints,
    *,
    is_binary_y: bool = False,
    strategy: Union[MergeStrategy, str] = MergeStrategy.HIGHEST_PVALUE,
    return_history: bool = False
) -> Union[List[Block], Tuple[List[Block], List[List[Dict]]]]
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **blocks** | `List[Block/Dict]` | required | Initial blocks from PAVA |
| **constraints** | `BinningConstraints` | required | Resolved constraints |
| **is_binary_y** | `bool` | `False` | Whether target is binary |
| **strategy** | `MergeStrategy/str` | `HIGHEST_PVALUE` | Merge selection strategy |
| **return_history** | `bool` | `False` | Whether to return merge history |

**Returns:**
- If `return_history=False`: List of merged blocks
- If `return_history=True`: Tuple of (blocks, history)

**Algorithm:**
1. **Phase 1**: Merge to satisfy max_bins constraint
2. **Phase 2**: Enforce minimum samples per block
3. **Phase 3**: Enforce minimum positives (binary targets)
4. **Validation**: Check final constraints

### `blocks_from_dicts()`
Convert list of dictionaries to Block objects.

```python
def blocks_from_dicts(rows: List[Dict[str, Any]]) -> List[Block]
```

**Parameters:**
- **rows**: List of dictionaries with block statistics

**Returns:** List of Block objects

**Field Mapping:**
- Handles both 'ymin'/'ymax' and 'min'/'max' field names
- Initializes merge_history as empty list

### `as_blocks()`
Flexible converter for various input formats.

```python
def as_blocks(
    rows: Union[List[Dict[str, Any]], List[Block]]
) -> List[Block]
```

**Parameters:**
- **rows**: List of dicts or Block objects

**Returns:** List of Block objects

## Utility Functions

### `validate_monotonicity()`
Check if blocks satisfy monotonicity constraint.

```python
def validate_monotonicity(
    blocks: List[Block],
    sign: str,
    tolerance: float = 1e-10
) -> bool
```

**Parameters:**
- **blocks**: Blocks to validate
- **sign**: '+' for non-decreasing, '-' for non-increasing
- **tolerance**: Numerical tolerance for comparisons

**Returns:** True if monotonicity is satisfied

### `get_merge_summary()`
Generate summary statistics about the merge process.

```python
def get_merge_summary(
    original_blocks: List[Block],
    merged_blocks: List[Block]
) -> Dict[str, Any]
```

**Returns:** Dictionary with statistics:
- Original and merged counts
- Compression ratio
- Size and mean statistics
- Balance metrics

## Merge Strategies

### HIGHEST_PVALUE (Default)
Merges blocks with highest p-value from two-sample t-test:
- Prioritizes statistically similar blocks
- Preserves distinct distributions
- Good for inference tasks

### SMALLEST_LOSS
Minimizes information loss when merging:
- Based on variance increase
- Preserves signal quality
- Good for prediction tasks

### BALANCED_SIZE
Balances block sizes:
- Creates more uniform bins
- Better for categorical encoding
- Reduces extreme bin sizes

## Constraint Enforcement

### Phase 1: Max Bins
```python
while len(blocks) > max_bins:
    # Find best merge according to strategy
    # Merge adjacent blocks
    # Update blocks list
```

### Phase 2: Min Samples
```python
for block in blocks:
    if block.n < min_samples:
        # Force merge with best neighbor
        # Continue until constraint met
```

### Phase 3: Min Positives (Binary Only)
```python
for block in blocks:
    positives = block.sum  # For 0/1 targets
    if positives < min_positives:
        # Merge with neighbor
```

## Usage Examples

### Basic Merging
```python
from MOBPY.core import merge_adjacent, BinningConstraints

# After PAVA
pava_blocks = [...]  # From PAVA.export_blocks()

# Define constraints
constraints = BinningConstraints(max_bins=5, min_samples=0.05)
constraints.resolve(total_n=1000, total_pos=200)

# Merge blocks
merged = merge_adjacent(
    blocks=pava_blocks,
    constraints=constraints,
    is_binary_y=True
)
```

### Different Strategies
```python
# Use smallest loss strategy
merged = merge_adjacent(
    blocks=pava_blocks,
    constraints=constraints,
    strategy=MergeStrategy.SMALLEST_LOSS
)

# Use balanced size strategy
merged = merge_adjacent(
    blocks=pava_blocks,
    constraints=constraints,
    strategy="balanced_size"  # String also works
)
```

### With History Tracking
```python
# Get merge history for debugging
blocks, history = merge_adjacent(
    blocks=pava_blocks,
    constraints=constraints,
    return_history=True
)

# History is list of snapshots after each merge
print(f"Performed {len(history)} merges")
for i, snapshot in enumerate(history):
    print(f"After merge {i+1}: {len(snapshot)} blocks")
```

### Validation
```python
# Validate monotonicity after merging
from MOBPY.core.merge import validate_monotonicity

assert validate_monotonicity(merged, sign='+'), "Lost monotonicity!"

# Get summary statistics
summary = get_merge_summary(pava_blocks, merged)
print(f"Compression ratio: {summary['compression_ratio']:.2f}x")
print(f"Size balance: {summary['size_balance']:.2%}")
```

## Statistical Tests

### Two-Sample T-Test
Used for HIGHEST_PVALUE strategy:
```python
t_stat = (mean1 - mean2) / sqrt(var1/n1 + var2/n2)
p_value = 2 * (1 - cdf(abs(t_stat)))
```

### Information Loss
Used for SMALLEST_LOSS strategy:
```python
loss = merged_var * merged_n - (var1 * n1 + var2 * n2)
```

## Performance Characteristics

- **Time Complexity**: O(k²) worst case, O(k log k) typical
- **Space Complexity**: O(k) for blocks + O(k²) if tracking history
- **Scalability**: Efficient for up to ~1000 blocks

## Error Handling

Common errors:
1. **FittingError**: Constraints cannot be satisfied
2. **ValueError**: Invalid strategy or parameters
3. **TypeError**: Wrong input types

## Integration Notes

- Takes blocks from PAVA output
- Respects resolved constraints from BinningConstraints
- Used by MonotonicBinner after PAVA step
- Output feeds into bin creation

## Best Practices

1. **Resolve Constraints First**: Always resolve before merging
2. **Choose Right Strategy**: 
   - HIGHEST_PVALUE for inference
   - SMALLEST_LOSS for prediction
   - BALANCED_SIZE for encoding
3. **Validate Results**: Check monotonicity and constraints
4. **Track History**: Use for debugging complex cases

## Dependencies
- numpy
- scipy.stats
- MOBPY.core.constraints
- MOBPY.exceptions
- MOBPY.config

## See Also
- [`PAVA`](./pava.md) - Creates initial monotonic blocks
- [`BinningConstraints`](./constraints.md) - Constraint specifications
- [`MonotonicBinner`](../binning/mob.md) - Main orchestrator
- [`utils`](./utils.md) - Helper functions
