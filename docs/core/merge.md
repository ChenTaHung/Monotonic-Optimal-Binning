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
    pvalue_history: List[float] = field(default_factory=list)
```

**Properties:**
| Property | Type | Description |
|----------|------|-------------|
| **mean** | `float` | Sample mean (sum/n) |
| **var** | `float` | Sample variance using stable computation |
| **std** | `float` | Sample standard deviation |
| **cv** | `float` | Coefficient of variation (std/mean) |

**Methods:**

#### `merge_with(other: Block) -> Block`
Merges with another block, pooling statistics.

**Parameters:**
- **other** (`Block`): Block to merge with

**Returns:** New Block with combined statistics

**Example:**
```python
block1 = Block(left=0, right=5, n=100, sum=45, sum2=25, ymin=0, ymax=1)
block2 = Block(left=5, right=10, n=80, sum=40, sum2=22, ymin=0, ymax=1)

merged = block1.merge_with(block2)
assert merged.left == 0
assert merged.right == 10
assert merged.n == 180
assert merged.sum == 85
```

#### `as_dict() -> Dict`
Exports block as dictionary.

**Returns:** Dictionary with block statistics

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

#### `score_pair(block1: Block, block2: Block) -> float`
Scores a potential merge between two blocks.

**Parameters:**
- **block1** (`Block`): First block
- **block2** (`Block`): Second block

**Returns:** Score (higher means better merge candidate)

**Scoring Logic:**
- **HIGHEST_PVALUE**: Uses two-sample t-test p-value
- **SMALLEST_LOSS**: Negative of variance increase
- **BALANCED_SIZE**: Prefers merging smaller blocks

## Main Functions

### `merge_adjacent()`
Main function for merging adjacent blocks with constraints.

```python
def merge_adjacent(
    blocks: Union[List[Block], List[Dict]],
    constraints: BinningConstraints,
    is_binary_y: bool = False,
    strategy: Union[MergeStrategy, str] = MergeStrategy.HIGHEST_PVALUE,
    history: Optional[List[List[Dict]]] = None,
    max_iterations: Optional[int] = None
) -> List[Block]
```

**Parameters:**
- **blocks** (`Union[List[Block], List[Dict]]`): Input blocks from PAVA
- **constraints** (`BinningConstraints`): Resolved binning constraints
- **is_binary_y** (`bool`): Whether target is binary
- **strategy** (`Union[MergeStrategy, str]`): Strategy for selecting merges
- **history** (`Optional[List]`): List to append merge snapshots to
- **max_iterations** (`Optional[int]`): Maximum merge iterations

**Returns:** List of merged blocks satisfying constraints

**Algorithm Phases:**
1. **Statistical Merging**: Merge based on strategy until max_bins reached
2. **Min Samples Enforcement**: Ensure each block has minimum samples
3. **Min Positives Check**: For binary targets, ensure minimum positives

**Raises:**
- `FittingError`: If merging produces invalid results

**Example:**
```python
from MOBPY.core import merge_adjacent, BinningConstraints, MergeStrategy

# After PAVA
pava_blocks = [...]  # From PAVA.export_blocks()

# Define constraints
constraints = BinningConstraints(max_bins=5, min_samples=0.05)
constraints.resolve(total_n=1000, total_pos=200)

# Merge blocks
merged = merge_adjacent(
    blocks=pava_blocks,
    constraints=constraints,
    is_binary_y=True,
    strategy=MergeStrategy.HIGHEST_PVALUE
)
```

### `as_blocks()`
Converts list of dictionaries or Blocks to Block objects.

```python
def as_blocks(blocks: Union[List[Block], List[Dict]]) -> List[Block]
```

**Parameters:**
- **blocks**: Input blocks (as Block objects or dictionaries)

**Returns:** List of Block objects

**Example:**
```python
# Convert dictionaries to Blocks
dict_blocks = [
    {'left': 0, 'right': 5, 'n': 100, 'sum': 45, ...},
    {'left': 5, 'right': 10, 'n': 80, 'sum': 40, ...}
]
block_objects = as_blocks(dict_blocks)
```

### `validate_monotonicity()`
Validates that blocks maintain monotonicity.

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

**Example:**
```python
from MOBPY.core.merge import validate_monotonicity

# Check monotonicity after merging
assert validate_monotonicity(merged_blocks, sign='+'), "Lost monotonicity!"
```

### `get_merge_summary()`
Generate summary statistics about the merge process.

```python
def get_merge_summary(
    original_blocks: List[Block],
    merged_blocks: List[Block]
) -> Dict[str, Any]
```

**Parameters:**
- **original_blocks**: Blocks before merging
- **merged_blocks**: Blocks after merging

**Returns:** Dictionary with statistics:
```python
{
    'original_count': int,
    'merged_count': int,
    'compression_ratio': float,
    'merges_performed': int,
    'original_size_stats': {
        'min': float,
        'max': float,
        'mean': float,
        'std': float
    },
    'merged_size_stats': {...},
    'original_mean_range': (float, float),
    'merged_mean_range': (float, float),
    'size_balance': float  # min_size / max_size
}
```

**Example:**
```python
from MOBPY.core.merge import get_merge_summary

summary = get_merge_summary(pava_blocks, merged_blocks)
print(f"Compression ratio: {summary['compression_ratio']:.2f}x")
print(f"Size balance: {summary['size_balance']:.2%}")
```

## Merge Strategies

### HIGHEST_PVALUE (Default)
Merges blocks with highest p-value from two-sample t-test.

**Formula:**
```python
t_stat = (mean1 - mean2) / sqrt(var1/n1 + var2/n2)
p_value = 2 * (1 - cdf(abs(t_stat)))
```

**Characteristics:**
- Prioritizes statistically similar blocks
- Preserves distinct distributions
- Good for inference tasks

### SMALLEST_LOSS
Minimizes information loss when merging.

**Formula:**
```python
# Variance increase after merging
loss = merged_var * merged_n - (var1 * n1 + var2 * n2)
```

**Characteristics:**
- Based on variance increase
- Preserves signal quality
- Good for prediction tasks

### BALANCED_SIZE
Balances block sizes.

**Score Formula:**
```python
# Prefer merging smaller blocks
score = -min(block1.n, block2.n)
```

**Characteristics:**
- Creates more uniform bins
- Better for categorical encoding
- Reduces extreme bin sizes

## Constraint Enforcement

### Phase 1: Max Bins
```python
while len(blocks) > max_bins:
    # Find best merge according to strategy
    best_pair = find_best_merge(blocks, strategy)
    # Merge adjacent blocks
    blocks = merge_pair(blocks, best_pair)
```

### Phase 2: Min Samples
```python
for block in blocks:
    if block.n < min_samples:
        # Force merge with best neighbor
        neighbor = find_best_neighbor(block)
        blocks = merge_pair(blocks, (block, neighbor))
```

### Phase 3: Min Positives (Binary Only)
```python
for block in blocks:
    positives = block.sum  # For 0/1 targets
    if positives < min_positives:
        # Merge with neighbor
        neighbor = find_neighbor_with_positives(block)
        blocks = merge_pair(blocks, (block, neighbor))
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

print(f"Merged {len(pava_blocks)} -> {len(merged)} blocks")
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
# Track merge history for debugging
merge_history = []
blocks = merge_adjacent(
    blocks=pava_blocks,
    constraints=constraints,
    history=merge_history
)

# History contains snapshots after each merge
print(f"Performed {len(merge_history)} merge operations")
for i, snapshot in enumerate(merge_history):
    print(f"After merge {i+1}: {len(snapshot)} blocks")
```

### Validation and Summary
```python
from MOBPY.core.merge import validate_monotonicity, get_merge_summary

# Validate monotonicity
assert validate_monotonicity(merged, sign='+'), "Lost monotonicity!"

# Get summary statistics
summary = get_merge_summary(pava_blocks, merged)
print(f"""
Merge Summary:
  Original blocks: {summary['original_count']}
  Final blocks: {summary['merged_count']}
  Compression: {summary['compression_ratio']:.2f}x
  Merges performed: {summary['merges_performed']}
  Size balance: {summary['size_balance']:.2%}
""")
```

## Statistical Tests

### Two-Sample T-Test
Used for HIGHEST_PVALUE strategy to test if two blocks have significantly different means.

```python
def compute_t_statistic(block1: Block, block2: Block) -> float:
    """Compute t-statistic for two blocks."""
    mean_diff = block1.mean - block2.mean
    se = sqrt(block1.var/block1.n + block2.var/block2.n)
    return mean_diff / se if se > 0 else 0.0
```

### Information Loss
Used for SMALLEST_LOSS strategy to quantify the loss of information from merging.

```python
def compute_merge_loss(block1: Block, block2: Block) -> float:
    """Compute information loss from merging."""
    merged = block1.merge_with(block2)
    original_var = block1.var * block1.n + block2.var * block2.n
    merged_var = merged.var * merged.n
    return merged_var - original_var
```

## Performance Characteristics

- **Time Complexity**: O(k²) worst case, O(k log k) typical where k is number of blocks
- **Space Complexity**: O(k) for blocks + O(k²) if tracking history
- **Scalability**: Efficient for up to ~1000 blocks
- **Optimization**: Uses cached statistics for O(1) merge operations

### Performance Tips
1. **Pre-compute Statistics**: Block maintains sum/sum2 for fast variance
2. **Adjacent Only**: Only considers adjacent blocks (maintains order)
3. **Early Stopping**: Stops when constraints satisfied
4. **Score Caching**: Caches merge scores when possible

## Error Handling

### Common Errors

1. **FittingError: Cannot satisfy constraints**
```python
# When min_samples * min_bins > total_samples
constraints = BinningConstraints(min_bins=10, min_samples=0.2)
# 10 bins * 20% each = 200% > 100%
merge_adjacent(blocks, constraints)  # Raises FittingError
```

2. **ValueError: Invalid strategy**
```python
merge_adjacent(blocks, constraints, strategy="invalid")  # Raises ValueError
```

3. **FittingError: Zero blocks produced**
```python
# Empty input
merge_adjacent([], constraints)  # Raises FittingError
```

## Advanced Features

### P-value Annealing
When no valid merges are found, the p-value threshold is reduced:
```python
current_pvalue = constraints.initial_pvalue
while not found_merge and current_pvalue > 0.01:
    current_pvalue *= 0.9  # Anneal by 10%
    # Retry with lower threshold
```

### Constraint Relaxation
When strict constraints cannot be satisfied:
```python
if cannot_satisfy_min_samples:
    # System may relax with warning
    relaxed_min = int(0.8 * original_min)
    logger.warning(
        "Relaxing min_samples from %d to %d",
        original_min, relaxed_min
    )
```

### Monotonicity Preservation
Merging maintains monotonicity through careful selection:
```python
# Only merge if monotonicity preserved
merged = block1.merge_with(block2)
if sign == '+' and merged.mean >= prev_block.mean:
    # Safe to merge
    perform_merge()
```

## Integration Notes

- Takes blocks from PAVA output
- Respects resolved constraints from BinningConstraints
- Used by MonotonicBinner after PAVA step
- Output feeds into bin creation

## Best Practices

1. **Resolve Constraints First**: Always resolve before merging
2. **Choose Right Strategy**: 
   - HIGHEST_PVALUE for statistical inference
   - SMALLEST_LOSS for predictive accuracy
   - BALANCED_SIZE for stable encoding
3. **Validate Results**: Check monotonicity and constraints
4. **Track History**: Use for debugging complex cases
5. **Monitor Warnings**: Pay attention to constraint relaxation

## Thread Safety

The merge functions are thread-safe for read operations. For concurrent modifications, use separate block lists or appropriate locking.

## Dependencies
- numpy
- scipy.stats (for t-distribution)
- MOBPY.core.constraints
- MOBPY.exceptions
- MOBPY.config
- MOBPY.logging_utils

## See Also
- [`PAVA`](./pava.md) - Creates initial monotonic blocks
- [`BinningConstraints`](./constraints.md) - Constraint specifications
- [`MonotonicBinner`](../binning/mob.md) - Main orchestrator using merge
- [`utils`](./utils.md) - Helper functions
- [`Block` class](./merge.md#block) - Data structure details