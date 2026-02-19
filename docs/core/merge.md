# Merge Module Documentation

## Overview
The merge module implements adjacent block merging strategies for satisfying binning constraints after PAVA. It provides flexible scoring strategies and ensures monotonicity is preserved.

## Module Location
`src/MOBPY/core/merge.py`

## Main Classes

### Block
Data structure representing a contiguous block of samples.

```python
@dataclass
class Block:
    left: float          # Left boundary
    right: float         # Right boundary  
    n: int              # Number of samples
    sum: float          # Sum of y values
    sum2: float         # Sum of squared y values
    ymin: float         # Minimum y value
    ymax: float         # Maximum y value
    
    @property
    def mean(self) -> float:
        """Calculate mean of y values."""
        return self.sum / self.n if self.n > 0 else 0.0
    
    @property
    def var(self) -> float:
        """Calculate unbiased sample variance."""
        # Implementation details...
    
    @property
    def std(self) -> float:
        """Calculate standard deviation."""
        return math.sqrt(self.var)
    
    @property
    def cv(self) -> float:
        """Calculate coefficient of variation."""
        # Implementation details...
    
    # NEW in v2.2.0: Class count properties for binary targets
    @property
    def positives(self) -> float:
        """Count of positive samples (y=1). Equals sum for binary y."""
        return self.sum
    
    @property
    def negatives(self) -> float:
        """Count of negative samples (y=0). Equals n - sum for binary y."""
        return self.n - self.sum
```

### Block Methods

| Method | Description |
|--------|-------------|
| `merge_with(other)` | Merge with another block, pooling statistics |
| `as_dict()` | Export block as dictionary (includes `positives` and `negatives`) |

### MergeStrategy
Enum defining available merge selection strategies.

```python
class MergeStrategy(Enum):
    HIGHEST_PVALUE = "highest_pvalue"  # Prefer statistically similar blocks
    SMALLEST_LOSS = "smallest_loss"    # Minimize information loss
    BALANCED_SIZE = "balanced_size"    # Prefer balanced bin sizes
```

### MergeScorer
Calculates merge scores based on selected strategy.

**Scoring Logic:**
- **HIGHEST_PVALUE**: Uses two-sample t-test p-value
- **SMALLEST_LOSS**: Negative of variance increase
- **BALANCED_SIZE**: Prefers merging smaller blocks

**Penalty/Bonus System:**
- 1.5x bonus for merging undersized bins (below `min_samples`)
- 1.3x bonus for merging bins with extreme event rates (0% or 100%)
- 1.4x bonus for merging bins below `min_positives` (binary targets)
- 1.4x bonus for merging bins below `min_negatives` (binary targets) *New in v2.2.0*
- Penalty for creating oversized bins (above `max_samples`)

## Main Function

### merge_adjacent()
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
- `blocks`: Input blocks from PAVA
- `constraints`: Resolved binning constraints
- `is_binary_y`: Whether target is binary
- `strategy`: Strategy for selecting merges
- `history`: List to append merge snapshots
- `max_iterations`: Maximum merge iterations

**Returns:** List of merged blocks satisfying constraints

## Algorithm Phases (Updated in v2.2.0)

The merging algorithm now operates in three phases:

### Phase 1: Statistical Merging
- Merge based on strategy (p-value, loss, or size balance)
- Continue until `max_bins` is reached
- Respects `initial_pvalue` threshold

### Phase 2: Min Samples Enforcement
- Identify bins below `min_samples`
- Merge undersized bins with best-scoring neighbor
- Stop at `min_bins` floor

### Phase 3: Class Count Enforcement (NEW in v2.2.0)
- For binary targets only
- Unified enforcement of both `min_positives` and `min_negatives`
- Intelligently selects merge direction based on which neighbor better satisfies constraints
- Stop at `min_bins` floor
- Issues warnings if constraints cannot be fully satisfied

```
┌─────────────────────────────────────────┐
│ Phase 1: Statistical Merging            │
│ (respect max_bins)                      │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│ Phase 2: Enforce min_samples            │
│ (stop at min_bins)                      │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│ Phase 3: Enforce min_positives AND      │
│          min_negatives (binary only)    │
│ (stop at min_bins)                      │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│ Validation & Warnings                   │
└─────────────────────────────────────────┘
```

## Usage Example

```python
from MOBPY.core import merge_adjacent, BinningConstraints, MergeStrategy

# After PAVA
pava_blocks = [...]  # From PAVA.export_blocks()

# Define and resolve constraints with class count requirements
constraints = BinningConstraints(
    max_bins=5, 
    min_samples=0.05,
    min_positives=10,   # At least 10 positives per bin
    min_negatives=20    # At least 20 negatives per bin
)
constraints.resolve(total_n=1000, total_pos=200)

# Merge blocks
merged = merge_adjacent(
    blocks=pava_blocks,
    constraints=constraints,
    is_binary_y=True,
    strategy=MergeStrategy.HIGHEST_PVALUE
)

# Check class counts on merged blocks
for block in merged:
    print(f"[{block.left}, {block.right}): "
          f"n={block.n}, pos={block.positives}, neg={block.negatives}")
```

## Merge Strategies

### HIGHEST_PVALUE (Default)
- Best for maintaining statistical homogeneity
- Merges blocks with most similar distributions
- Uses Welch's t-test for scoring

### SMALLEST_LOSS
- Minimizes information loss during merging
- Best for preserving predictive power
- Calculates variance increase as loss metric

### BALANCED_SIZE
- Creates more uniform bin sizes
- Useful for operational constraints
- Scores based on combined block size

## Helper Functions

### as_blocks()
Converts list of dictionaries to Block objects.

```python
dict_blocks = [
    {'left': 0, 'right': 5, 'n': 100, 'sum': 45, ...},
    {'left': 5, 'right': 10, 'n': 80, 'sum': 40, ...}
]
block_objects = as_blocks(dict_blocks)
```

### validate_monotonicity()
Validates that blocks maintain monotonicity after merging.

### get_merge_summary()
Generates summary statistics about the merge process.

```python
summary = get_merge_summary(original_blocks, merged_blocks)
print(f"Compression: {summary['compression_ratio']:.2f}x")
print(f"Size balance: {summary['size_balance']:.2f}")
```

## Validation (Updated in v2.2.0)

The `_validate_merge_result()` function now checks:

| Constraint | Action if Violated |
|------------|-------------------|
| `max_bins` | Raises `FittingError` |
| `min_samples` | Warning (if above `min_bins`) |
| `max_samples` | Warning |
| `min_positives` | Warning with WoE stability note |
| `min_negatives` | Warning with WoE stability note |

**Warning Example:**
```
UserWarning: 2 bins have fewer than min_positives=10, but cannot merge 
further without violating min_bins=3. WoE calculations may be unstable. 
Consider relaxing min_positives or min_bins constraint.
```

## Performance Notes
- Time Complexity: O(k²) where k is number of blocks
- Space Complexity: O(k) for block storage
- Typically k << n, making this efficient

## See Also
- [PAVA Algorithm](./pava.md) - Creates initial blocks
- [BinningConstraints](./constraints.md) - Defines merge constraints
- [MonotonicBinner](../binning/mob.md) - Uses merge functionality