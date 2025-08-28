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
    
    @property
    def mean(self) -> float:
        """Calculate mean of y values."""
        return self.sum / self.n if self.n > 0 else 0.0
    
    @property
    def variance(self) -> float:
        """Calculate variance of y values."""
        # Implementation details...
```

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

**Algorithm Phases:**
1. **Statistical Merging**: Merge based on strategy until max_bins reached
2. **Min Samples Enforcement**: Ensure minimum samples per block
3. **Min Positives Check**: For binary targets, ensure minimum positives

## Usage Example

```python
from MOBPY.core import merge_adjacent, BinningConstraints, MergeStrategy

# After PAVA
pava_blocks = [...]  # From PAVA.export_blocks()

# Define and resolve constraints
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

## Performance Notes
- Time Complexity: O(k²) where k is number of blocks
- Space Complexity: O(k) for block storage
- Typically k << n, making this efficient

## See Also
- [PAVA Algorithm](./pava.md) - Creates initial blocks
- [BinningConstraints](./constraints.md) - Defines merge constraints
- [MonotonicBinner](../binning/mob.md) - Uses merge functionality