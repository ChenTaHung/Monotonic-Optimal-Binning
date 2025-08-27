# Utilities Module Documentation

## Overview
The utils module provides helper functions for data validation, partitioning, statistical calculations, and other common operations used throughout the MOBPY library. These utilities ensure data integrity and support the core binning algorithms.

## Module Location
`src/MOBPY/core/utils.py`

## Main Functions

### Data Validation Functions

#### `ensure_numeric_series()`
Validates that a pandas Series contains numeric data.

```python
def ensure_numeric_series(s: pd.Series, name: str) -> None
```

**Parameters:**
- **s** (`pd.Series`): Series to validate
- **name** (`str`): Name for error messages

**Raises:**
- `DataError`: If series is not numeric or contains infinite values

**Checks:**
- Data type is numeric
- No infinite values (±∞)
- NaN values are allowed

**Example:**
```python
ensure_numeric_series(df['feature'], 'feature')
# Raises DataError if not numeric or has infinite values
```

#### `is_binary_series()`
Checks if a Series contains binary values.

```python
def is_binary_series(
    s: pd.Series,
    strict: bool = True
) -> bool
```

**Parameters:**
- **s** (`pd.Series`): Series to check
- **strict** (`bool`): If True, requires exactly {0, 1} values. If False, allows any two unique values that can be coerced to {0, 1}

**Returns:** `bool` - True if series is binary

**Notes:**
- NaN values are ignored in the check
- Empty series returns False
- Single unique value returns False

**Example:**
```python
# Strict mode (default)
s1 = pd.Series([0, 1, 1, 0, np.nan])
is_binary_series(s1)  # True

s2 = pd.Series([True, False, True])
is_binary_series(s2, strict=True)   # False (not exactly 0/1)
is_binary_series(s2, strict=False)  # True (can be coerced)
```

#### `validate_column_exists()`
Validates that columns exist in a DataFrame.

```python
def validate_column_exists(
    df: pd.DataFrame,
    columns: Union[str, list]
) -> None
```

**Parameters:**
- **df** (`pd.DataFrame`): DataFrame to check
- **columns** (`str` or `list`): Column name(s) to validate

**Raises:**
- `DataError`: If any column is missing

**Example:**
```python
df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
validate_column_exists(df, 'a')        # OK
validate_column_exists(df, ['a', 'b']) # OK
validate_column_exists(df, 'c')        # Raises DataError
```

### Data Partitioning

#### `partition_df()`
Partitions DataFrame into clean, missing, and excluded subsets.

```python
def partition_df(
    df: pd.DataFrame,
    x: str,
    exclude_values: Optional[Iterable] = None,
    validate: bool = True
) -> Parts
```

**Parameters:**
- **df** (`pd.DataFrame`): Input DataFrame
- **x** (`str`): Column name to partition on
- **exclude_values** (`Iterable`): Values to exclude from clean partition (e.g., [-999, -1] for special codes)
- **validate** (`bool`): Whether to validate the input column exists

**Returns:** `Parts` object containing three DataFrames

**Partitioning Logic:**
- **Clean**: Valid numeric values not in exclude_values
- **Missing**: Rows where x is NaN/null
- **Excluded**: Rows where x matches any exclude_values

**Example:**
```python
df = pd.DataFrame({
    'feature': [1, 2, np.nan, -999, 5, 6],
    'target': [0, 1, 1, 0, 1, 0]
})

parts = partition_df(df, 'feature', exclude_values=[-999])
print(f"Clean: {len(parts.clean)}")     # 4
print(f"Missing: {len(parts.missing)}")  # 1
print(f"Excluded: {len(parts.excluded)}") # 1
```

#### `Parts` Dataclass
Container for partitioned data.

```python
@dataclass(frozen=True)
class Parts:
    clean: pd.DataFrame
    missing: pd.DataFrame
    excluded: pd.DataFrame
```

**Methods:**

- `summary() -> Dict[str, int]`: Get partition sizes
- `validate() -> bool`: Check that partitions don't overlap

**Example:**
```python
parts = partition_df(df, 'feature', exclude_values=[-999])

# Get summary
summary = parts.summary()
print(f"Total rows: {summary['total']}")

# Validate no overlap
assert parts.validate()  # True if no overlapping indices
```

### Statistical Calculations

#### `calculate_correlation()`
Calculates correlation between two series.

```python
def calculate_correlation(
    x: pd.Series,
    y: pd.Series,
    method: str = 'pearson'
) -> float
```

**Parameters:**
- **x, y** (`pd.Series`): Series to correlate
- **method** (`str`): Correlation method ('pearson', 'spearman', 'kendall')

**Returns:** `float` - Correlation coefficient

**Example:**
```python
corr = calculate_correlation(df['feature'], df['target'])
print(f"Correlation: {corr:.3f}")
```

#### `woe_iv()`
Calculates Weight of Evidence and Information Value.

```python
def woe_iv(
    goods: np.ndarray,
    bads: np.ndarray,
    smoothing: float = 0.5,
    return_components: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Dict[str, np.ndarray]]
```

**Parameters:**
- **goods** (`np.ndarray`): Count of good outcomes per bin
- **bads** (`np.ndarray`): Count of bad outcomes per bin
- **smoothing** (`float`): Laplace smoothing factor to prevent division by zero
- **return_components** (`bool`): If True, return detailed components

**Returns:**
- If `return_components=False`: Tuple of (woe_array, iv_array)
- If `return_components=True`: Dict with keys: 'woe', 'iv', 'good_rate', 'bad_rate', 'total_iv'

**Formulas:**
```python
# WoE (Weight of Evidence)
WoE = ln(good_rate / bad_rate)

# IV (Information Value)
IV = (good_rate - bad_rate) * WoE

# With smoothing
good_rate = (goods + smoothing) / (total_goods + smoothing * n_bins)
bad_rate = (bads + smoothing) / (total_bads + smoothing * n_bins)
```

**Example:**
```python
# Bin statistics
goods = np.array([80, 60, 40])  # Good counts per bin
bads = np.array([20, 40, 60])   # Bad counts per bin

# Calculate WoE/IV
woe_vals, iv_vals = woe_iv(goods, bads)

# Or get detailed components
result = woe_iv(goods, bads, return_components=True)
print(f"Total IV: {result['total_iv']:.3f}")
print(f"WoE values: {result['woe']}")
```

### Type Checking Functions

#### `is_numeric_dtype()`
Checks if Series has numeric dtype.

```python
def is_numeric_dtype(s: pd.Series) -> bool
```

**Returns:** `bool` - True if numeric dtype

#### `has_special_values()`
Checks for special values in Series.

```python
def has_special_values(
    s: pd.Series,
    special_values: Iterable
) -> bool
```

**Returns:** `bool` - True if any special values found

## WoE/IV Interpretation Guide

### Weight of Evidence (WoE)
- **Positive WoE**: Good rate > Bad rate (lower risk)
- **Negative WoE**: Good rate < Bad rate (higher risk)
- **Zero WoE**: Good rate = Bad rate (neutral)

### Information Value (IV) Ranges
| IV Range | Predictive Power |
|----------|-----------------|
| < 0.02 | Useless |
| 0.02 - 0.1 | Weak |
| 0.1 - 0.3 | Medium |
| 0.3 - 0.5 | Strong |
| > 0.5 | Suspicious (too good) |

## Usage Patterns

### Complete Validation Pipeline
```python
from MOBPY.core.utils import (
    validate_column_exists,
    ensure_numeric_series,
    is_binary_series,
    partition_df
)

# Step 1: Validate columns exist
validate_column_exists(df, ['feature', 'target'])

# Step 2: Check target type
is_binary = is_binary_series(df['target'])

# Step 3: Partition data
parts = partition_df(
    df, 
    x='feature',
    exclude_values=[-999, -1]
)

# Step 4: Validate clean data
ensure_numeric_series(parts.clean['feature'], 'feature')
ensure_numeric_series(parts.clean['target'], 'target')

print(f"Ready for binning: {len(parts.clean)} clean rows")
```

### WoE/IV Calculation Workflow
```python
# After binning, calculate WoE/IV for each bin
bin_stats = []
for bin in bins:
    mask = (df['feature'] >= bin['left']) & (df['feature'] < bin['right'])
    bin_data = df[mask]
    
    goods = (bin_data['target'] == 0).sum()
    bads = (bin_data['target'] == 1).sum()
    bin_stats.append({'good': goods, 'bad': bads})

# Extract arrays
goods = np.array([b['good'] for b in bin_stats])
bads = np.array([b['bad'] for b in bin_stats])

# Calculate WoE/IV
result = woe_iv(goods, bads, return_components=True)

# Add to bin summary
for i, bin in enumerate(bins):
    bin['woe'] = result['woe'][i]
    bin['iv'] = result['iv'][i]

print(f"Total IV: {result['total_iv']:.3f}")
```

## Error Handling

### Common Errors

1. **DataError**: Missing columns, non-numeric data, infinite values
2. **ValueError**: Invalid correlation method, empty data
3. **Warning**: All NaN values, zero variance

### Defensive Programming
```python
# Always validate before processing
try:
    validate_column_exists(df, required_cols)
    ensure_numeric_series(df[x_col], x_col)
    
    if is_binary_series(df[y_col]):
        # Binary target processing
        pass
    else:
        # Continuous target processing
        pass
        
except DataError as e:
    logger.error(f"Data validation failed: {e}")
    raise
```

## Performance Notes

- **Partitioning**: O(n) for single pass through data
- **Validation**: O(n) for checking values
- **WoE/IV**: O(k) where k is number of bins
- **Correlation**: O(n) for computation

## Best Practices

1. **Always Validate First**: Check data before processing
2. **Handle Edge Cases**: Empty data, all NaN, single values
3. **Use Smoothing**: Prevent division by zero in WoE/IV
4. **Check Partitions**: Ensure no data loss during partitioning
5. **Log Issues**: Use logging for debugging data problems

## Configuration

The module respects global configuration:
```python
from MOBPY.config import get_config

config = get_config()
# Uses config.epsilon for numerical comparisons
# Uses config.validate_inputs for validation behavior
```

## Dependencies
- numpy
- pandas
- scipy
- MOBPY.exceptions
- MOBPY.config

## See Also
- [`MonotonicBinner`](../binning/mob.md) - Uses these utilities
- [`BinningConstraints`](./constraints.md) - Works with partitioned data
- [`PAVA`](./pava.md) - Uses validation functions
- [`merge_adjacent`](./merge.md) - Uses statistical functions
