# Core Utilities Module Documentation

## Overview
The utils module provides essential helper functions for data validation, partitioning, statistical calculations, and WoE/IV computations. These utilities are used throughout the MOBPY package to ensure data quality and provide common functionality.

## Module Location
`src/MOBPY/core/utils.py`

## Data Validation Functions

### `ensure_numeric_series(s: pd.Series, name: str) -> pd.Series`
Validates that a pandas Series contains numeric data and converts if possible.

**Parameters:**
- **s** (`pd.Series`): Series to validate
- **name** (`str`): Name of the series for error messages

**Returns:**
- `pd.Series`: Numeric series

**Raises:**
- `DataError`: If series cannot be converted to numeric

**Example:**
```python
from MOBPY.core.utils import ensure_numeric_series

# Validates and converts
numeric_series = ensure_numeric_series(df['age'], 'age')

# Raises DataError if non-numeric
text_series = pd.Series(['a', 'b', 'c'])
ensure_numeric_series(text_series, 'text')  # Raises DataError
```

### `is_binary_series(s: pd.Series, strict: bool = False) -> bool`
Checks if a series contains only binary values (0 and 1).

**Parameters:**
- **s** (`pd.Series`): Series to check
- **strict** (`bool`): If True, requires exactly {0, 1}. If False, allows {0} or {1} only

**Returns:**
- `bool`: True if series is binary

**Example:**
```python
from MOBPY.core.utils import is_binary_series

# Binary series
binary = pd.Series([0, 1, 1, 0, 1])
assert is_binary_series(binary)  # True

# Non-strict allows single value
all_zeros = pd.Series([0, 0, 0])
assert is_binary_series(all_zeros, strict=False)  # True
assert not is_binary_series(all_zeros, strict=True)  # False

# Non-binary
continuous = pd.Series([0.1, 0.5, 0.9])
assert not is_binary_series(continuous)  # False
```

### `validate_column_exists(df: pd.DataFrame, columns: Union[str, List[str]]) -> None`
Validates that specified columns exist in a DataFrame.

**Parameters:**
- **df** (`pd.DataFrame`): DataFrame to check
- **columns** (`Union[str, List[str]]`): Column name(s) to validate

**Raises:**
- `DataError`: If any column is missing

**Example:**
```python
from MOBPY.core.utils import validate_column_exists

# Single column
validate_column_exists(df, 'age')

# Multiple columns
validate_column_exists(df, ['age', 'income', 'default'])

# Raises DataError if missing
validate_column_exists(df, 'nonexistent')  # Raises DataError
```

## Data Partitioning

### `partition_df(df: pd.DataFrame, x: str, exclude_values: Optional[Iterable] = None) -> Parts`
Partitions a DataFrame based on missing values and excluded values in the x column.

**Parameters:**
- **df** (`pd.DataFrame`): Input DataFrame
- **x** (`str`): Column name to partition by
- **exclude_values** (`Optional[Iterable]`): Values to exclude (e.g., special codes like -999)

**Returns:**
- `Parts`: Object containing clean, missing, and excluded DataFrames

**Example:**
```python
from MOBPY.core.utils import partition_df

# Basic partitioning
parts = partition_df(df, x='age')
print(f"Clean rows: {len(parts.clean)}")
print(f"Missing rows: {len(parts.missing)}")

# With excluded values (special codes)
parts = partition_df(df, x='age', exclude_values=[-999, -1])
print(f"Excluded rows: {len(parts.excluded)}")

# Access partitions
clean_df = parts.clean
missing_df = parts.missing
excluded_df = parts.excluded

# Get summary
summary = parts.summary()
print(summary)
```

### `Parts` Class
Container for partitioned DataFrames with validation and summary methods.

**Attributes:**
- **clean** (`pd.DataFrame`): Rows with valid x values
- **missing** (`pd.DataFrame`): Rows with missing x values
- **excluded** (`pd.DataFrame`): Rows with excluded x values

**Methods:**

#### `summary() -> Dict[str, Any]`
Returns a summary dictionary with partition statistics.

**Returns:**
```python
{
    'total_rows': int,
    'clean_rows': int,
    'missing_rows': int,
    'excluded_rows': int,
    'clean_pct': float,
    'missing_pct': float,
    'excluded_pct': float
}
```

#### `validate() -> None`
Validates that partitions are disjoint and complete.

**Raises:**
- `DataError`: If partitions overlap or don't sum to total

**Example:**
```python
parts = partition_df(df, x='age', exclude_values=[-999])

# Get summary statistics
summary = parts.summary()
print(f"Clean: {summary['clean_pct']:.1%}")
print(f"Missing: {summary['missing_pct']:.1%}")
print(f"Excluded: {summary['excluded_pct']:.1%}")

# Validate partitions
parts.validate()  # Raises if inconsistent
```

## Statistical Functions

### `calculate_correlation(x: pd.Series, y: pd.Series, method: str = 'pearson') -> float`
Calculates correlation between two series.

**Parameters:**
- **x** (`pd.Series`): First variable
- **y** (`pd.Series`): Second variable  
- **method** (`str`): Correlation method ('pearson', 'spearman', 'kendall')

**Returns:**
- `float`: Correlation coefficient

**Raises:**
- `ValueError`: If method is invalid or series have different lengths

**Example:**
```python
from MOBPY.core.utils import calculate_correlation

# Pearson correlation (default)
corr = calculate_correlation(df['age'], df['income'])

# Spearman rank correlation
corr = calculate_correlation(df['age'], df['income'], method='spearman')

# Kendall tau correlation
corr = calculate_correlation(df['age'], df['income'], method='kendall')
```

### `woe_iv(goods: Union[int, np.ndarray], bads: Union[int, np.ndarray], smoothing: float = 0.5) -> Tuple[np.ndarray, np.ndarray]`
Calculates Weight of Evidence (WoE) and Information Value (IV) for binary classification.

**Parameters:**
- **goods** (`Union[int, np.ndarray]`): Count of good outcomes (y=0)
- **bads** (`Union[int, np.ndarray]`): Count of bad outcomes (y=1)
- **smoothing** (`float`): Smoothing factor to avoid division by zero

**Returns:**
- `Tuple[np.ndarray, np.ndarray]`: (WoE values, IV contributions)

**Formulas:**
```
WoE = ln((goods_i / total_goods) / (bads_i / total_bads))
IV_i = (goods_i / total_goods - bads_i / total_bads) * WoE_i
```

**Example:**
```python
from MOBPY.core.utils import woe_iv

# Single bin
goods, bads = 100, 20
woe, iv = woe_iv(goods, bads)

# Multiple bins
goods = np.array([100, 150, 80, 120])
bads = np.array([20, 40, 35, 25])
woe_values, iv_values = woe_iv(goods, bads)

print(f"WoE: {woe_values}")
print(f"IV contributions: {iv_values}")
print(f"Total IV: {iv_values.sum():.4f}")
```

### `compute_gini(y_true: np.ndarray, y_score: np.ndarray) -> float`
Calculates Gini coefficient for model performance.

**Parameters:**
- **y_true** (`np.ndarray`): True binary labels
- **y_score** (`np.ndarray`): Predicted scores or probabilities

**Returns:**
- `float`: Gini coefficient (2 * AUC - 1)

**Example:**
```python
from MOBPY.core.utils import compute_gini

y_true = np.array([0, 0, 1, 1, 0, 1])
y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])

gini = compute_gini(y_true, y_scores)
print(f"Gini coefficient: {gini:.4f}")
```

## Helper Functions

### `safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float`
Safely divides two numbers, returning a default value if denominator is zero.

**Parameters:**
- **numerator** (`float`): Numerator
- **denominator** (`float`): Denominator
- **default** (`float`): Value to return if denominator is zero

**Returns:**
- `float`: Result of division or default value

**Example:**
```python
from MOBPY.core.utils import safe_divide

# Normal division
result = safe_divide(10, 2)  # 5.0

# Division by zero
result = safe_divide(10, 0)  # 0.0 (default)
result = safe_divide(10, 0, default=-1)  # -1
```

### `format_number(value: float, precision: int = 4) -> str`
Formats a number for display with appropriate precision.

**Parameters:**
- **value** (`float`): Number to format
- **precision** (`int`): Number of decimal places

**Returns:**
- `str`: Formatted string

**Example:**
```python
from MOBPY.core.utils import format_number

print(format_number(0.123456))  # "0.1235"
print(format_number(1234567))   # "1.235e+06"
print(format_number(np.inf))    # "inf"
```

### `check_monotonicity(values: np.ndarray, sign: str, epsilon: float = 1e-10) -> bool`
Checks if an array is monotonic in the specified direction.

**Parameters:**
- **values** (`np.ndarray`): Array to check
- **sign** (`str`): '+' for non-decreasing, '-' for non-increasing
- **epsilon** (`float`): Tolerance for numerical comparisons

**Returns:**
- `bool`: True if monotonic in specified direction

**Example:**
```python
from MOBPY.core.utils import check_monotonicity

# Non-decreasing
values = np.array([1, 2, 2, 3, 5])
assert check_monotonicity(values, '+')

# Non-increasing
values = np.array([5, 3, 3, 2, 1])
assert check_monotonicity(values, '-')

# Not monotonic
values = np.array([1, 3, 2, 4])
assert not check_monotonicity(values, '+')
```

## Integration Examples

### Complete Data Preparation Pipeline
```python
from MOBPY.core.utils import (
    validate_column_exists,
    partition_df,
    ensure_numeric_series,
    is_binary_series
)

def prepare_data(df, x_col, y_col, exclude_values=None):
    """Complete data preparation pipeline."""
    
    # 1. Validate columns exist
    validate_column_exists(df, [x_col, y_col])
    
    # 2. Partition by missing/excluded
    parts = partition_df(df, x=x_col, exclude_values=exclude_values)
    
    # 3. Ensure numeric data on clean partition
    clean_df = parts.clean.copy()
    clean_df[x_col] = ensure_numeric_series(clean_df[x_col], x_col)
    clean_df[y_col] = ensure_numeric_series(clean_df[y_col], y_col)
    
    # 4. Check if target is binary
    is_binary = is_binary_series(clean_df[y_col])
    
    # 5. Print summary
    summary = parts.summary()
    print(f"Data summary:")
    print(f"  Total rows: {summary['total_rows']}")
    print(f"  Clean: {summary['clean_rows']} ({summary['clean_pct']:.1%})")
    print(f"  Missing: {summary['missing_rows']} ({summary['missing_pct']:.1%})")
    print(f"  Excluded: {summary['excluded_rows']} ({summary['excluded_pct']:.1%})")
    print(f"  Target is binary: {is_binary}")
    
    return clean_df, is_binary, parts
```

### WoE/IV Calculation for Binning
```python
from MOBPY.core.utils import woe_iv
import pandas as pd

def calculate_bin_woe_iv(bins_df, y_binary):
    """Calculate WoE and IV for bins."""
    
    results = []
    for _, bin_row in bins_df.iterrows():
        # Count goods (y=0) and bads (y=1)
        mask = (df['x'] >= bin_row['left']) & (df['x'] < bin_row['right'])
        bin_y = y_binary[mask]
        
        goods = (bin_y == 0).sum()
        bads = (bin_y == 1).sum()
        
        # Calculate WoE and IV
        woe, iv = woe_iv(goods, bads, smoothing=0.5)
        
        results.append({
            'bin': f"[{bin_row['left']}, {bin_row['right']})",
            'goods': goods,
            'bads': bads,
            'woe': woe[0],
            'iv': iv[0]
        })
    
    results_df = pd.DataFrame(results)
    total_iv = results_df['iv'].sum()
    
    print(f"Total IV: {total_iv:.4f}")
    return results_df
```

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

## Performance Notes

- **Vectorized Operations**: Uses numpy/pandas for efficiency
- **Memory Efficient**: Avoids unnecessary copies
- **Lazy Validation**: Only validates when necessary
- **Caching**: Some functions cache results when appropriate

## Error Handling

Common error scenarios:

1. **Non-numeric Data**
```python
# Handled by ensure_numeric_series
try:
    ensure_numeric_series(text_series, 'feature')
except DataError as e:
    print(f"Cannot convert to numeric: {e}")
```

2. **Missing Columns**
```python
# Handled by validate_column_exists
try:
    validate_column_exists(df, 'nonexistent')
except DataError as e:
    print(f"Column not found: {e}")
```

3. **Invalid Partitions**
```python
# Handled by Parts.validate()
try:
    parts.validate()
except DataError as e:
    print(f"Partition error: {e}")
```

## Thread Safety

All functions are thread-safe for read operations. For write operations on shared data, use appropriate locking.

## Dependencies
- numpy
- pandas  
- scipy (for correlation calculations)
- MOBPY.exceptions
- MOBPY.config

## See Also
- [`MonotonicBinner`](../binning/mob.md) - Uses these utilities extensively
- [`BinningConstraints`](./constraints.md) - Works with partitioned data
- [`PAVA`](./pava.md) - Uses validation functions
- [`merge_adjacent`](./merge.md) - Uses statistical functions