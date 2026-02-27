# Binning Module Documentation

## Overview
The binning module provides the main user-facing API for monotonic optimal binning. It orchestrates the complete pipeline from data partitioning through PAVA and merging to final bin creation.

## Module Location
`src/MOBPY/binning/`

## Main Class: MonotonicBinner

End-to-end monotonic optimal binning orchestrator that handles the complete binning workflow for both numeric and categorical features.

**Import:**

```python
from MOBPY.binning import MonotonicBinner
# or
from MOBPY import MonotonicBinner  # Preferred
```

**Key Responsibilities:**

- Data partitioning and preprocessing
- **Numeric path**: PAVA (Pool-Adjacent-Violators Algorithm) + adjacent block merging via Welch's t-test
- **Categorical path**: Chi-square block merging with multiple comparison correction (Holm by default)
- Constraint-based merging (min/max bins, min samples, min positives, min negatives)
- WoE (Weight of Evidence) and IV (Information Value) calculation for binary targets
- Bin assignment, transformation, and diagnostics

## Quick Reference

For detailed API documentation, see [MonotonicBinner Class Documentation](./mob.md)

### Numeric Binning (default)

```python
from MOBPY import MonotonicBinner, BinningConstraints

binner = MonotonicBinner(df, x='age', y='default')
binner.fit()
bins = binner.bins_()
```

### Categorical Binning

```python
from MOBPY import MonotonicBinner, BinningConstraints

binner = MonotonicBinner(
    df, x='merchant_category', y='is_fraud',
    x_type='categorical',
    categorical_alpha=0.05,
)
binner.fit()
ba = binner.bin_assignment()   # category → bin index mapping
```

### Key Methods

- `fit()` - Run complete binning pipeline (numeric or categorical)
- `bins_()` - Get bin boundaries (numeric) or category groups (categorical)
- `summary_()` - Get detailed statistics with WoE/IV
- `transform(x_values, assign)` - Transform new data
- `bin_assignment()` - Map categories to 0-based bin indices (categorical only)
- `get_diagnostics()` - Fitting diagnostics and constraint satisfaction
- `pava_blocks_()` - Raw PAVA blocks before merging (numeric only)
- `pava_groups_()` - Grouped statistics used by PAVA (numeric only)

## Integration with Core Modules

The binning module integrates with:

1. [`core.constraints`](../core/constraints.md) - BinningConstraints configuration
2. [`core.pava`](../core/pava.md) - PAVA algorithm for monotone fitting (numeric)
3. [`core.merge`](../core/merge.md) - Adjacent block merging strategies (numeric)
4. [`core.categorical_merge`](../core/categorical_merge.md) - Chi-square merging (categorical)
5. [`core.utils`](../core/utils.md) - Helper functions
6. [`plot`](../plot/init.md) - Visualization capabilities

## Design Principles

1. **Single Responsibility**: Focus solely on binning orchestration
2. **Clean API**: Simple, intuitive interface
3. **Separation of Concerns**: Delegates algorithmic details to core modules
4. **Extensibility**: Accommodates future binning strategies
5. **Robustness**: Comprehensive error handling and validation