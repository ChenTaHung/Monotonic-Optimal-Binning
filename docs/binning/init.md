# Binning Module Documentation

## Overview
The binning module provides the main user-facing API for monotonic optimal binning. It orchestrates the complete pipeline from data partitioning through PAVA and merging to final bin creation.

## Module Location
`src/MOBPY/binning/`

## Main Class: MonotonicBinner

End-to-end monotonic optimal binning orchestrator that handles the complete binning workflow.

**Import:**
```python
from MOBPY.binning import MonotonicBinner
# or
from MOBPY import MonotonicBinner  # Preferred
```

**Key Responsibilities:**
- Data partitioning and preprocessing
- PAVA (Pool-Adjacent-Violators Algorithm) fitting
- Constraint-based block merging
- WoE (Weight of Evidence) and IV (Information Value) calculation for binary targets
- Bin edge management and transformation

## Quick Reference

For detailed API documentation, see [MonotonicBinner Class Documentation](./mob.md)

### Basic Usage
```python
from MOBPY import MonotonicBinner, BinningConstraints

binner = MonotonicBinner(df, x='age', y='default')
binner.fit()
bins = binner.bins_()
```

### Key Methods
- `fit()` - Run complete binning pipeline
- `bins_()` - Get bin boundaries  
- `summary_()` - Get detailed statistics with WoE/IV
- `transform(x_values, assign)` - Transform new data

## Integration with Core Modules

The binning module integrates with:
1. [`core.constraints`](../core/constraints.md) - BinningConstraints configuration
2. [`core.pava`](../core/pava.md) - PAVA algorithm for monotone fitting
3. [`core.merge`](../core/merge.md) - Block merging strategies
4. [`core.utils`](../core/utils.md) - Helper functions
5. [`plot`](../plot/init.md) - Visualization capabilities

## Design Principles

1. **Single Responsibility**: Focus solely on binning orchestration
2. **Clean API**: Simple, intuitive interface
3. **Separation of Concerns**: Delegates algorithmic details to core modules
4. **Extensibility**: Accommodates future binning strategies
5. **Robustness**: Comprehensive error handling and validation