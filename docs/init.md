# MOBPY Package Documentation

## Overview
MOBPY (Monotonic Optimal Binning for Python) is a fast, deterministic library for creating monotonic bins with respect to a target variable. It implements PAVA (Pool-Adjacent-Violators Algorithm) followed by constrained adjacent merging.

## Package Information
- **Version**: 2.0.0
- **License**: MIT
- **Python Support**: 3.9 - 3.12

## Main Components

### Core Public API

#### MonotonicBinner
The main orchestrator class for monotonic optimal binning.
- **Import**: `from MOBPY import MonotonicBinner`
- **Purpose**: End-to-end monotonic binning pipeline
- **Documentation**: [Full MonotonicBinner API](./binning/mob.md)

#### BinningConstraints
Configuration class for binning constraints.
- **Import**: `from MOBPY import BinningConstraints`
- **Purpose**: Define constraints for the binning process
- **Documentation**: [Full BinningConstraints API](./core/constraints.md)

### Quick Start

```python
from MOBPY import MonotonicBinner, BinningConstraints
import pandas as pd

# Load and prepare data
df = pd.read_csv('your_data.csv')

# Configure constraints
constraints = BinningConstraints(
    max_bins=6,
    min_samples=0.05  # 5% minimum
)

# Fit and transform
binner = MonotonicBinner(df, x='feature', y='target', constraints=constraints)
binner.fit()

# Get results
bins = binner.bins_()
summary = binner.summary_()
```

For detailed examples and parameters, see:
- [Complete workflow example](./MOBPY-Overview.md#complete-workflow-example)
- [MonotonicBinner documentation](./binning/mob.md)
- [Plotting examples](./plot/init.md)

## Available Functions

### get_version()
Returns the current version of MOBPY.

```python
from MOBPY import get_version
print(get_version())  # Output: "2.0.0"
```

## Module Access

Advanced users can access internal modules directly. See individual module documentation:
- [Core algorithms](./core/) - PAVA, merging, utilities
- [Binning module](./binning/) - MonotonicBinner implementation
- [Plot module](./plot/) - Visualization tools

## Key Features
- **Deterministic & Fast**: Stack-based PAVA with O(k) adjacent merges
- **Robust Constraints**: Min/max samples, min positives, min/max bins
- **Safe Edges**: First bin starts at -∞, last bin ends at +∞
- **Well-tested**: Comprehensive unit and property-based tests

## Dependencies
- numpy
- pandas  
- scipy
- matplotlib (for plotting)
- pytest, hypothesis (for testing, optional)