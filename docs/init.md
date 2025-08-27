# MOBPY Package Documentation

## Overview
MOBPY (Monotonic Optimal Binning for Python) is a fast, deterministic library for creating monotonic bins with respect to a target variable. It implements PAVA (Pool-Adjacent-Violators Algorithm) followed by constrained adjacent merging.

## Package Information
- **Version**: 2.0.0
- **License**: MIT
- **Python Support**: 3.9 - 3.12

## Main Components

### Core Public API

#### `MonotonicBinner`
The main orchestrator class for monotonic optimal binning.
- **Import**: `from MOBPY import MonotonicBinner`
- **Purpose**: End-to-end monotonic binning pipeline

#### `BinningConstraints`
Configuration class for binning constraints.
- **Import**: `from MOBPY import BinningConstraints`
- **Purpose**: Define constraints for the binning process

### Module Structure

```
MOBPY/
├── binning/          # Binning orchestrators
├── core/             # Core algorithms and utilities
└── plot/             # Visualization utilities
```

## Quick Start Example

```python
from MOBPY import MonotonicBinner, BinningConstraints
import pandas as pd

# Create sample data
df = pd.DataFrame({
    'feature': [...],
    'target': [...]
})

# Define constraints
constraints = BinningConstraints(
    max_bins=6,
    min_samples=0.05
)

# Create and fit binner
binner = MonotonicBinner(
    df=df, 
    x='feature', 
    y='target',
    constraints=constraints
)
binner.fit()

# Get results
bins = binner.bins_()
summary = binner.summary_()
```

## Available Functions

### `get_version() -> str`
Returns the current version of MOBPY.

**Returns:**
- `str`: Version string in semantic versioning format (e.g., "2.0.0")

**Example:**
```python
from MOBPY import get_version
print(get_version())  # Output: "2.0.0"
```

## Module Access
Advanced users can access internal modules directly:

```python
from MOBPY import core, binning, plot

# Access specific submodules
from MOBPY.core import pava, merge, utils
from MOBPY.binning import mob
from MOBPY.plot import mob_plot, csd_gcm
```

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
