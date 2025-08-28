# MOBPY Project Documentation

## Overview
MOBPY (Monotonic Optimal Binning for Python) is a comprehensive library for creating monotonic bins using the Pool-Adjacent-Violators Algorithm (PAVA) followed by constrained merging.

## Project Structure

```
MOBPY/
├── src/
│   └── MOBPY/
│       ├── __init__.py                 # Main package initialization
│       ├── config.py                   # Global configuration system
│       ├── exceptions.py               # Custom exception classes
│       ├── logging_utils.py            # Logging and progress utilities
│       │
│       ├── binning/                    # Binning orchestration layer
│       │   ├── __init__.py
│       │   └── mob.py                  # MonotonicBinner main class
│       │
│       ├── core/                       # Core algorithms and utilities
│       │   ├── __init__.py
│       │   ├── constraints.py          # BinningConstraints class
│       │   ├── pava.py                 # PAVA algorithm implementation
│       │   ├── merge.py                # Block merging algorithms
│       │   └── utils.py                # Helper functions and utilities
│       │
│       └── plot/                       # Visualization tools
│           ├── __init__.py
│           ├── csd_gcm.py              # PAVA visualization functions
│           └── mob_plot.py             # Binning result visualizations
```

## Module Dependencies Graph

```
MonotonicBinner (binning.mob)
    ├── BinningConstraints (core.constraints)
    ├── PAVA (core.pava)
    │   └── calculate_correlation (core.utils)
    ├── merge_adjacent (core.merge)
    │   ├── Block
    │   ├── MergeScorer
    │   └── MergeStrategy
    ├── partition_df (core.utils)
    ├── woe_iv (core.utils)
    └── Plotting Functions (plot.*)
```

## Complete Workflow Example

```python
import pandas as pd
from MOBPY import MonotonicBinner, BinningConstraints
from MOBPY.plot import plot_bin_statistics
from MOBPY.config import set_config
from MOBPY.logging_utils import set_verbosity

# 1. Configure
set_config(epsilon=1e-10, enable_progress_bar=True)
set_verbosity('INFO')

# 2. Load data
df = pd.read_csv('data/german_data_credit_cat.csv')

# 3. Set constraints
constraints = BinningConstraints(
    max_bins=6,
    min_samples=0.05,    # 5% of data
    min_positives=0.01   # 1% of positives
)

# 4. Fit binner
binner = MonotonicBinner(
    df, 
    x='Age', 
    y='default',
    constraints=constraints,
    exclude_values=[-999]  # Special codes
)
binner.fit()

# 5. Get results
bins = binner.bins_()       # Bin boundaries
summary = binner.summary_()  # With WoE/IV

# 6. Visualize
fig = plot_bin_statistics(binner)
plt.show()

# 7. Transform new data
new_bins = binner.transform(new_df['Age'], assign='interval')
woe_scores = binner.transform(new_df['Age'], assign='woe')
```

## Module Documentation Links

### Core Components
- [Configuration Module](./config.md) - Global settings management
- [Exceptions Module](./exceptions.md) - Error handling classes  
- [Logging Module](./logging_utils.md) - Progress tracking

### Binning Module
- [MonotonicBinner Class](./binning/mob.md) - Main binning orchestrator
- [Binning Module Overview](./binning/init.md) - Module structure

### Core Algorithms
- [BinningConstraints](./core/constraints.md) - Constraint configuration
- [PAVA Algorithm](./core/pava.md) - Monotone fitting
- [Merge Module](./core/merge.md) - Block merging strategies
- [Utilities](./core/utils.md) - Helper functions

### Visualization
- [Plot Module Overview](./plot/init.md) - All visualization functions
- [PAVA Visualizations](./plot/csd_gcm.md) - Algorithm process plots
- [Result Visualizations](./plot/mob_plot/) - Binning result plots