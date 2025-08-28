# `MOBPY` Project Orchestration Documentation

## Overview
MOBPY (Monotonic Optimal Binning for Python) is a comprehensive library for creating monotonic bins using the Pool-Adjacent-Violators Algorithm (PAVA) followed by constrained merging. This document provides a complete map of all modules, classes, and functions in the project.

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
        ├── plot_bin_statistics
        ├── plot_woe_bars
        └── plot_pava_comparison
```

---

## Core Package (`src/MOBPY/__init__.py`)

### Main Exports
| Export | Type | Description |
|--------|------|-------------|
| `MonotonicBinner` | Class | Main binning orchestrator |
| `BinningConstraints` | Class | Constraint configuration |
| `__version__` | String | Package version (2.0.0) |
| `get_version()` | Function | Returns version string |

### Usage
```python
from MOBPY import MonotonicBinner, BinningConstraints

# Quick start
binner = MonotonicBinner(df, x='age', y='default')
binner.fit()
```

---

## Configuration Module (`config.py`)

### Classes
| Class | Description |
|-------|-------------|
| `MOBPYConfig` | Global configuration settings dataclass |

### Functions
| Function | Description |
|----------|-------------|
| `get_config()` | Get global configuration instance |
| `set_config(**kwargs)` | Update global configuration |
| `reset_config()` | Reset to default configuration |

### Key Settings
- `epsilon`: Numerical tolerance (default: 1e-12)
- `max_iterations`: Maximum iterations (default: 1000)
- `enable_progress_bar`: Show progress (default: False)
- `n_jobs`: Parallel jobs (default: 1)
- `validate_inputs`: Input validation (default: True)

### Usage
```python
from MOBPY.config import get_config, set_config

# Adjust numerical tolerance
set_config(epsilon=1e-10, n_jobs=-1)
```

---

## Exception Module (`exceptions.py`)

### Exception Classes
| Class | Inherits From | Used For |
|-------|--------------|----------|
| `MOBPYError` | `Exception` | Base exception class |
| `DataError` | `MOBPYError` | Data validation issues |
| `FittingError` | `MOBPYError` | Fitting/algorithm failures |
| `ConstraintError` | `MOBPYError` | Constraint violations |
| `NotFittedError` | `MOBPYError` | Accessing unfitted model |
| `BinningWarning` | `UserWarning` | Non-critical warnings |

### Usage
```python
from MOBPY.exceptions import DataError

if df.empty:
    raise DataError("Input DataFrame cannot be empty")
```

---

## Logging Module (`logging_utils.py`)

### Classes
| Class | Description |
|-------|-------------|
| `BinningProgressLogger` | Progress tracking for long operations |

### Functions
| Function | Description |
|----------|-------------|
| `get_logger(name, level=None)` | Get configured logger instance |
| `set_verbosity(level)` | Set global logging verbosity |

### Usage
```python
from MOBPY.logging_utils import get_logger, set_verbosity

# Set logging level
set_verbosity('DEBUG')  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Get module logger
logger = get_logger(__name__)
logger.info("Starting binning process")

# Progress tracking
with BinningProgressLogger("fitting") as progress:
    progress.update("Running PAVA")
    progress.update("Merging blocks")
```

---

## Binning Module (`binning/`)

### MonotonicBinner Class (`mob.py`)

#### Constructor Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | required | Input data |
| `x` | str | required | Feature column |
| `y` | str | required | Target column |
| `metric` | str | "mean" | Aggregation metric |
| `sign` | str | "auto" | Monotonicity direction (+/-/auto) |
| `strict` | bool | True | Enforce strict monotonicity |
| `constraints` | BinningConstraints | None | Binning constraints |
| `exclude_values` | Iterable | None | Values to exclude |
| `merge_strategy` | MergeStrategy | HIGHEST_PVALUE | Merge selection strategy |

#### Main Methods
| Method | Returns | Description |
|--------|---------|-------------|
| `fit()` | self | Run complete binning pipeline |
| `bins_()` | DataFrame | Get bin boundaries and statistics |
| `summary_()` | DataFrame | Get detailed summary with WoE/IV |
| `transform(x_values, assign)` | Series | Transform new data to bins |
| `pava_blocks_()` | List | Get PAVA blocks before merging |
| `pava_groups_()` | DataFrame | Get grouped statistics from PAVA |

#### Usage
```python
from MOBPY import MonotonicBinner

# Binary classification
binner = MonotonicBinner(
    df, x='age', y='default',
    constraints=BinningConstraints(max_bins=5)
)
binner.fit()

# Get results
bins = binner.bins_()      # Numeric bins
summary = binner.summary_() # With WoE/IV for binary

# Transform new data
labels = binner.transform(new_df['age'], assign='interval')
```

---

## Core Module (`core/`)

### Constraints (`constraints.py`)

#### BinningConstraints Class
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_bins` | int | 6 | Maximum number of bins |
| `min_bins` | int | 4 | Minimum number of bins |
| `max_samples` | float/int | None | Max samples per bin |
| `min_samples` | float/int | None | Min samples per bin |
| `min_positives` | float/int | None | Min positives per bin |
| `initial_pvalue` | float | 0.4 | Merge threshold |
| `maximize_bins` | bool | True | Maximize vs minimize bins |

#### Methods
| Method | Description |
|--------|-------------|
| `resolve(total_n, total_pos)` | Convert fractional to absolute |
| `validate()` | Check constraint consistency |
| `copy()` | Create a copy of constraints |
| `is_resolved()` | Check if constraints are resolved |

#### Usage
```python
from MOBPY.core import BinningConstraints

constraints = BinningConstraints(
    max_bins=6,
    min_samples=0.05,    # 5% of data
    min_positives=0.01   # 1% of positives
)
```

### PAVA Algorithm (`pava.py`)

#### PAVA Class
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `df` | DataFrame | required | Input data |
| `x` | str | required | Feature to group by |
| `y` | str | required | Target to aggregate |
| `sign` | str | "auto" | Monotonicity direction |
| `strict` | bool | True | Enforce strict monotonicity |

#### Methods
| Method | Returns | Description |
|--------|---------|-------------|
| `fit()` | self | Run PAVA algorithm |
| `export_blocks(as_dict)` | List | Export monotonic blocks |
| `validate_monotonicity()` | bool | Check if blocks are monotonic |
| `get_diagnostics()` | Dict | Get fitting diagnostics |

#### Usage
```python
from MOBPY.core.pava import PAVA

pava = PAVA(df=data, x='feature', y='target', sign='auto')
pava.fit()
blocks = pava.export_blocks(as_dict=True)
```

### Block Merging (`merge.py`)

#### Classes
| Class | Description |
|-------|-------------|
| `Block` | Data structure for block statistics |
| `MergeScorer` | Scoring system for merge decisions |
| `MergeStrategy` | Enum of merge strategies |

#### Main Function: merge_adjacent
| Parameter | Type | Description |
|-----------|------|-------------|
| `blocks` | List[Block/Dict] | Initial blocks from PAVA |
| `constraints` | BinningConstraints | Resolved constraints |
| `is_binary_y` | bool | Whether target is binary |
| `strategy` | MergeStrategy | Merge selection strategy |
| `history` | List | Optional merge history tracking |

#### MergeStrategy Options
- `HIGHEST_PVALUE`: Merge most similar blocks (default)
- `SMALLEST_LOSS`: Minimize information loss
- `BALANCED_SIZE`: Balance block sizes

#### Usage
```python
from MOBPY.core.merge import merge_adjacent, MergeStrategy

merged_blocks = merge_adjacent(
    blocks=pava_blocks,
    constraints=constraints,
    is_binary_y=True,
    strategy=MergeStrategy.HIGHEST_PVALUE
)
```

### Utilities (`utils.py`)

#### Data Validation Functions
| Function | Description |
|----------|-------------|
| `ensure_numeric_series(s, name)` | Validate numeric data |
| `is_binary_series(s, strict)` | Check if series is binary |
| `validate_column_exists(df, columns)` | Check columns exist |

#### Data Partitioning
| Function | Returns | Description |
|----------|---------|-------------|
| `partition_df(df, x, exclude_values)` | Parts | Split into clean/missing/excluded |

#### Parts Class
- Attributes: `clean`, `missing`, `excluded` (DataFrames)
- Methods: `summary()`, `validate()`

#### Statistical Functions
| Function | Description |
|----------|-------------|
| `calculate_correlation(x, y, method)` | Compute correlation coefficient |
| `woe_iv(goods, bads, smoothing)` | Calculate WoE and IV |

#### Usage
```python
from MOBPY.core.utils import (
    partition_df, woe_iv, is_binary_series
)

# Partition data
parts = partition_df(df, x='feature', exclude_values=[-999])

# Check target type
is_binary = is_binary_series(df['target'])

# Calculate WoE/IV (corrected function name)
woe_vals, iv_vals = woe_iv(goods, bads, smoothing=0.5)
```

---

## Plotting Module (`plot/`)

### PAVA Visualization (`csd_gcm.py`)

| Function | Description | Main Parameters |
|----------|-------------|-----------------|
| `plot_gcm()` | Visualize Greatest Convex Minorant | groups_df, blocks, ax |
| `plot_pava_comparison()` | Side-by-side CSD and GCM | binner, figsize |
| `plot_pava_process()` | Step-by-step PAVA process | groups_df, blocks |

#### Usage
```python
from MOBPY.plot import plot_gcm, plot_pava_comparison

# After fitting
fig = plot_pava_comparison(binner)
```

### Binning Results (`mob_plot.py`)

| Function | Description | Main Parameters |
|----------|-------------|-----------------|
| `plot_woe_bars()` | WoE bar chart | summary_df, ax |
| `plot_event_rate()` | Event rate with distribution | summary_df, ax |
| `plot_bin_statistics()` | Multi-panel comprehensive view | binner, figsize |
| `plot_sample_distribution()` | Sample distribution across bins | summary_df, plot_type |
| `plot_bin_boundaries()` | Boundaries on feature distribution | df, binner, kde |
| `plot_binning_stability()` | Compare train/test stability | train_summary, test_summary |

#### Usage
```python
from MOBPY.plot import (
    plot_bin_statistics,
    plot_woe_bars,
    plot_binning_stability
)

# Comprehensive view
fig = plot_bin_statistics(binner)

# WoE analysis
ax = plot_woe_bars(binner.summary_())

# Stability check
fig = plot_binning_stability(train_summary, test_summary)
```

---

## Complete Workflow Example

```python
import pandas as pd
from MOBPY import MonotonicBinner, BinningConstraints
from MOBPY.plot import plot_bin_statistics
from MOBPY.config import set_config
from MOBPY.logging_utils import set_verbosity

# 1. Configure
set_config(epsilon=1e-10, enable_progress_bar=True)
set_verbosity('INFO')  # Enable info logging

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

# 8. Check stability
test_binner = MonotonicBinner(test_df, x='Age', y='default')
test_binner.fit()
fig = plot_binning_stability(
    binner.summary_(), 
    test_binner.summary_()
)
```

---

## Testing Coverage

### Test Modules
| Module | Tests | Coverage Focus |
|--------|-------|----------------|
| `test_config.py` | Configuration system | Settings, persistence, environment vars |
| `test_constraints.py` | BinningConstraints | Validation, resolution, edge cases |
| `test_pava.py` | PAVA algorithm | Monotonicity, convergence, edge cases |
| `test_merge.py` | Block merging | Strategies, constraints, statistics |
| `test_mob.py` | MonotonicBinner | End-to-end pipeline, transformations |
| `test_utils.py` | Utilities | Validation, partitioning, WoE/IV |
| `test_plotting.py` | Visualizations | Plot creation, no errors |
| `test_exceptions.py` | Custom exceptions | Error handling, messages |
| `test_logging_utils.py` | Logging utilities | Logger creation, verbosity settings |

### Running Tests
```bash
# Run all tests
pytest

# Run specific module
pytest tests/test_pava.py

# With coverage
pytest --cov=MOBPY --cov-report=html

# Run property-based tests
pytest tests/test_property_based.py
```

---

## Key Design Principles

1. **Separation of Concerns**
   - Core algorithms (PAVA, merge) separated from orchestration
   - Visualization independent from computation
   - Configuration isolated in dedicated module

2. **Extensibility**
   - Strategy pattern for merge decisions
   - Pluggable visualization functions
   - Configurable constraints system

3. **Robustness**
   - Comprehensive input validation
   - Custom exception hierarchy
   - Edge case handling (∞ boundaries)

4. **Performance**
   - O(n log n) sorting + O(n) PAVA
   - Vectorized operations where possible
   - Optional progress tracking

5. **Testability**
   - Property-based testing with Hypothesis
   - Unit tests for all components
   - Integration tests for full pipeline

---

## Dependencies

### Core Dependencies
- numpy: Numerical operations
- pandas: Data handling
- scipy: Statistical tests (t-test in merging)
- matplotlib: Plotting (optional)

### Development Dependencies
- pytest: Testing framework
- hypothesis: Property-based testing

---

## License & Contributing

- **License**: MIT
- **Issues**: GitHub issue tracker
- **PRs**: Welcome with tests