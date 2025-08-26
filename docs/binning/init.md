Great! I'll update the dependencies to include scipy. Now let's create the documentation for the `binning/__init__.py` module:

## **Binning Module Documentation** (`binning/__init__.py`)

```markdown
# Binning Module Documentation

## Overview
The binning module provides the main user-facing API for monotonic optimal binning. It orchestrates the complete pipeline from data partitioning through PAVA and merging to final bin creation.

## Module Location
`src/MOBPY/binning/__init__.py`

## Main Classes

### `MonotonicBinner`
End-to-end monotonic optimal binning orchestrator that handles the complete binning workflow.

**Import:**
```python
from MOBPY.binning import MonotonicBinner
```

**Key Responsibilities:**
- Data partitioning and preprocessing
- PAVA (Pool-Adjacent-Violators Algorithm) fitting
- Constraint-based block merging
- WoE (Weight of Evidence) and IV (Information Value) calculation for binary targets
- Bin edge management and transformation

## Module Structure

```
binning/
├── __init__.py       # Module initialization and exports
└── mob.py            # MonotonicBinner implementation
```

## Usage Example

```python
from MOBPY.binning import MonotonicBinner
from MOBPY.core import BinningConstraints
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'age': np.random.uniform(18, 80, n),
    'default': np.random.binomial(1, 0.2, n)
})

# Configure constraints
constraints = BinningConstraints(
    max_bins=5,           # Maximum number of bins
    min_samples=0.05      # Minimum 5% of samples per bin
)

# Initialize and fit the binner
binner = MonotonicBinner(
    df=df,
    x='age',
    y='default', 
    constraints=constraints
)
binner.fit()

# Retrieve binning results
bins = binner.bins_()        # Get bin boundaries
summary = binner.summary_()  # Get detailed statistics

# Transform new data
new_ages = [25, 45, 65]
bin_labels = binner.transform(new_ages, assign='interval')
```

## Integration with Core Modules

The binning module integrates with several core components:

1. **`core.constraints`**: Uses `BinningConstraints` for configuration
2. **`core.pava`**: Leverages PAVA algorithm for monotone fitting
3. **`core.merge`**: Applies block merging strategies
4. **`core.utils`**: Utilizes helper functions for data processing
5. **`plot`**: Provides visualization capabilities

## Export Configuration

The module exports:
```python
__all__ = ["MonotonicBinner"]
```

This ensures clean namespace when using:
```python
from MOBPY.binning import *
```

## Design Principles

1. **Single Responsibility**: The module focuses solely on binning orchestration
2. **Clean API**: Provides a simple, intuitive interface for end users
3. **Separation of Concerns**: Delegates algorithmic details to core modules
4. **Extensibility**: Designed to accommodate future binning strategies

## Error Handling

The module handles various error conditions:
- Invalid input data formats
- Constraint violations
- Insufficient data for binning
- Numerical instabilities

## Performance Considerations

- Optimized for datasets with 10² to 10⁶ samples
- Memory efficient through grouped operations
- Scales linearly with number of unique x values

## Dependencies
- numpy
- pandas
- scipy
- MOBPY.core modules
- MOBPY.plot modules (optional, for visualization)

## See Also
- [`MonotonicBinner`](./mob.md) - Detailed documentation of the main class
- [`BinningConstraints`](../core/constraints.md) - Constraint configuration
- [`PAVA`](../core/pava.md) - Core algorithm documentation
```
