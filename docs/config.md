# Configuration Module Documentation

## Overview
The configuration module provides global settings management for MOBPY through the `MOBPYConfig` dataclass. It supports runtime configuration, environment variables, and persistence.

## Module Location
`src/MOBPY/config.py`

## MOBPYConfig Class

### Default Configuration
```python
@dataclass
class MOBPYConfig:
    epsilon: float = 1e-12          # Numerical tolerance
    max_iterations: int = 1000      # Max iterations for algorithms
    enable_progress_bar: bool = False  # Progress tracking
    n_jobs: int = 1                 # Parallel processing
    random_state: Optional[int] = None  # RNG seed
    warn_on_constraint_violation: bool = True
    validate_inputs: bool = True    # Input validation
    cache_size: int = 128           # MB for caching
    precision: int = 6              # Decimal precision
```

## Global Functions

### get_config()
Gets the global MOBPY configuration singleton.

```python
from MOBPY.config import get_config

config = get_config()
print(f"Current epsilon: {config.epsilon}")
```

### set_config(**kwargs)
Updates global configuration parameters.

```python
from MOBPY.config import set_config

set_config(epsilon=1e-10, n_jobs=4, enable_progress_bar=True)
```

### reset_config()
Resets global configuration to defaults.

```python
from MOBPY.config import reset_config

reset_config()  # Back to all defaults
```

## Class Methods

### Instance Management

```python
# Create custom configuration
config = MOBPYConfig(epsilon=1e-10, n_jobs=-1)

# Save to file
config.save("my_config.json")

# Load from file
config = MOBPYConfig.from_file("my_config.json")

# Load from environment variables (MOBPY_ prefix)
config = MOBPYConfig.from_env()

# Export as dictionary
config_dict = config.to_dict()
```

## Environment Variables

Set configuration via environment:

```bash
export MOBPY_EPSILON=1e-10
export MOBPY_N_JOBS=4
export MOBPY_ENABLE_PROGRESS_BAR=true
```

Then load in Python:
```python
config = MOBPYConfig.from_env()
```

## Usage Examples

### Basic Configuration
```python
from MOBPY.config import set_config

# Configure for production
set_config(
    epsilon=1e-10,      # Higher precision
    n_jobs=-1,          # Use all cores
    validate_inputs=True,  # Ensure data quality
    enable_progress_bar=False  # No output in production
)
```

### Development Configuration
```python
# Configure for debugging
set_config(
    enable_progress_bar=True,
    warn_on_constraint_violation=True,
    validate_inputs=True,
    precision=8  # More decimal places for debugging
)
```

### Configuration Context Manager
```python
from MOBPY.config import get_config, set_config

# Temporarily change configuration
original_epsilon = get_config().epsilon
try:
    set_config(epsilon=1e-15)
    # Do precision-critical work
finally:
    set_config(epsilon=original_epsilon)
```

## Performance Impact

| Setting | Impact | Recommendation |
|---------|--------|----------------|
| `epsilon` | Affects convergence | Use 1e-12 for most cases |
| `n_jobs` | Parallel processing | -1 for all cores |
| `validate_inputs` | Adds overhead | Disable in production if data is trusted |
| `cache_size` | Memory usage | Increase for large datasets |

## See Also
- [MOBPYConfig API](../api/config.md)
- [Logging Configuration](./logging_utils.md)
- [MonotonicBinner](./binning/mob.md)