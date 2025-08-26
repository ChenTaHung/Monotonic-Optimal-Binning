# Configuration Module Documentation

## Overview
The configuration module provides global settings that affect the behavior of the entire MOBPY package. Users can customize numerical tolerances, display options, performance settings, and other behaviors through a centralized configuration system.

## Module Location
`src/MOBPY/config.py`

## Main Class

### `MOBPYConfig`
Global configuration dataclass for the MOBPY package.

```python
@dataclass
class MOBPYConfig:
    # Numerical settings
    epsilon: float = 1e-12
    max_iterations: int = 1000
    
    # Display settings  
    enable_progress_bar: bool = False
    plot_style: str = "seaborn-v0_8-darkgrid"
    display_precision: int = 4
    
    # Performance settings
    n_jobs: int = 1
    random_state: Optional[int] = None
    
    # Warning settings
    warn_on_small_bins: bool = True
    small_bin_threshold: int = 30
    
    # Advanced settings
    cache_intermediate_results: bool = False
    validate_inputs: bool = True
```

## Configuration Parameters

### Numerical Settings
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **epsilon** | `float` | `1e-12` | Numerical tolerance for floating-point comparisons |
| **max_iterations** | `int` | `1000` | Maximum iterations for iterative algorithms |

### Display Settings
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **enable_progress_bar** | `bool` | `False` | Whether to show progress bars in long operations |
| **plot_style** | `str` | `"seaborn-v0_8-darkgrid"` | Default matplotlib style for plots |
| **display_precision** | `int` | `4` | Number of decimal places in displays |

### Performance Settings
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **n_jobs** | `int` | `1` | Number of parallel jobs (-1 for all CPUs) |
| **random_state** | `Optional[int]` | `None` | Random seed for reproducibility |

### Warning Settings
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **warn_on_small_bins** | `bool` | `True` | Whether to warn when bins have few samples |
| **small_bin_threshold** | `int` | `30` | Threshold for small bin warnings |

### Advanced Settings
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **cache_intermediate_results** | `bool` | `False` | Whether to cache intermediate computation results |
| **validate_inputs** | `bool` | `True` | Whether to validate input data types and values |

## Methods

### Instance Methods

#### `set(**kwargs) -> None`
Updates configuration values.

```python
config = MOBPYConfig()
config.set(epsilon=1e-10, n_jobs=-1)
```

**Raises:** `AttributeError` if trying to set non-existent parameter

#### `reset() -> None`
Resets all configuration to default values.

```python
config.reset()  # Back to defaults
```

#### `to_dict() -> Dict[str, Any]`
Exports configuration as dictionary.

```python
config_dict = config.to_dict()
# {'epsilon': 1e-12, 'max_iterations': 1000, ...}
```

#### `from_dict(config_dict: Dict[str, Any]) -> None`
Loads configuration from dictionary.

```python
config.from_dict({'epsilon': 1e-10, 'n_jobs': 4})
```

#### `save(filepath: str) -> None`
Saves configuration to JSON file.

```python
config.save("my_config.json")
```

#### `load(filepath: str) -> None`
Loads configuration from JSON file.

```python
config.load("my_config.json")
```

**Raises:** `FileNotFoundError` if file doesn't exist

### Class Methods

#### `from_file(filepath: str) -> MOBPYConfig`
Creates configuration instance from file.

```python
config = MOBPYConfig.from_file("config.json")
```

#### `from_env() -> MOBPYConfig`
Creates configuration from environment variables.

```python
# Set environment variable
os.environ["MOBPY_EPSILON"] = "1e-10"
os.environ["MOBPY_N_JOBS"] = "4"

config = MOBPYConfig.from_env()
```

**Environment Variable Format:** `MOBPY_<PARAMETER_NAME>`
- Boolean values: 'true', '1', 'yes' → True
- All other values → False

## Global Configuration Functions

### `get_config() -> MOBPYConfig`
Gets the global MOBPY configuration singleton.

```python
from MOBPY.config import get_config

config = get_config()
print(f"Current epsilon: {config.epsilon}")
```

### `set_config(**kwargs) -> None`
Updates global configuration.

```python
from MOBPY.config import set_config

set_config(epsilon=1e-10, n_jobs=4)
```

### `reset_config() -> None`
Resets global configuration to defaults.

```python
from MOBPY.config import reset_config

reset_config()  # Back to all defaults
```

## Usage Examples

### Basic Configuration
```python
from MOBPY.config import get_config, set_config

# Get current configuration
config = get_config()
print(f"Epsilon: {config.epsilon}")

# Update specific settings
set_config(
    epsilon=1e-10,
    enable_progress_bar=True,
    n_jobs=-1  # Use all CPUs
)
```

### Temporary Configuration
```python
# Store original
original = get_config().to_dict()

try:
    # Use different settings
    set_config(epsilon=1e-8, max_iterations=100)
    
    # Run your code here
    # ...
    
finally:
    # Restore original
    get_config().from_dict(original)
```

### Configuration from File
```python
# Save current configuration
config = get_config()
config.save("production_config.json")

# Later, load configuration
from MOBPY.config import MOBPYConfig
config = MOBPYConfig.from_file("production_config.json")

# Or update global config
get_config().load("production_config.json")
```

### Environment-Based Configuration
```python
# In your shell or .env file
export MOBPY_EPSILON=1e-10
export MOBPY_MAX_ITERATIONS=500
export MOBPY_ENABLE_PROGRESS_BAR=true
export MOBPY_N_JOBS=4

# In Python
from MOBPY.config import MOBPYConfig
config = MOBPYConfig.from_env()
```

### Context-Specific Configuration
```python
# For debugging
debug_config = {
    'epsilon': 1e-15,  # Very strict
    'validate_inputs': True,
    'enable_progress_bar': True,
    'warn_on_small_bins': True
}

# For production
prod_config = {
    'epsilon': 1e-10,
    'validate_inputs': False,  # Skip for speed
    'enable_progress_bar': False,
    'n_jobs': -1  # Use all CPUs
}

# Apply based on environment
import os
if os.getenv('DEBUG'):
    set_config(**debug_config)
else:
    set_config(**prod_config)
```

## Impact on MOBPY Components

### Numerical Comparisons
All floating-point comparisons use `epsilon`:
```python
# In PAVA
if abs(mean1 - mean2) <= config.epsilon:
    # Treat as equal
```

### Progress Reporting
Progress bars controlled by `enable_progress_bar`:
```python
if config.enable_progress_bar:
    # Show progress
```

### Parallel Processing
`n_jobs` affects parallel operations:
```python
# n_jobs = -1: Use all CPUs
# n_jobs = 1: No parallelization
# n_jobs > 1: Use specific number of jobs
```

### Input Validation
Controlled by `validate_inputs`:
```python
if config.validate_inputs:
    ensure_numeric_series(data)
```

## Configuration Best Practices

### 1. Use Appropriate Epsilon
```python
# For financial data with high precision
set_config(epsilon=1e-15)

# For general statistics
set_config(epsilon=1e-10)

# For noisy data
set_config(epsilon=1e-8)
```

### 2. Optimize for Your Use Case
```python
# Development/debugging
set_config(
    validate_inputs=True,
    enable_progress_bar=True,
    warn_on_small_bins=True
)

# Production
set_config(
    validate_inputs=False,
    enable_progress_bar=False,
    n_jobs=-1
)
```

### 3. Maintain Configuration Files
```python
# Create configs for different environments
configs = {
    'dev': {'epsilon': 1e-12, 'validate_inputs': True},
    'test': {'epsilon': 1e-10, 'max_iterations': 100},
    'prod': {'epsilon': 1e-10, 'n_jobs': -1}
}

# Save each
for env, cfg in configs.items():
    config = MOBPYConfig()
    config.from_dict(cfg)
    config.save(f"{env}_config.json")
```

## Thread Safety Considerations

**Important:** The global configuration is a singleton shared across all threads.

```python
# Changes affect all threads
set_config(epsilon=1e-8)  # All threads see this change

# For thread-local config, create separate instances
thread_config = MOBPYConfig()
thread_config.set(epsilon=1e-10)
# Use thread_config instead of global
```

## Performance Implications

| Setting | Impact | Recommendation |
|---------|--------|----------------|
| **epsilon** | Smaller = more iterations | Balance precision vs speed |
| **max_iterations** | Higher = better convergence | Set based on data size |
| **validate_inputs** | Adds overhead | Disable in production |
| **n_jobs** | Parallelization overhead | Use for large datasets |
| **cache_intermediate_results** | Memory usage | Enable for repeated operations |

## Debugging Configuration Issues

```python
# Check current configuration
config = get_config()
print(config.to_dict())

# Verify settings are applied
assert config.epsilon == expected_value

# Log configuration changes
import logging
logger = logging.getLogger(__name__)
logger.info(f"Configuration: {config.to_dict()}")
```

## Integration with Other Modules

- **PAVA**: Uses `epsilon` for monotonicity checks
- **Merge**: Uses `epsilon` for statistical tests
- **Utils**: Uses `validate_inputs` for data validation
- **Plot**: Uses `plot_style` for visualizations
- **All modules**: Respect `enable_progress_bar` setting

## Future Enhancements

Potential future additions:
- Context manager for temporary changes
- Configuration profiles
- Automatic tuning based on data
- Per-module configuration overrides

## See Also
- [`MonotonicBinner`](./binning/mob.md) - Uses configuration
- [`PAVA`](./core/pava.md) - Respects epsilon setting
- [`merge_adjacent`](./core/merge.md) - Uses numerical tolerances
- [`MOBPlot`](./plot/mob_plot.md) - Uses plot_style setting
