# Logging Utilities Module Documentation

## Overview
The logging_utils module provides consistent logging infrastructure across the MOBPY package. It includes logger creation, verbosity control, and progress tracking utilities for long-running operations.

## Module Location
`src/MOBPY/logging_utils.py`

## Main Functions

### `get_logger(name: str, level: Optional[int] = None) -> logging.Logger`
Get or create a logger with MOBPY-specific configuration.

**Parameters:**
- **name** (`str`): Logger name, typically `__name__` from the calling module
- **level** (`Optional[int]`): Optional logging level. If None, uses package default (WARNING)

**Returns:**
- `logging.Logger`: Configured logger instance

**Behavior:**
- Creates a logger with the given name if it doesn't exist
- Adds a StreamHandler with formatted output
- Avoids duplicate handlers on repeated calls
- Default level is WARNING to avoid cluttering output

**Example:**
```python
from MOBPY.logging_utils import get_logger

# In a module
logger = get_logger(__name__)
logger.info("Starting binning process")
logger.debug("Processing block %d", block_id)
logger.warning("Small bin detected: %d samples", n_samples)
logger.error("Failed to merge blocks: %s", error_msg)
```

### `set_verbosity(level: str) -> None`
Set global verbosity for all MOBPY loggers.

**Parameters:**
- **level** (`str`): One of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL' (case-insensitive)

**Raises:**
- `ValueError`: If level is not a valid logging level

**Behavior:**
- Sets the logging level for the base 'MOBPY' logger
- Updates all child loggers under the MOBPY namespace
- Affects all subsequent logging output package-wide

**Example:**
```python
from MOBPY.logging_utils import set_verbosity

# Enable detailed debug output
set_verbosity('DEBUG')

# Normal operation messages
set_verbosity('INFO')

# Only warnings and errors (default)
set_verbosity('WARNING')

# Only errors
set_verbosity('ERROR')

# Only critical errors
set_verbosity('CRITICAL')
```

## Classes

### `BinningProgressLogger`
Context manager for logging binning progress with structured output.

```python
class BinningProgressLogger:
    def __init__(self, stage: str, logger: Optional[logging.Logger] = None)
    def update(self, message: str) -> None
```

**Constructor Parameters:**
- **stage** (`str`): Name of the current stage (e.g., "data_preparation", "fitting")
- **logger** (`Optional[logging.Logger]`): Logger instance. If None, creates one

**Attributes:**
- **stage** (`str`): The stage name
- **logger** (`logging.Logger`): The logger being used
- **steps_completed** (`int`): Counter for completed steps

**Methods:**

#### `update(message: str) -> None`
Log a progress update within the current stage.

**Parameters:**
- **message** (`str`): Progress message to log

**Behavior:**
- Increments the steps_completed counter
- Logs message at DEBUG level with stage prefix

**Context Manager Behavior:**
- `__enter__`: Logs stage start at INFO level
- `__exit__`: Logs completion with step count at INFO level
- `__exit__` (on exception): Logs failure with error message at ERROR level

**Example:**
```python
from MOBPY.logging_utils import BinningProgressLogger

# Basic usage
with BinningProgressLogger("data_preparation") as progress:
    progress.update("Cleaning missing values")
    # ... do work ...
    progress.update("Partitioning data")
    # ... do work ...
    progress.update("Validating types")
# Automatically logs completion

# Nested progress tracking
with BinningProgressLogger("full_pipeline") as pipeline:
    pipeline.update("Starting preprocessing")
    
    with BinningProgressLogger("preprocessing") as preprocess:
        preprocess.update("Remove outliers")
        preprocess.update("Handle missing")
    
    pipeline.update("Starting fitting")
    
    with BinningProgressLogger("fitting") as fit:
        fit.update("Running PAVA")
        fit.update("Merging blocks")
    
    pipeline.update("Generating output")

# With custom logger
import logging
custom_logger = logging.getLogger('my_app.binning')
with BinningProgressLogger("analysis", logger=custom_logger) as progress:
    progress.update("Step 1")
    progress.update("Step 2")
```

## Logging Levels Guide

### When to Use Each Level

| Level | Value | When to Use | Example |
|-------|-------|-------------|---------|
| **DEBUG** | 10 | Detailed diagnostic info | Block merge scores, iteration details |
| **INFO** | 20 | General informational messages | Pipeline stages, major steps |
| **WARNING** | 30 | Something unexpected but handled | Small bins, relaxed constraints |
| **ERROR** | 40 | Serious problem occurred | Fitting failed, invalid data |
| **CRITICAL** | 50 | Program may not continue | Unrecoverable state |

## Integration with MOBPY Pipeline

### Typical Logging Flow
```python
from MOBPY import MonotonicBinner
from MOBPY.logging_utils import set_verbosity

# Enable detailed logging for debugging
set_verbosity('DEBUG')

# Create binner - will log at various levels
binner = MonotonicBinner(df, x='age', y='default')

# Fit process logs:
# INFO: Starting data_preparation
# DEBUG: [data_preparation] Validating columns
# DEBUG: [data_preparation] Partitioning by missing values
# INFO: Completed data_preparation (2 steps)
# INFO: Starting fitting
# DEBUG: [fitting] Running PAVA with sign=auto
# DEBUG: [fitting] Merging 15 blocks to satisfy constraints
# INFO: Completed fitting (2 steps)
binner.fit()

# Disable verbose output for production
set_verbosity('WARNING')
```

### Module-Specific Loggers

Different modules use their own loggers:

```python
# Core modules
logger = get_logger('MOBPY.core.pava')
logger = get_logger('MOBPY.core.merge')
logger = get_logger('MOBPY.core.utils')

# Binning module
logger = get_logger('MOBPY.binning.mob')

# Plot module
logger = get_logger('MOBPY.plot.mob_plot')
```

## Best Practices

### 1. Use Appropriate Levels
```python
logger = get_logger(__name__)

# Detailed algorithm state (DEBUG)
logger.debug("Merge score for blocks %d-%d: %.4f", i, j, score)

# Major pipeline steps (INFO)
logger.info("Starting PAVA with %d unique values", n_unique)

# Potential issues (WARNING)
logger.warning("Bin %d has only %d samples (threshold: %d)", 
              bin_id, n_samples, threshold)

# Actual failures (ERROR)
logger.error("PAVA convergence failed after %d iterations", max_iter)
```

### 2. Use Progress Logger for Stages
```python
def complex_operation(data):
    with BinningProgressLogger("complex_operation") as progress:
        progress.update("Preprocessing")
        preprocessed = preprocess(data)
        
        progress.update("Main computation")
        result = compute(preprocessed)
        
        progress.update("Postprocessing")
        final = postprocess(result)
        
    return final
```

### 3. Include Context in Messages
```python
# Good: includes context
logger.info("Merged %d blocks -> %d bins (compression: %.1fx)", 
           n_blocks, n_bins, n_blocks/n_bins)

# Less helpful
logger.info("Merging complete")
```

### 4. Use Structured Logging
```python
# Log with structured data for parsing
logger.info("Fitting complete: blocks=%d, bins=%d, time=%.2fs",
           n_blocks, n_bins, elapsed_time)
```

## Configuration Examples

### Development Environment
```python
# Verbose output for debugging
from MOBPY.logging_utils import set_verbosity
set_verbosity('DEBUG')

# Or configure specific module
import logging
logging.getLogger('MOBPY.core.merge').setLevel(logging.DEBUG)
```

### Production Environment
```python
# Minimal output
from MOBPY.logging_utils import set_verbosity
set_verbosity('WARNING')

# Or completely disable
set_verbosity('CRITICAL')
```

### Custom Handler
```python
import logging
from MOBPY.logging_utils import get_logger

# Create custom handler
file_handler = logging.FileHandler('mobpy.log')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)

# Add to MOBPY logger
mobpy_logger = logging.getLogger('MOBPY')
mobpy_logger.addHandler(file_handler)
```

## Thread Safety

The logging module is thread-safe by default:
- Multiple threads can safely call logging functions
- Each thread's messages are atomic
- No additional synchronization needed

```python
import threading
from MOBPY.logging_utils import get_logger

def worker(thread_id):
    logger = get_logger(f'MOBPY.worker_{thread_id}')
    logger.info("Thread %d starting", thread_id)
    # ... do work ...
    logger.info("Thread %d complete", thread_id)

threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

## Performance Considerations

### Logging Overhead
- Disabled log levels have minimal overhead (level check only)
- Use `logger.isEnabledFor(level)` for expensive message formatting

```python
# Expensive string formatting only if DEBUG is enabled
if logger.isEnabledFor(logging.DEBUG):
    debug_msg = expensive_debug_string(data)
    logger.debug(debug_msg)
```

### Message Formatting
```python
# Good: deferred formatting (faster when disabled)
logger.debug("Processing block %d of %d", current, total)

# Less efficient: immediate formatting
logger.debug(f"Processing block {current} of {total}")
```

## Troubleshooting

### No Output Visible
```python
# Check current verbosity
import logging
print(logging.getLogger('MOBPY').level)

# Enable output
from MOBPY.logging_utils import set_verbosity
set_verbosity('DEBUG')
```

### Too Much Output
```python
# Reduce verbosity
set_verbosity('WARNING')

# Or disable specific module
logging.getLogger('MOBPY.core.pava').setLevel(logging.WARNING)
```

### Output to File
```python
import logging

# Add file handler
handler = logging.FileHandler('debug.log')
logging.getLogger('MOBPY').addHandler(handler)
```

## Dependencies
- Python standard library `logging` module
- No external dependencies

## See Also
- [`MonotonicBinner`](./binning/mob.md) - Uses progress logging
- [`PAVA`](./core/pava.md) - Logs algorithm details
- [`merge_adjacent`](./core/merge.md) - Logs merge decisions
- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)