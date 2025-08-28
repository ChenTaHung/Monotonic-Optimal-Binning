# Logging Module Documentation

## Overview
The logging module provides configurable logging and progress tracking for MOBPY operations. It supports different verbosity levels and progress bars for long-running operations.

## Module Location
`src/MOBPY/logging_utils.py`

## Main Functions

### set_verbosity(level: str)
Sets the global logging verbosity level.

**Parameters:**
- `level`: One of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'

**Example:**
```python
from MOBPY.logging_utils import set_verbosity

set_verbosity('DEBUG')  # Show all messages
set_verbosity('WARNING')  # Only warnings and errors
```

### get_logger(name: str, level: Optional[str] = None)
Gets a configured logger instance for a module.

**Parameters:**
- `name`: Logger name (typically `__name__`)
- `level`: Optional specific level for this logger

**Returns:** Configured logger instance

**Example:**
```python
from MOBPY.logging_utils import get_logger

logger = get_logger(__name__)
logger.info("Starting binning process")
logger.debug("Processing %d samples", len(df))
```

## BinningProgressLogger Class

Context manager for progress tracking in long operations.

**Example:**
```python
from MOBPY.logging_utils import BinningProgressLogger

with BinningProgressLogger("Fitting MonotonicBinner") as progress:
    progress.update("Running PAVA algorithm")
    # ... PAVA code ...
    progress.update("Merging blocks")
    # ... merge code ...
    progress.update("Building final bins")
    # ... finalization ...
```

## Verbosity Levels

| Level | Shows | Use Case |
|-------|-------|----------|
| **DEBUG** | All messages | Development, debugging |
| **INFO** | Progress updates | Normal operation |
| **WARNING** | Potential issues | Production monitoring |
| **ERROR** | Errors only | Minimal output |
| **CRITICAL** | Critical failures | Emergency only |

## Usage Patterns

### Basic Setup
```python
from MOBPY.logging_utils import set_verbosity

# For development
set_verbosity('DEBUG')

# For production
set_verbosity('WARNING')

# For silent operation
set_verbosity('CRITICAL')
```

### Module-Specific Logging
```python
import logging
from MOBPY.logging_utils import get_logger

# Module logger
logger = get_logger(__name__)

def process_data(df):
    logger.info("Processing %d rows", len(df))
    
    if df.empty:
        logger.warning("Empty DataFrame provided")
        return None
    
    try:
        result = complex_operation(df)
        logger.debug("Operation successful: %s", result)
    except Exception as e:
        logger.error("Operation failed: %s", str(e))
        raise
```

### Progress Tracking
```python
from MOBPY.logging_utils import BinningProgressLogger

def long_operation(data, steps=100):
    with BinningProgressLogger(f"Processing {len(data)} items") as progress:
        for i in range(steps):
            # Do work
            if i % 10 == 0:
                progress.update(f"Step {i}/{steps}")
            
        progress.update("Finalizing")
        return results
```

## Integration with Config Module

```python
from MOBPY.config import get_config
from MOBPY.logging_utils import set_verbosity

config = get_config()
if config.enable_progress_bar:
    set_verbosity('INFO')
else:
    set_verbosity('WARNING')
```

## Best Practices

1. **Use lazy formatting**: `logger.debug("Value: %s", value)` not `f"Value: {value}"`
2. **Set appropriate levels**: DEBUG for development, WARNING for production
3. **Use progress loggers**: For operations > 1 second
4. **Log exceptions**: Always log errors before re-raising

## Troubleshooting

### No Output Visible
Check current verbosity and enable output:
```python
set_verbosity('DEBUG')
```

### Too Much Output
Reduce verbosity or disable specific module:
```python
set_verbosity('WARNING')
logging.getLogger('MOBPY.core.pava').setLevel(logging.WARNING)
```

### Output to File
```python
import logging
handler = logging.FileHandler('mobpy.log')
logging.getLogger('MOBPY').addHandler(handler)
```

## See Also
- [Configuration Module](./config.md) - Global settings
- [MonotonicBinner](./binning/mob.md) - Uses progress logging
- [PAVA Algorithm](./core/pava.md) - Logs algorithm details