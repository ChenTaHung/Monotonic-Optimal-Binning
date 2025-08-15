"""Logging utilities for MOBPY.

Provides a consistent logging interface across the package with appropriate
default configurations and helper functions.
"""

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get or create a logger with MOBPY-specific configuration.
    
    Args:
        name: Logger name, typically __name__ from the calling module.
        level: Optional logging level. If None, uses package default.
        
    Returns:
        logging.Logger: Configured logger instance.
        
    Examples:
        >>> from MOBPY.logging_utils import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting binning process")
    """
    logger = logging.getLogger(name)
    
    # Only configure if no handlers exist (avoid duplicate handlers)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Set level from environment or use default
        if level is not None:
            logger.setLevel(level)
        else:
            # Default to WARNING to avoid cluttering user output
            logger.setLevel(logging.WARNING)
    
    return logger


def set_verbosity(level: str) -> None:
    """Set global verbosity for all MOBPY loggers.
    
    Args:
        level: One of 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
        
    Raises:
        ValueError: If level is not a valid logging level.
        
    Examples:
        >>> from MOBPY.logging_utils import set_verbosity
        >>> set_verbosity('DEBUG')  # Show all debug messages
        >>> set_verbosity('ERROR')  # Only show errors and critical
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Set level for all MOBPY loggers
    base_logger = logging.getLogger('MOBPY')
    base_logger.setLevel(numeric_level)
    
    # Also update all child loggers
    for logger_name in logging.Logger.manager.loggerDict:
        if logger_name.startswith('MOBPY'):
            logging.getLogger(logger_name).setLevel(numeric_level)


class BinningProgressLogger:
    """Context manager for logging binning progress.
    
    Provides structured logging for the binning pipeline stages.
    
    Examples:
        >>> with BinningProgressLogger("data_preparation") as progress:
        ...     progress.update("Cleaning missing values")
        ...     # ... do work ...
        ...     progress.update("Partitioning data")
    """
    
    def __init__(self, stage: str, logger: Optional[logging.Logger] = None):
        """Initialize progress logger.
        
        Args:
            stage: Name of the current stage.
            logger: Logger instance. If None, creates one.
        """
        self.stage = stage
        self.logger = logger or get_logger('MOBPY.progress')
        self.steps_completed = 0
        
    def __enter__(self):
        """Enter context and log stage start."""
        self.logger.info(f"Starting {self.stage}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and log stage completion or failure."""
        if exc_type is None:
            self.logger.info(f"Completed {self.stage} ({self.steps_completed} steps)")
        else:
            self.logger.error(f"Failed in {self.stage}: {exc_val}")
        return False
    
    def update(self, message: str) -> None:
        """Log a progress update.
        
        Args:
            message: Progress message to log.
        """
        self.steps_completed += 1
        self.logger.debug(f"  [{self.stage}] {message}")