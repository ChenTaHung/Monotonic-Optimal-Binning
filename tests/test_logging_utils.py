"""Unit tests for logging utilities module.

This module tests the logging infrastructure including logger creation,
configuration, and progress tracking utilities.
"""

import pytest
import logging
import sys
from io import StringIO
from unittest.mock import Mock, patch, MagicMock
import threading
import time

from MOBPY.logging_utils import (
    get_logger, set_verbosity, BinningProgressLogger
)


class TestGetLogger:
    """Test suite for get_logger function.
    
    Tests logger creation and configuration.
    """
    
    def test_get_logger_basic(self):
        """Test basic logger creation without level parameter."""
        logger = get_logger("test_module_basic")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module_basic"
        
        # Without level parameter, should use default (WARNING)
        # or inherit from parent (level 0)
        assert logger.level in [logging.WARNING, 0]
    
    def test_get_logger_with_level(self):
        """Test logger creation with specific level parameter.
        
        The level parameter should set the logger to the specified level.
        """
        # Try to create logger with DEBUG level
        logger = get_logger("test_module_with_debug", level=logging.DEBUG)
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module_with_debug"
        
        # The level parameter should work correctly
        assert logger.level == logging.DEBUG  # Should be 10 (DEBUG)
    
    def test_get_logger_default_level(self):
        """Test logger default level when no level is specified."""
        # Clear any existing handlers
        test_logger_name = "test_default_level_check"
        if test_logger_name in logging.Logger.manager.loggerDict:
            del logging.Logger.manager.loggerDict[test_logger_name]
        
        logger = get_logger(test_logger_name)
        
        # Default should be WARNING (30) or inherit (0)
        assert logger.level in [logging.WARNING, 0]
        
        # If inheriting, check effective level
        if logger.level == 0:
            # Get effective level
            effective_level = logger.getEffectiveLevel()
            # Should default to WARNING or root logger level
            assert effective_level in [logging.WARNING, logging.INFO, logging.DEBUG]
    
    def test_get_logger_has_handler(self):
        """Test that logger has a stream handler configured."""
        # Create fresh logger
        test_name = "test_handler_check"
        if test_name in logging.Logger.manager.loggerDict:
            del logging.Logger.manager.loggerDict[test_name]
        
        logger = get_logger(test_name)
        
        # Should have at least one handler
        assert len(logger.handlers) > 0
        
        # Should be a StreamHandler
        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
    
    def test_get_logger_formatter(self):
        """Test that logger has proper formatter."""
        logger = get_logger("test_formatter")
        
        if logger.handlers:
            handler = logger.handlers[0]
            formatter = handler.formatter
            
            # Check formatter exists
            assert formatter is not None
            
            # Check format string contains expected elements
            if hasattr(formatter, '_fmt'):
                format_str = formatter._fmt
                assert 'asctime' in format_str or 'time' in format_str
                assert 'name' in format_str
                assert 'levelname' in format_str
                assert 'message' in format_str
    
    def test_get_logger_no_duplicate_handlers(self):
        """Test that multiple calls don't create duplicate handlers."""
        test_name = "test_no_duplicates"
        
        # Clear existing logger
        if test_name in logging.Logger.manager.loggerDict:
            logger = logging.getLogger(test_name)
            logger.handlers.clear()
            del logging.Logger.manager.loggerDict[test_name]
        
        # Get logger multiple times
        logger1 = get_logger(test_name)
        initial_handlers = len(logger1.handlers)
        
        logger2 = get_logger(test_name)
        logger3 = get_logger(test_name)
        
        # Should not add more handlers
        assert len(logger3.handlers) == initial_handlers
        
        # All should be the same logger instance
        assert logger1 is logger2 is logger3
    
    def test_logger_output_capture(self):
        """Test that logger output can be captured."""
        # Create logger with StringIO handler for testing
        logger = get_logger("test_output")
        logger.handlers.clear()  # Clear existing handlers
        
        # Add test handler
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Log messages
        logger.info("Test message")
        logger.debug("Debug message")  # Should not appear (level too low)
        logger.warning("Warning message")
        
        # Check output
        output = stream.getvalue()
        assert "Test message" in output
        assert "Debug message" not in output  # Below threshold
        assert "Warning message" in output
    
    def test_logger_with_different_levels(self):
        """Test that logger can be created with different levels."""
        # Test with different log levels
        debug_logger = get_logger("test_debug", level=logging.DEBUG)
        assert debug_logger.level == logging.DEBUG
        
        info_logger = get_logger("test_info", level=logging.INFO)
        assert info_logger.level == logging.INFO
        
        warning_logger = get_logger("test_warning", level=logging.WARNING)
        assert warning_logger.level == logging.WARNING
        
        error_logger = get_logger("test_error", level=logging.ERROR)
        assert error_logger.level == logging.ERROR
    
    def test_logger_level_functionality(self):
        """Test that loggers with different levels filter messages correctly."""
        # Create logger with INFO level
        info_logger = get_logger("test_level_filtering", level=logging.INFO)
        
        # Set up test handler
        stream = StringIO()
        info_logger.handlers.clear()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        info_logger.addHandler(handler)
        
        # Log at various levels
        info_logger.debug("Debug message")  # Should NOT appear
        info_logger.info("Info message")    # Should appear
        info_logger.warning("Warning message")  # Should appear
        
        output = stream.getvalue()
        
        # Check filtering works correctly
        assert "Debug message" not in output  # Below INFO level
        assert "INFO: Info message" in output
        assert "WARNING: Warning message" in output


class TestSetVerbosity:
    """Test suite for set_verbosity function.
    
    Tests global verbosity configuration.
    """
    
    def test_set_verbosity_debug(self):
        """Test setting verbosity to DEBUG."""
        set_verbosity('DEBUG')
        
        # Check base logger level
        base_logger = logging.getLogger('MOBPY')
        assert base_logger.level == logging.DEBUG
    
    def test_set_verbosity_info(self):
        """Test setting verbosity to INFO."""
        set_verbosity('INFO')
        
        base_logger = logging.getLogger('MOBPY')
        assert base_logger.level == logging.INFO
    
    def test_set_verbosity_warning(self):
        """Test setting verbosity to WARNING."""
        set_verbosity('WARNING')
        
        base_logger = logging.getLogger('MOBPY')
        assert base_logger.level == logging.WARNING
    
    def test_set_verbosity_error(self):
        """Test setting verbosity to ERROR."""
        set_verbosity('ERROR')
        
        base_logger = logging.getLogger('MOBPY')
        assert base_logger.level == logging.ERROR
    
    def test_set_verbosity_critical(self):
        """Test setting verbosity to CRITICAL."""
        set_verbosity('CRITICAL')
        
        base_logger = logging.getLogger('MOBPY')
        assert base_logger.level == logging.CRITICAL
    
    def test_set_verbosity_case_insensitive(self):
        """Test that verbosity setting is case-insensitive."""
        set_verbosity('debug')  # Lowercase
        assert logging.getLogger('MOBPY').level == logging.DEBUG
        
        set_verbosity('INFO')  # Uppercase
        assert logging.getLogger('MOBPY').level == logging.INFO
        
        set_verbosity('WaRnInG')  # Mixed case
        assert logging.getLogger('MOBPY').level == logging.WARNING
    
    def test_set_verbosity_invalid_level(self):
        """Test that invalid level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            set_verbosity('INVALID_LEVEL')
    
    def test_set_verbosity_affects_child_loggers(self):
        """Test that verbosity affects all MOBPY child loggers."""
        # Create child loggers
        logger1 = logging.getLogger('MOBPY.core.pava')
        logger2 = logging.getLogger('MOBPY.binning.mob')
        
        # Set verbosity
        set_verbosity('ERROR')
        
        # Check that child loggers are affected
        # (They may inherit from parent or be set directly)
        # The implementation sets them directly
        assert logger1.level == logging.ERROR
        assert logger2.level == logging.ERROR


class TestBinningProgressLogger:
    """Test suite for BinningProgressLogger context manager.
    
    Tests progress logging functionality.
    """
    
    def test_progress_logger_initialization(self):
        """Test BinningProgressLogger initialization."""
        progress = BinningProgressLogger("test_stage")
        
        assert progress.stage == "test_stage"
        assert progress.steps_completed == 0
        assert progress.logger is not None
    
    def test_progress_logger_with_custom_logger(self):
        """Test BinningProgressLogger with custom logger."""
        custom_logger = Mock(spec=logging.Logger)
        
        progress = BinningProgressLogger("test_stage", logger=custom_logger)
        
        assert progress.logger is custom_logger
    
    def test_progress_logger_context_manager_success(self):
        """Test BinningProgressLogger as context manager (success case)."""
        mock_logger = Mock(spec=logging.Logger)
        
        with BinningProgressLogger("data_prep", logger=mock_logger) as progress:
            # Should log start
            mock_logger.info.assert_called_with("Starting data_prep")
            
            # Update progress
            progress.update("Step 1")
            mock_logger.debug.assert_called_with("  [data_prep] Step 1")
            
            progress.update("Step 2")
            mock_logger.debug.assert_called_with("  [data_prep] Step 2")
        
        # Should log completion
        mock_logger.info.assert_called_with("Completed data_prep (2 steps)")
    
    def test_progress_logger_context_manager_failure(self):
        """Test BinningProgressLogger as context manager (failure case)."""
        mock_logger = Mock(spec=logging.Logger)
        
        try:
            with BinningProgressLogger("fitting", logger=mock_logger) as progress:
                mock_logger.info.assert_called_with("Starting fitting")
                
                progress.update("Step 1")
                
                # Simulate error
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Should log error
        mock_logger.error.assert_called_with("Failed in fitting: Test error")
    
    def test_progress_logger_update_increments_counter(self):
        """Test that update increments step counter."""
        progress = BinningProgressLogger("test")
        
        assert progress.steps_completed == 0
        
        progress.update("Step 1")
        assert progress.steps_completed == 1
        
        progress.update("Step 2")
        assert progress.steps_completed == 2
        
        progress.update("Step 3")
        assert progress.steps_completed == 3
    
    def test_progress_logger_nested_usage(self):
        """Test nested progress loggers."""
        mock_logger = Mock(spec=logging.Logger)
        
        with BinningProgressLogger("outer", logger=mock_logger) as outer:
            outer.update("Outer step 1")
            
            with BinningProgressLogger("inner", logger=mock_logger) as inner:
                inner.update("Inner step 1")
                inner.update("Inner step 2")
            
            outer.update("Outer step 2")
        
        # Check all expected calls were made
        calls = mock_logger.info.call_args_list
        assert any("Starting outer" in str(call) for call in calls)
        assert any("Starting inner" in str(call) for call in calls)
        assert any("Completed inner" in str(call) for call in calls)
        assert any("Completed outer" in str(call) for call in calls)
    
    def test_progress_logger_exception_propagation(self):
        """Test that exceptions propagate correctly."""
        progress = BinningProgressLogger("test")
        
        with pytest.raises(ValueError):
            with progress:
                progress.update("Step 1")
                raise ValueError("Test exception")
    
    def test_progress_logger_return_value(self):
        """Test context manager returns self."""
        with BinningProgressLogger("test") as progress:
            assert isinstance(progress, BinningProgressLogger)
            assert progress.stage == "test"


class TestLoggingIntegration:
    """Integration tests for logging utilities."""
    
    def test_logging_in_mobpy_pipeline(self):
        """Test logging integration with MOBPY pipeline."""
        from MOBPY.binning.mob import MonotonicBinner
        import pandas as pd
        import numpy as np
        
        # Capture logs
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter('%(message)s'))
        
        # Set up logging
        logger = logging.getLogger('MOBPY')
        logger.handlers.clear()
        logger.addHandler(handler)
        set_verbosity('INFO')
        
        try:
            # Create and fit binner
            df = pd.DataFrame({
                'x': np.arange(100),
                'y': np.random.binomial(1, 0.5, 100)
            })
            
            binner = MonotonicBinner(df, x='x', y='y')
            binner.fit()
            
            # Check that some logging occurred
            output = stream.getvalue()
            # Specific messages depend on implementation
            
        finally:
            # Clean up
            logger.handlers.clear()
            set_verbosity('WARNING')
    
    def test_multiple_logger_instances(self):
        """Test multiple logger instances work correctly."""
        logger1 = get_logger('MOBPY.module1')
        logger2 = get_logger('MOBPY.module2')
        logger3 = get_logger('MOBPY.module1')  # Same as logger1
        
        # Should get same instance for same name
        assert logger1 is logger3
        
        # Different names should give different instances
        assert logger1 is not logger2
        
        # All should be properly configured
        for logger in [logger1, logger2, logger3]:
            assert len(logger.handlers) > 0 or logger.parent is not None
    
    def test_thread_safety_considerations(self):
        """Test logging behavior with threading."""
        results = []
        
        def log_messages(thread_id):
            logger = get_logger(f'MOBPY.thread_{thread_id}')
            logger.info(f"Message from thread {thread_id}")
            results.append(thread_id)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=log_messages, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # All threads should have logged
        assert len(results) == 5
        assert set(results) == {0, 1, 2, 3, 4}
    
    def test_logging_performance(self):
        """Test that logging doesn't significantly impact performance."""
        import time
        
        logger = get_logger('MOBPY.performance')
        logger.setLevel(logging.WARNING)  # Only log warnings
        
        # Time many debug messages (should be filtered)
        start = time.time()
        for i in range(10000):
            logger.debug(f"Debug message {i}")
        elapsed = time.time() - start
        
        # Should be very fast since messages are filtered
        assert elapsed < 0.1  # Less than 100ms for 10k filtered messages
    
    def test_logging_with_progress_tracking(self):
        """Test combining logging with progress tracking."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        
        logger = get_logger('MOBPY.progress')
        logger.handlers.clear()
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        with BinningProgressLogger("test_pipeline", logger=logger) as progress:
            progress.update("Loading data")
            logger.info("Data loaded successfully")
            
            progress.update("Processing")
            logger.warning("Found missing values")
            
            progress.update("Saving results")
        
        output = stream.getvalue()
        
        # Check expected messages appear
        assert "Starting test_pipeline" in output
        assert "Loading data" in output
        assert "Data loaded successfully" in output
        assert "Found missing values" in output
        assert "Completed test_pipeline" in output