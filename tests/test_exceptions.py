"""Unit tests for custom exceptions module.

This module tests the custom exception hierarchy and ensures
proper error handling throughout the MOBPY package.
"""

import pytest
import warnings
from unittest.mock import Mock, patch

from MOBPY.exceptions import (
    MOBPYError, DataError, ConstraintError, 
    FittingError, NotFittedError, BinningWarning
)


class TestExceptionHierarchy:
    """Test suite for exception class hierarchy.
    
    Verifies inheritance relationships and basic functionality.
    """
    
    def test_base_exception_class(self):
        """Test MOBPYError is the base for all custom exceptions."""
        # Create base exception
        error = MOBPYError("Base error message")
        
        assert isinstance(error, Exception)
        assert str(error) == "Base error message"
    
    def test_data_error_inheritance(self):
        """Test DataError inherits from MOBPYError."""
        error = DataError("Data validation failed")
        
        assert isinstance(error, MOBPYError)
        assert isinstance(error, Exception)
        assert str(error) == "Data validation failed"
    
    def test_constraint_error_inheritance(self):
        """Test ConstraintError inherits from MOBPYError."""
        error = ConstraintError("Invalid constraints")
        
        assert isinstance(error, MOBPYError)
        assert isinstance(error, Exception)
        assert str(error) == "Invalid constraints"
    
    def test_fitting_error_inheritance(self):
        """Test FittingError inherits from MOBPYError."""
        error = FittingError("Fitting failed")
        
        assert isinstance(error, MOBPYError)
        assert isinstance(error, Exception)
        assert str(error) == "Fitting failed"
    
    def test_not_fitted_error_inheritance(self):
        """Test NotFittedError inherits from MOBPYError."""
        error = NotFittedError("Model not fitted")
        
        assert isinstance(error, MOBPYError)
        assert isinstance(error, Exception)
        assert str(error) == "Model not fitted"
    
    def test_binning_warning_inheritance(self):
        """Test BinningWarning inherits from UserWarning."""
        warning = BinningWarning("Warning message")
        
        assert isinstance(warning, UserWarning)
        assert str(warning) == "Warning message"
    
    def test_catch_all_mobpy_errors(self):
        """Test that all MOBPY errors can be caught with base class."""
        errors = [
            DataError("data error"),
            ConstraintError("constraint error"),
            FittingError("fitting error"),
            NotFittedError("not fitted error")
        ]
        
        for error in errors:
            try:
                raise error
            except MOBPYError as e:
                # Should catch all MOBPY-specific errors
                assert isinstance(e, MOBPYError)
            except Exception:
                pytest.fail(f"Failed to catch {type(error).__name__} with MOBPYError")


class TestDataError:
    """Test suite for DataError exception.
    
    Tests various data validation error scenarios.
    """
    
    def test_missing_column_error(self):
        """Test DataError for missing column."""
        error = DataError("Column 'x' not found in DataFrame")
        
        assert "Column" in str(error)
        assert "not found" in str(error)
    
    def test_non_numeric_data_error(self):
        """Test DataError for non-numeric data."""
        error = DataError(
            "Column 'feature' must be numeric, but got dtype=object. "
            "Sample values: ['a', 'b', 'c']"
        )
        
        assert "numeric" in str(error)
        assert "dtype=object" in str(error)
    
    def test_infinite_values_error(self):
        """Test DataError for infinite values."""
        error = DataError(
            "Column 'y' contains 5 non-finite values (inf or -inf) "
            "at indices [10, 20, 30, 40, 50]"
        )
        
        assert "non-finite" in str(error)
        assert "inf" in str(error)
    
    def test_empty_dataframe_error(self):
        """Test DataError for empty DataFrame."""
        error = DataError("DataFrame is empty after removing missing values")
        
        assert "empty" in str(error).lower()


class TestConstraintError:
    """Test suite for ConstraintError exception.
    
    Tests constraint validation error scenarios.
    """
    
    def test_min_exceeds_max_error(self):
        """Test ConstraintError when min > max."""
        error = ConstraintError(
            "min_bins (10) cannot exceed max_bins (5)"
        )
        
        assert "cannot exceed" in str(error)
        assert "10" in str(error)
        assert "5" in str(error)
    
    def test_invalid_range_error(self):
        """Test ConstraintError for invalid parameter range."""
        error = ConstraintError(
            "min_samples (1.5) must be in range (0, 1] for fractional "
            "specification or > 1 for absolute specification"
        )
        
        assert "range" in str(error)
        assert "1.5" in str(error)
    
    def test_already_resolved_error(self):
        """Test ConstraintError for double resolution."""
        error = ConstraintError(
            "Constraints have already been resolved. "
            "Create a new instance to resolve with different values."
        )
        
        assert "already been resolved" in str(error)
    
    def test_infeasible_constraints_error(self):
        """Test ConstraintError for infeasible constraint combination."""
        error = ConstraintError(
            "Constraints cannot be satisfied: requires 5 bins with "
            "minimum 300 samples each, but only 1000 samples available"
        )
        
        assert "cannot be satisfied" in str(error)


class TestFittingError:
    """Test suite for FittingError exception.
    
    Tests fitting process error scenarios.
    """
    
    def test_pava_convergence_error(self):
        """Test FittingError for PAVA convergence issues."""
        error = FittingError(
            "PAVA failed to converge after 1000 iterations"
        )
        
        assert "failed to converge" in str(error)
        assert "1000" in str(error)
    
    def test_monotonicity_violation_error(self):
        """Test FittingError for monotonicity violations."""
        error = FittingError(
            "PAVA failed to produce monotonic blocks"
        )
        
        assert "monotonic" in str(error)
    
    def test_merge_failure_error(self):
        """Test FittingError for merge process failure."""
        error = FittingError(
            "Merge process failed: unable to satisfy minimum samples constraint"
        )
        
        assert "Merge process failed" in str(error)
    
    def test_numerical_instability_error(self):
        """Test FittingError for numerical issues."""
        error = FittingError(
            "Numerical instability detected: variance calculation resulted in NaN"
        )
        
        assert "Numerical instability" in str(error)
        assert "NaN" in str(error)


class TestNotFittedError:
    """Test suite for NotFittedError exception.
    
    Tests errors when accessing unfitted model results.
    """
    
    def test_bins_before_fit_error(self):
        """Test NotFittedError when accessing bins before fitting."""
        error = NotFittedError(
            "This MonotonicBinner instance is not fitted yet. "
            "Call 'fit' before using this method."
        )
        
        assert "not fitted" in str(error)
        assert "Call 'fit'" in str(error)
    
    def test_transform_before_fit_error(self):
        """Test NotFittedError when transforming before fitting."""
        error = NotFittedError(
            "Cannot transform data: model has not been fitted. "
            "Call fit() first."
        )
        
        assert "Cannot transform" in str(error)
        assert "not been fitted" in str(error)
    
    def test_summary_before_fit_error(self):
        """Test NotFittedError when getting summary before fitting."""
        error = NotFittedError(
            "Summary not available: model must be fitted first"
        )
        
        assert "not available" in str(error)
        assert "fitted first" in str(error)


class TestBinningWarning:
    """Test suite for BinningWarning.
    
    Tests warning scenarios in the binning process.
    """
    
    def test_small_bin_warning(self):
        """Test warning for bins with few samples."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            warnings.warn(
                "Bin 3 has only 15 samples, which is below the recommended "
                "threshold of 30",
                BinningWarning
            )
            
            assert len(w) == 1
            assert issubclass(w[0].category, BinningWarning)
            assert "15 samples" in str(w[0].message)
    
    def test_constraint_relaxation_warning(self):
        """Test warning when constraints are relaxed."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            warnings.warn(
                "Could not satisfy min_samples=100 constraint. "
                "Relaxed to min_samples=80 to create valid bins.",
                BinningWarning
            )
            
            assert len(w) == 1
            assert issubclass(w[0].category, BinningWarning)
            assert "Relaxed" in str(w[0].message)
    
    def test_fallback_strategy_warning(self):
        """Test warning when falling back to alternative strategy."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            warnings.warn(
                "maximize_bins=True could not be satisfied. "
                "Falling back to standard merging strategy.",
                BinningWarning
            )
            
            assert len(w) == 1
            assert issubclass(w[0].category, BinningWarning)
            assert "Falling back" in str(w[0].message)


class TestExceptionUsagePatterns:
    """Test common usage patterns for exceptions."""
    
    def test_exception_chaining(self):
        """Test exception chaining for better debugging."""
        try:
            try:
                # Simulate lower-level error
                raise ValueError("Invalid value in data")
            except ValueError as e:
                # Chain with higher-level MOBPY error
                raise DataError(f"Data validation failed: {e}") from e
        except DataError as e:
            assert "Data validation failed" in str(e)
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)
    
    def test_exception_with_context(self):
        """Test exceptions with additional context information."""
        error = DataError(
            "Column 'age' validation failed:\n"
            "  - Contains 50 missing values\n"
            "  - Contains 10 infinite values\n"
            "  - Range: [-inf, inf]"
        )
        
        error_str = str(error)
        assert "missing values" in error_str
        assert "infinite values" in error_str
        assert "Range" in error_str
    
    def test_exception_error_codes(self):
        """Test that exceptions can carry error codes (future enhancement)."""
        # This could be enhanced in future versions
        error = ConstraintError("Invalid constraints")
        
        # Could add error_code attribute in future
        # assert hasattr(error, 'error_code')
        
        # For now, just verify basic functionality
        assert str(error) == "Invalid constraints"
    
    def test_warning_to_error_escalation(self):
        """Test pattern of escalating warnings to errors."""
        # First, issue warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            warnings.warn(
                "Constraints may be too restrictive",
                BinningWarning
            )
            
            assert len(w) == 1
        
        # Then, if condition persists, raise error
        with pytest.raises(ConstraintError):
            raise ConstraintError(
                "Constraints are infeasible after multiple attempts"
            )


class TestExceptionIntegration:
    """Integration tests for exception handling across modules."""
    
    def test_exception_propagation_in_pipeline(self):
        """Test that exceptions propagate correctly through the pipeline."""
        from MOBPY.binning.mob import MonotonicBinner
        import pandas as pd
        
        # Test with invalid data
        df = pd.DataFrame({
            'x': ['a', 'b', 'c'],  # Non-numeric
            'y': [1, 2, 3]
        })
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        
        # May raise ValueError or DataError depending on where validation happens
        with pytest.raises((DataError, ValueError)) as exc_info:
            binner.fit()
        
        # Should indicate the problem is with conversion/numeric data
        error_msg = str(exc_info.value).lower()
        assert "convert" in error_msg or "numeric" in error_msg or "float" in error_msg
    
    def test_exception_in_constraint_resolution(self):
        """Test exception during constraint resolution."""
        from MOBPY.core.constraints import BinningConstraints
        
        constraints = BinningConstraints(
            min_samples=200,  # Absolute
            max_samples=0.1   # 10% fractional
        )
        
        # After resolution, min > max, should raise
        with pytest.raises(ConstraintError) as exc_info:
            constraints.resolve(total_n=1000, total_pos=200)
        
        # Check that error message mentions the issue
        assert "exceeds" in str(exc_info.value) and "max_samples" in str(exc_info.value)
    
    def test_not_fitted_error_in_transform(self):
        """Test NotFittedError when using unfitted model."""
        from MOBPY.binning.mob import MonotonicBinner
        import pandas as pd
        
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 0, 1]})
        binner = MonotonicBinner(df=df, x='x', y='y')
        
        # Try to use before fitting
        with pytest.raises(NotFittedError) as exc_info:
            binner.transform([1, 2, 3])
        
        # Check error message mentions fitting
        error_msg = str(exc_info.value).lower()
        assert "fit" in error_msg and "before" in error_msg