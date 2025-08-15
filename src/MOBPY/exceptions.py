"""Custom exceptions for MOBPY.

This module defines specific exception types for different error scenarios
in the binning pipeline, making debugging easier for users.
"""


class MOBPYError(Exception):
    """Base exception for all MOBPY-specific errors.
    
    All custom exceptions inherit from this base class, allowing users to
    catch all MOBPY errors with a single except clause if desired.
    """
    pass


class DataError(MOBPYError):
    """Raised when input data has issues.
    
    Examples:
        - Missing required columns
        - Non-numeric target when numeric is expected
        - Empty DataFrame after cleaning
    """
    pass


class ConstraintError(MOBPYError):
    """Raised when constraints are invalid or cannot be satisfied.
    
    Examples:
        - min_samples > max_samples
        - Impossible constraint combinations
        - Constraints that would result in zero bins
    """
    pass


class FittingError(MOBPYError):
    """Raised when the fitting process fails.
    
    Examples:
        - PAVA convergence issues
        - Merge process produces invalid bins
        - Numerical instability
    """
    pass


class NotFittedError(MOBPYError):
    """Raised when trying to access results before fitting.
    
    Examples:
        - Calling bins_() before fit()
        - Calling transform() on unfitted model
    """
    pass


class BinningWarning(UserWarning):
    """Warning for non-critical binning issues.
    
    Examples:
        - Target constraints could not be fully satisfied
        - Falling back to alternative strategies
    """
    pass