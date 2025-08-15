"""Binning constraints with validation and resolution logic.

This module provides the BinningConstraints class which encapsulates all
user-defined constraints for the binning process. Constraints can be specified
as fractions (0-1) or absolute values and are resolved at fit time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import warnings

from MOBPY.exceptions import ConstraintError


@dataclass
class BinningConstraints:
    """Constraints for monotonic binning with automatic resolution.
    
    Constraints control the binning behavior by setting limits on:
    - Number of bins (min/max)
    - Samples per bin (min/max) 
    - Positives per bin (min, for binary targets)
    
    Fractional constraints (values in (0,1]) are resolved to absolute values
    based on the clean data partition at fit time.
    
    Args:
        max_bins: Maximum number of bins allowed. Must be >= 1.
        min_bins: Minimum number of bins to maintain. Must be >= 1.
        max_samples: Maximum samples per bin. If in (0,1], treated as fraction.
            If > 1, treated as absolute count. None means no upper limit.
        min_samples: Minimum samples per bin. If in (0,1], treated as fraction.
            If > 1, treated as absolute count. None defaults to 0.
        min_positives: Minimum positive samples per bin (binary targets only).
            If in (0,1], treated as fraction of total positives.
            If > 1, treated as absolute count. None defaults to 0.
        initial_pvalue: Initial p-value threshold for merge decisions.
            Higher values make merging more aggressive. Range: (0, 1].
        maximize_bins: If True, prioritize staying at/below max_bins.
            If False, prioritize staying at/above min_bins.
    
    Attributes:
        abs_max_samples: Resolved absolute maximum samples (after resolve()).
        abs_min_samples: Resolved absolute minimum samples (after resolve()).
        abs_min_positives: Resolved absolute minimum positives (after resolve()).
        
    Raises:
        ConstraintError: If constraints are invalid or contradictory.
        
    Examples:
        >>> # Use fractions for adaptive constraints
        >>> constraints = BinningConstraints(
        ...     max_bins=6,
        ...     min_samples=0.05,  # Each bin gets at least 5% of data
        ...     min_positives=0.01  # Each bin gets at least 1% of positives
        ... )
        
        >>> # Use absolute values for fixed constraints  
        >>> constraints = BinningConstraints(
        ...     max_bins=5,
        ...     min_samples=100,  # Each bin needs at least 100 samples
        ...     max_samples=1000  # No bin can exceed 1000 samples
        ... )
    """
    
    max_bins: int = 6
    min_bins: int = 4
    max_samples: Optional[float] = None
    min_samples: Optional[float] = None
    min_positives: Optional[float] = None
    initial_pvalue: float = 0.4
    maximize_bins: bool = True
    
    # Resolved absolute values (populated by resolve())
    abs_max_samples: Optional[int] = field(default=None, init=False)
    abs_min_samples: int = field(default=0, init=False)
    abs_min_positives: int = field(default=0, init=False)
    _resolved: bool = field(default=False, init=False)
    
    def __post_init__(self) -> None:
        """Validate constraints immediately after initialization.
        
        Raises:
            ConstraintError: If initial constraints are invalid.
        """
        # Validate bin counts
        if not isinstance(self.max_bins, int) or self.max_bins < 1:
            raise ConstraintError(f"max_bins must be an integer >= 1, got {self.max_bins}")
        
        if not isinstance(self.min_bins, int) or self.min_bins < 1:
            raise ConstraintError(f"min_bins must be an integer >= 1, got {self.min_bins}")
        
        # Special case: when maximizing, min cannot exceed max
        if self.maximize_bins and self.min_bins > self.max_bins:
            raise ConstraintError(
                f"min_bins ({self.min_bins}) cannot exceed max_bins ({self.max_bins}) "
                f"when maximize_bins=True"
            )
        
        # Validate p-value
        if not 0 < self.initial_pvalue <= 1:
            raise ConstraintError(
                f"initial_pvalue must be in (0, 1], got {self.initial_pvalue}"
            )
        
        # Validate sample constraints
        if self.max_samples is not None and self.max_samples <= 0:
            raise ConstraintError(f"max_samples must be positive, got {self.max_samples}")
        
        if self.min_samples is not None and self.min_samples < 0:
            raise ConstraintError(f"min_samples cannot be negative, got {self.min_samples}")
        
        if self.min_positives is not None and self.min_positives < 0:
            raise ConstraintError(f"min_positives cannot be negative, got {self.min_positives}")
    
    def resolve(self, *, total_n: int, total_pos: int = 0) -> None:
        """Resolve fractional constraints to absolute values.
        
        This method converts any fractional constraints (0 < value <= 1) to
        absolute counts based on the actual data size. Must be called before
        using the constraints in the binning process.
        
        Args:
            total_n: Total number of samples in the clean partition.
            total_pos: Total number of positive samples (for binary targets).
                Ignored for non-binary targets.
        
        Raises:
            ConstraintError: If resolved constraints are contradictory
                (e.g., min_samples > max_samples).
            ValueError: If total_n or total_pos are negative.
            
        Note:
            This method modifies the constraint object in place by setting
            the abs_* attributes.
        """
        if total_n < 0:
            raise ValueError(f"total_n must be non-negative, got {total_n}")
        if total_pos < 0:
            raise ValueError(f"total_pos must be non-negative, got {total_pos}")
        
        # Resolve max_samples
        if self.max_samples is None:
            self.abs_max_samples = None
        else:
            if 0 < self.max_samples <= 1:
                # Fraction of total
                self.abs_max_samples = max(1, int(self.max_samples * total_n))
            else:
                # Absolute value
                self.abs_max_samples = int(self.max_samples)
            
            # Cap at total available
            if total_n > 0:
                self.abs_max_samples = min(self.abs_max_samples, total_n)
        
        # Resolve min_samples
        if self.min_samples is None:
            self.abs_min_samples = 0
        else:
            if 0 < self.min_samples <= 1:
                # Fraction of total
                self.abs_min_samples = max(0, int(self.min_samples * total_n))
            else:
                # Absolute value
                self.abs_min_samples = max(0, int(self.min_samples))
            
            # Cap at total available
            if total_n > 0:
                self.abs_min_samples = min(self.abs_min_samples, total_n)
        
        # Cross-validate min vs max samples
        if self.abs_max_samples is not None:
            if self.abs_min_samples > self.abs_max_samples:
                raise ConstraintError(
                    f"min_samples ({self.abs_min_samples}) exceeds "
                    f"max_samples ({self.abs_max_samples}) after resolution"
                )
        
        # Resolve min_positives (binary targets only)
        if self.min_positives is None:
            self.abs_min_positives = 0
        else:
            if 0 < self.min_positives <= 1:
                # Fraction of total positives
                self.abs_min_positives = max(0, int(self.min_positives * total_pos))
            else:
                # Absolute value
                self.abs_min_positives = max(0, int(self.min_positives))
            
            # Cap at total available
            if total_pos > 0:
                self.abs_min_positives = min(self.abs_min_positives, total_pos)
        
        # Sanity check: can we create at least min_bins with these constraints?
        if self.abs_min_samples > 0 and total_n > 0:
            max_possible_bins = total_n // self.abs_min_samples
            if max_possible_bins < self.min_bins:
                warnings.warn(
                    f"With min_samples={self.abs_min_samples}, only "
                    f"{max_possible_bins} bins are possible, but min_bins={self.min_bins}. "
                    f"Some constraints may not be satisfied.",
                    UserWarning
                )
        
        self._resolved = True
    
    def is_resolved(self) -> bool:
        """Check if constraints have been resolved to absolute values.
        
        Returns:
            bool: True if resolve() has been called successfully.
        """
        return self._resolved
    
    def copy(self) -> "BinningConstraints":
        """Create a deep copy of the constraints.
        
        Returns:
            BinningConstraints: A new instance with the same values.
            
        Note:
            The copy will not be resolved even if the original was.
        """
        return BinningConstraints(
            max_bins=self.max_bins,
            min_bins=self.min_bins,
            max_samples=self.max_samples,
            min_samples=self.min_samples,
            min_positives=self.min_positives,
            initial_pvalue=self.initial_pvalue,
            maximize_bins=self.maximize_bins
        )
    
    def __repr__(self) -> str:
        """String representation for debugging.
        
        Returns:
            str: Detailed representation showing all constraint values.
        """
        resolved_info = ""
        if self._resolved:
            resolved_info = (
                f", resolved=(max_samples={self.abs_max_samples}, "
                f"min_samples={self.abs_min_samples}, "
                f"min_positives={self.abs_min_positives})"
            )
        
        return (
            f"BinningConstraints(max_bins={self.max_bins}, min_bins={self.min_bins}, "
            f"max_samples={self.max_samples}, min_samples={self.min_samples}, "
            f"min_positives={self.min_positives}, initial_pvalue={self.initial_pvalue}, "
            f"maximize_bins={self.maximize_bins}{resolved_info})"
        )