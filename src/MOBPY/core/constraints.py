# src/MOBPY/core/constraints.py
"""
Binning constraints and p-value scheduling.

This module defines a small, well-documented container for the binning
constraints used by the adjacent-merge stage. It supports both **fractional**
(percentage) and **absolute** thresholds, and provides a `resolve(...)` method
to translate fractional inputs into absolute counts based on the data size.

Key concepts
------------
- max_bins / min_bins:
    Upper / lower bounds on the number of bins after merging.

- max_samples / min_samples:
    Constraints on bin sizes. Can be absolute integers (e.g., 1000) or
    fractions in (0,1] meaning a proportion of the total clean sample size.

- min_positives:
    (Binary targets only) Minimum positives per bin. Like min_samples, can be
    absolute or fractional (of total positives in the clean subset).

- initial_pvalue:
    Starting p-value threshold for merge decisions. If no adjacent pair exceeds
    the threshold while the bin-count constraint is still violated, a scheduler
    reduces the threshold: step by -0.05 down to 0.01, then multiply by 0.1
    on each further update, with a floor at 1e-8.

- maximize_bins:
    If True: merge until #bins <= max_bins (then enforce min_samples pass).
    If False: merge until #bins <= min_bins (alternate workflow).

Usage
-----
>>> cons = BinningConstraints(max_bins=6, min_bins=4,
...                           max_samples=0.4, min_samples=0.05, min_positives=0.05,
...                           initial_pvalue=0.4, maximize_bins=True)
>>> cons.resolve(total_n=1000, total_pos=200)
>>> cons.abs_min_samples
50
>>> cons.abs_min_positives
10
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


Number = Union[int, float]


def _is_fraction(x: Optional[Number]) -> bool:
    """Return True if x is a valid fraction in (0, 1]."""
    return x is not None and isinstance(x, (int, float)) and (0.0 < float(x) <= 1.0)


def _is_nonneg_int(x: Optional[Number]) -> bool:
    """Return True if x is a valid non-negative integer (as int-like)."""
    return x is not None and int(x) == x and int(x) >= 0


@dataclass
class BinningConstraints:
    """
    Container for binning constraints and p-value schedule.

    Args:
        max_bins: Upper bound on the number of bins when `maximize_bins=True`.
        min_bins: Lower bound on the number of bins when `maximize_bins=False`.
        max_samples: Maximum bin size (absolute int or fraction of total_n).
        min_samples: Minimum bin size (absolute int or fraction of total_n).
        min_positives: Minimum positives per bin for binary targets (absolute
            int or fraction of total_pos). Use 0/None to disable.
        initial_pvalue: Initial merge threshold in (0, 1]. See schedule in
            `next_threshold`.
        maximize_bins: Control the merge regime (see module docstring).

    Notes:
        Call `resolve(total_n, total_pos)` before using these constraints in
        the merger to translate fractional thresholds into absolute integers.
    """

    # --- bin-count constraints ---
    max_bins: int = 6
    min_bins: int = 4

    # --- bin-size constraints (can be int or fraction in (0,1]) ---
    max_samples: Optional[Number] = 0.4
    min_samples: Optional[Number] = 0.05

    # --- binary-only constraint (can be int or fraction in (0,1]) ---
    min_positives: Optional[Number] = 0.0

    # --- merge schedule ---
    initial_pvalue: float = 0.4
    maximize_bins: bool = True

    # --- resolved (absolute) thresholds; populated by `resolve` ---
    abs_max_samples: Optional[int] = field(default=None, init=False)
    abs_min_samples: int = field(default=0, init=False)
    abs_min_positives: int = field(default=0, init=False)

    # --- status ---
    _resolved: bool = field(default=False, init=False)

    # ------------------------------- Validation ------------------------------- #

    def resolve(self, *, total_n: int, total_pos: int = 0) -> None:
        """Resolve fractional thresholds into absolute counts.

        Args:
            total_n: Total sample size of the **clean** subset used for PAVA.
            total_pos: Total positives (sum of y) in the clean subset; only
                relevant when `min_positives` is fractional.

        Raises:
            ValueError: If inputs are invalid or constraints are inconsistent.
        """
        # Basic sanity checks
        if not isinstance(total_n, int) or total_n <= 0:
            raise ValueError(f"`total_n` must be a positive integer; got {total_n!r}.")
        if not isinstance(total_pos, int) or total_pos < 0:
            raise ValueError(f"`total_pos` must be a non-negative integer; got {total_pos!r}.")

        # Validate bin-count bounds
        if not isinstance(self.max_bins, int) or self.max_bins <= 0:
            raise ValueError(f"`max_bins` must be a positive integer; got {self.max_bins!r}.")
        if not isinstance(self.min_bins, int) or self.min_bins <= 0:
            raise ValueError(f"`min_bins` must be a positive integer; got {self.min_bins!r}.")

        if self.maximize_bins:
            if self.min_bins > self.max_bins:
                raise ValueError("`min_bins` cannot exceed `max_bins` when maximize_bins=True.")
        # When maximize_bins=False we keep >= min_bins; still ensure min_bins sensible
        # (max_bins is unused in that regime but we keep it validated above.)

        # Validate initial p-value
        if not (0.0 < float(self.initial_pvalue) <= 1.0):
            raise ValueError("`initial_pvalue` must be in (0, 1].")

        # Resolve min_samples
        if _is_fraction(self.min_samples):
            self.abs_min_samples = max(int(round(float(self.min_samples) * total_n)), 0)
        elif _is_nonneg_int(self.min_samples):
            self.abs_min_samples = int(self.min_samples)  # already absolute
        elif self.min_samples is None:
            self.abs_min_samples = 0
        else:
            raise ValueError("`min_samples` must be None, a non-negative int, or a fraction in (0,1].")

        # Resolve max_samples (None means no upper cap)
        if _is_fraction(self.max_samples):
            self.abs_max_samples = max(int(round(float(self.max_samples) * total_n)), 0)
        elif _is_nonneg_int(self.max_samples):
            v = int(self.max_samples)
            self.abs_max_samples = v if v > 0 else None  # treat 0 as "no cap"
        elif self.max_samples is None:
            self.abs_max_samples = None
        else:
            raise ValueError("`max_samples` must be None, a non-negative int, or a fraction in (0,1].")

        # Resolve min_positives (binary only). 0/None disables.
        if _is_fraction(self.min_positives):
            self.abs_min_positives = max(int(round(float(self.min_positives) * total_pos)), 0)
        elif _is_nonneg_int(self.min_positives):
            self.abs_min_positives = int(self.min_positives)
        elif self.min_positives is None:
            self.abs_min_positives = 0
        else:
            raise ValueError("`min_positives` must be None, a non-negative int, or a fraction in (0,1].")

        # Final consistency checks
        if self.abs_max_samples is not None and self.abs_min_samples > self.abs_max_samples:
            raise ValueError("`min_samples` cannot exceed `max_samples` after resolution.")

        self._resolved = True

    @property
    def resolved(self) -> bool:
        """Whether `resolve` has been called successfully."""
        return self._resolved

    # --------------------------- P-value scheduling --------------------------- #

    @staticmethod
    def next_threshold(current: float) -> float:
        """Return the next (lower) p-value threshold per project schedule.

        Schedule:
            - If current > 0.051: subtract 0.05
            - Else if current > 0.01: snap to 0.01
            - Else: multiply by 0.1 (down to a practical floor of 1e-8)
        """
        c = float(current)
        if c > 0.051:
            return c - 0.05
        if c > 0.01:
            return 0.01
        # multiplicative decay below 0.01
        return max(c * 0.1, 1e-8)

    # ------------------------------ Convenience ------------------------------ #

    def to_dict(self) -> dict:
        """Serialize constraints, including resolved absolute thresholds."""
        return {
            "max_bins": self.max_bins,
            "min_bins": self.min_bins,
            "max_samples": self.max_samples,
            "min_samples": self.min_samples,
            "min_positives": self.min_positives,
            "initial_pvalue": self.initial_pvalue,
            "maximize_bins": self.maximize_bins,
            "abs_max_samples": self.abs_max_samples,
            "abs_min_samples": self.abs_min_samples,
            "abs_min_positives": self.abs_min_positives,
            "resolved": self._resolved,
        }
