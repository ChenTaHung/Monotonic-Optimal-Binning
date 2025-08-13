from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BinningConstraints:
    """Constraints for binning; fractionals are resolved at fit-time.

    Users can specify min/max samples and min positives as *fractions* or as
    absolute counts. Fractions are resolved against the **clean** partition
    totals inside `resolve()`.

    Args:
        max_bins: Maximum number of bins for the clean portion.
        min_bins: Minimum number of bins for the clean portion.
        max_samples: If in (0,1], interpreted as fraction of total rows; else absolute.
        min_samples: If in (0,1], interpreted as fraction of total rows; else absolute.
        min_positives: For binary targets only. If in (0,1], fraction of total positives; else absolute.
        initial_pvalue: Starting merge threshold (two-sample test). Caller may lower later.
        maximize_bins: If True, emphasize staying ≤ max_bins. If False, enforce ≥ min_bins (unless impossible).

    Resolved (computed in `resolve`):
        abs_max_samples: Absolute upper bound on bin size (or None if unbounded).
        abs_min_samples: Absolute lower bound on bin size (default 0).
        abs_min_positives: Absolute lower bound on positives per bin (default 0).
    """

    max_bins: int = 6
    min_bins: int = 4
    max_samples: Optional[float] = None
    min_samples: Optional[float] = None
    min_positives: Optional[float] = None
    initial_pvalue: float = 0.4
    maximize_bins: bool = True

    # resolved/absolute values (populated by resolve)
    abs_max_samples: Optional[int] = None
    abs_min_samples: int = 0
    abs_min_positives: int = 0

    _resolved: bool = False

    def resolve(self, *, total_n: int, total_pos: int = 0) -> None:
        """Resolve fractional constraints to absolutes and validate.

        This must be called on the **clean** partition (after excluding missing/specials).

        Raises:
            ValueError: if `min_samples` exceeds `max_samples` after resolution.
        """
        if total_n < 0:
            raise ValueError("total_n must be nonnegative")
        if total_pos < 0:
            raise ValueError("total_pos must be nonnegative")

        # --- max_samples
        if self.max_samples is None:
            self.abs_max_samples = None
        else:
            self.abs_max_samples = (
                int(self.max_samples * total_n) if 0 < self.max_samples <= 1 else int(self.max_samples)
            )
            if self.abs_max_samples < 1 and total_n > 0:
                self.abs_max_samples = 1
            if self.abs_max_samples is not None and total_n > 0:
                self.abs_max_samples = min(self.abs_max_samples, total_n)

        # --- min_samples
        if self.min_samples is None:
            self.abs_min_samples = 0
        else:
            self.abs_min_samples = (
                int(self.min_samples * total_n) if 0 < self.min_samples <= 1 else int(self.min_samples)
            )
            if self.abs_min_samples < 0:
                self.abs_min_samples = 0
            if total_n > 0:
                self.abs_min_samples = min(self.abs_min_samples, total_n)

        # --- cross-check: min cannot exceed max
        if self.abs_max_samples is not None:
            if self.abs_min_samples > self.abs_max_samples:
                raise ValueError("`min_samples` cannot exceed `max_samples` after resolution.")

        # --- min positives (binary)
        if self.min_positives is None:
            self.abs_min_positives = 0
        else:
            self.abs_min_positives = (
                int(self.min_positives * total_pos) if 0 < self.min_positives <= 1 else int(self.min_positives)
            )
            if self.abs_min_positives < 0:
                self.abs_min_positives = 0
            if total_pos > 0:
                self.abs_min_positives = min(self.abs_min_positives, total_pos)

        # sanity on bins
        if not (isinstance(self.max_bins, int) and self.max_bins >= 1):
            raise ValueError("max_bins must be an integer >= 1")
        if not (isinstance(self.min_bins, int) and self.min_bins >= 1):
            raise ValueError("min_bins must be an integer >= 1")
        if self.maximize_bins and self.min_bins > self.max_bins:
            raise ValueError("min_bins cannot exceed max_bins when maximize_bins=True")

        self._resolved = True
