from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_numeric_series


@dataclass
class _Block:
    """Internal PAVA block with sufficient statistics.

    Attributes:
        left: Left (inclusive) edge.
        right: Right (exclusive) edge.
        n: Count of observations.
        sum: Sum of y.
        sum2: Sum of y**2 (for variance).
        ymin: Minimum y.
        ymax: Maximum y.
    """
    left: float
    right: float
    n: int
    sum: float
    sum2: float
    ymin: float
    ymax: float

    @property
    def mean(self) -> float:
        return 0.0 if self.n == 0 else self.sum / self.n

    @property
    def var(self) -> float:
        """Unbiased sample variance from aggregated stats."""
        if self.n <= 1:
            return 0.0
        return max((self.sum2 - (self.sum ** 2) / self.n) / (self.n - 1), 0.0)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))

    def as_dict(self) -> dict:
        return dict(left=self.left, right=self.right, n=self.n, sum=self.sum, sum2=self.sum2, ymin=self.ymin, ymax=self.ymax)

    def merge_with(self, other: "_Block") -> "_Block":
        """Return a new block equal to the union of self and other (adjacent)."""
        n = self.n + other.n
        s = self.sum + other.sum
        s2 = self.sum2 + other.sum2
        ymin = min(self.ymin, other.ymin)
        ymax = max(self.ymax, other.ymax)
        return _Block(left=self.left, right=other.right, n=n, sum=s, sum2=s2, ymin=ymin, ymax=ymax)


VALID_SORT_KINDS = (None, "quicksort", "mergesort", "heapsort", "stable")

class PAVA:
    """Pool-Adjacent Violators Algorithm for monotone means on grouped x.

    Implementation notes:
      * We sort by `x`, group to atomic blocks (one per unique x),
        then pool adjacent blocks while monotonicity is violated.
      * The structure is logically equivalent to your original
        double-linked list approach, but we use a simple **stack**:
        push each block; while the top two violate, pop & merge, push back.

    Only `metric="mean"` is supported here (MOB special case).

    Args:
        df: Input DataFrame.
        x: Column name (feature being binned).
        y: Column name (target or metric column).
        metric: Must be `"mean"`.
        sign: `"+"`, `"-"`, or `"auto"`.
        strict: If True, equal-means plateaus are merged (strict monotone).
        sort_kind: Sorting algorithm used before groupby (see pandas `sort_values(kind=...)`).

    Raises:
        KeyError: If required columns missing.
        TypeError/ValueError: If `y` is not numeric/finite on clean rows.
        ValueError: If no clean rows exist after dropping missing x or y.
    """

    def __init__(
        self,
        *,
        df: pd.DataFrame,
        x: str,
        y: str,
        metric: str = "mean",
        sign: str = "auto",
        strict: bool = True,
        sort_kind: Optional[str] = "quicksort",
    ):
        if metric != "mean":
            raise ValueError("PAVA currently supports only metric='mean'.")
        self.df = df
        self.x = x
        self.y = y
        self.metric = metric
        self.sign = sign
        self.strict = strict

        if sort_kind not in VALID_SORT_KINDS:
            raise ValueError(f"sort_kind must be one of {VALID_SORT_KINDS}, got {sort_kind!r}")
        self.sort_kind = sort_kind

        self.blocks_: List[_Block] = []
        self.groups_: Optional[pd.DataFrame] = None
        self.resolved_sign_: Literal["+", "-"] | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fit(self) -> "PAVA":
        """Run PAVA on grouped/ordered data and cache blocks."""

        # Validate columns
        if self.x not in self.df.columns or self.y not in self.df.columns:
            missing = [c for c in (self.x, self.y) if c not in self.df.columns]
            raise KeyError(f"Missing columns in df: {missing}")

        # PAVA runs on **clean** rows only
        sub = self.df[[self.x, self.y]].dropna()
        if sub.empty:
            raise ValueError("No rows with non-missing x and y for PAVA.")

        # y must be numeric and finite
        ensure_numeric_series(sub[self.y], self.y)

        # ---- sort first with desired algorithm, then groupby(sort=False) ----
        if self.sort_kind is None:
            sub_sorted = self.df[[self.x, self.y]].dropna().sort_values(
                by=self.x, na_position="last"
            )
        else:
            sub_sorted = self.df[[self.x, self.y]].dropna().sort_values(
                by=self.x, kind=self.sort_kind, na_position="last"
            )

        gb = sub_sorted.groupby(self.x, sort=False)[self.y]

        # Aggregate per unique x
        counts = gb.count().to_numpy(dtype=np.int64)
        sums = gb.sum().to_numpy(dtype=float)
        # Sum of squares for pooled variance
        sums2 = gb.apply(lambda s: np.square(s.to_numpy(dtype=float)).sum()).to_numpy(dtype=float)
        mins = gb.min().to_numpy(dtype=float)
        maxs = gb.max().to_numpy(dtype=float)
        xs = gb.mean().index.to_numpy(dtype=float)  # index = sorted unique x

        # groups_ mirrors your original CSD table (columns renamed per current code)
        self.groups_ = pd.DataFrame(
            {"x": xs, "count": counts, "sum": sums, "sum2": sums2, "ymin": mins, "ymax": maxs}
        )

        # Resolve monotone direction
        if self.sign in {"+", "-"}:
            resolved_sign = self.sign
        else:
            # Infer direction from correlation of x vs. group means,
            # but avoid corrcoef when one vector is constant (std == 0)
            xs = self.groups_["x"].to_numpy(dtype=float)
            means_by_x = (self.groups_["sum"] / self.groups_["count"]).to_numpy(dtype=float)

            if len(xs) <= 1 or np.std(xs) == 0.0 or np.std(means_by_x) == 0.0:
                corr = 1.0  # treat as non-decreasing (also avoids warnings)
            else:
                corr = float(np.corrcoef(xs, means_by_x)[0, 1])

            resolved_sign = "+" if corr >= 0 else "-"
        self.resolved_sign_ = resolved_sign

        # One block per unique x (convert right edge to next x)
        blocks: List[_Block] = []
        for _, row in self.groups_.iterrows():
            blocks.append(
                _Block(
                    left=float(row["x"]),
                    right=float(row["x"]),  # temp; set below
                    n=int(row["count"]),
                    sum=float(row["sum"]),
                    sum2=float(row["sum2"]),
                    ymin=float(row["ymin"]),
                    ymax=float(row["ymax"]),
                )
            )

        # Right edge = next unique x; last is +inf for clean searchsorted
        xs = self.groups_["x"].to_numpy(dtype=float)
        for i in range(len(blocks) - 1):
            blocks[i].right = xs[i + 1]
        blocks[-1].right = float("inf")

        # Stack-based pooling (equiv to DLL merging from your original design)
        stack: List[_Block] = []
        for b in blocks:
            stack.append(b)
            # While the top two violate monotonicity → merge them
            while len(stack) >= 2:
                b2 = stack[-1]
                b1 = stack[-2]
                if self._violates(b1, b2, resolved_sign):
                    merged = b1.merge_with(b2)
                    stack.pop()
                    stack.pop()
                    stack.append(merged)
                else:
                    # If strict, collapse plateaus (equal means) as well
                    if self.strict and abs(b2.mean - b1.mean) <= 1e-12:
                        merged = b1.merge_with(b2)
                        stack.pop()
                        stack.pop()
                        stack.append(merged)
                    else:
                        break

        self.blocks_ = stack
        return self

    def export_blocks(self, as_dict: bool = True) -> List[dict] | List[Tuple]:
        """Export fitted blocks as safe primitives (dicts or tuples)."""
        if as_dict:
            return [b.as_dict() for b in self.blocks_]
        return [(b.left, b.right, b.n, b.sum, b.sum2, b.ymin, b.ymax) for b in self.blocks_]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _violates(b1: _Block, b2: _Block, sign: Literal["+", "-"]) -> bool:
        """Return True if the pair (b1,b2) violates monotonicity for `sign`."""
        if sign == "+":
            return b2.mean < b1.mean - 1e-12
        else:
            return b2.mean > b1.mean + 1e-12
