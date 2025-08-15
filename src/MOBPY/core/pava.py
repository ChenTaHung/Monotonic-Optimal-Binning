from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_numeric_series


@dataclass
class _Block:
    """Internal block used by PAVA.

    Stores sufficient statistics so merges are O(1).

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

    # --- derived stats ---

    @property
    def mean(self) -> float:
        return 0.0 if self.n == 0 else self.sum / self.n

    @property
    def var(self) -> float:
        if self.n <= 1:
            return 0.0
        # unbiased sample variance from aggregated stats
        return max((self.sum2 - (self.sum ** 2) / self.n) / (self.n - 1), 0.0)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))

    def as_dict(self) -> dict:
        return dict(left=self.left, right=self.right, n=self.n, sum=self.sum, sum2=self.sum2, ymin=self.ymin, ymax=self.ymax)

    # --- merge ---

    def merge_with(self, other: "_Block") -> "_Block":
        """Return a new block equal to the union of self and other."""
        n = self.n + other.n
        s = self.sum + other.sum
        s2 = self.sum2 + other.sum2
        ymin = min(self.ymin, other.ymin)
        ymax = max(self.ymax, other.ymax)
        return _Block(left=self.left, right=other.right, n=n, sum=s, sum2=s2, ymin=ymin, ymax=ymax)


VALID_SORT_KINDS = (None, "quicksort", "mergesort", "heapsort", "stable")


class PAVA:
    """Pool-Adjacent-Violators Algorithm for isotonic regression on grouped x.

    Only `metric="mean"` is supported (MOB setting). We:
      1) group by `x` (sorted),
      2) build one block per unique x with count/sum/sum2/min/max,
      3) pool adjacent blocks until monotonicity (by mean) holds.

    After `fit()`:
      - `self.blocks_` holds the *monotone* blocks,
      - `self.groups_` includes cumulative columns:
          * `cum_count`, `cum_sum`, `cum_mean` — to draw CSD-style curves.

    Args:
        df: Input DataFrame.
        x: Column name (feature being binned).
        y: Column name (target or metric column).
        metric: Must be "mean".
        sign: "+", "-", or "auto".
        strict: If True, equal-mean plateaus are merged.
        sort_kind: pandas sort_kind for sorting by `x`.
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
            # Future work: support other metrics like median; keeping mean only keeps code simple & efficient.
            raise ValueError("Only metric='mean' is supported in this version.")
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

    # --------------------------- public API --------------------------- #

    def fit(self) -> "PAVA":
        """Run PAVA on the grouped/ordered data and compute cumulative means.

        Raises:
            KeyError: If x or y missing.
            TypeError/ValueError: For non-numeric or non-finite y on clean rows.
            ValueError: If no rows after dropping rows with missing x or y.
        """
        if self.x not in self.df.columns or self.y not in self.df.columns:
            missing = [c for c in (self.x, self.y) if c not in self.df.columns]
            raise KeyError(f"Missing columns in df: {missing}")

        # Clean rows only for PAVA
        sub = self.df[[self.x, self.y]].dropna()
        if sub.empty:
            raise ValueError("No rows with non-missing x and y for PAVA.")

        ensure_numeric_series(sub[self.y], self.y)

        # Sort (stable order) then group (no re-sort) for consistent stats/curves
        if self.sort_kind is None:
            sub_sorted = sub.sort_values(by=self.x, na_position="last")
        else:
            sub_sorted = sub.sort_values(by=self.x, kind=self.sort_kind, na_position="last")
        gb = sub_sorted.groupby(self.x, sort=False)[self.y]

        # Aggregate per unique x
        counts = gb.count().to_numpy(dtype=np.int64)
        sums = gb.sum().to_numpy(dtype=float)
        sums2 = gb.apply(lambda s: np.square(s.to_numpy(dtype=float)).sum()).to_numpy(dtype=float)
        mins = gb.min().to_numpy(dtype=float)
        maxs = gb.max().to_numpy(dtype=float)
        xs = gb.mean().index.to_numpy(dtype=float)  # sorted unique x

        # Build table of groups (+ cumulative columns for CSD-like plots)
        groups = pd.DataFrame(
            {"x": xs, "count": counts, "sum": sums, "sum2": sums2, "ymin": mins, "ymax": maxs}
        )
        groups["cum_count"] = groups["count"].cumsum().astype(float)
        groups["cum_sum"] = groups["sum"].cumsum().astype(float)
        groups["cum_mean"] = np.divide(
            groups["cum_sum"].to_numpy(),
            groups["cum_count"].to_numpy(),
            out=np.zeros_like(groups["cum_sum"].to_numpy(), dtype=float),
            where=groups["cum_count"].to_numpy() > 0,
        )
        self.groups_ = groups

        # Decide monotone direction
        means_by_x = groups["sum"] / groups["count"]
        if self.sign in {"+", "-"}:
            resolved_sign = self.sign
        else:
            corr = np.corrcoef(groups["x"], means_by_x)[0, 1] if len(groups) > 1 else 1.0
            resolved_sign = "+" if (corr >= 0 or np.isnan(corr)) else "-"
        self.resolved_sign_ = resolved_sign

        # Initial blocks, right edge set to next unique x
        blocks: List[_Block] = []
        for _, row in groups.iterrows():
            blk = _Block(
                left=float(row["x"]),
                right=float(row["x"]),  # placeholder; adjusted below
                n=int(row["count"]),
                sum=float(row["sum"]),
                sum2=float(row["sum2"]),
                ymin=float(row["ymin"]),
                ymax=float(row["ymax"]),
            )
            blocks.append(blk)

        xs = groups["x"].to_numpy(dtype=float)
        for i in range(len(blocks) - 1):
            blocks[i].right = xs[i + 1]
        # IMPORTANT: ensure global coverage at the PAVA level itself
        blocks[0].left = float("-inf")
        blocks[-1].right = float("inf")

        # PAVA pooling with a stack
        stack: List[_Block] = []
        for b in blocks:
            stack.append(b)
            while len(stack) >= 2:
                b2 = stack[-1]
                b1 = stack[-2]
                if self._violates(b1, b2, resolved_sign):
                    merged = b1.merge_with(b2)
                    stack.pop(); stack.pop()
                    stack.append(merged)
                else:
                    if self.strict and abs(b2.mean - b1.mean) <= 1e-12:
                        merged = b1.merge_with(b2)
                        stack.pop(); stack.pop()
                        stack.append(merged)
                    else:
                        break

        self.blocks_ = stack
        return self

    def export_blocks(self, as_dict: bool = True) -> List[dict] | List[Tuple]:
        """Export fitted blocks (safe copy)."""
        if as_dict:
            return [b.as_dict() for b in self.blocks_]
        return [(b.left, b.right, b.n, b.sum, b.sum2, b.ymin, b.ymax) for b in self.blocks_]

    # --------------------------- helpers --------------------------- #

    @staticmethod
    def _violates(b1: _Block, b2: _Block, sign: Literal["+", "-"]) -> bool:
        """Return True if (b1, b2) violates monotonicity for given sign."""
        if sign == "+":
            return b2.mean < b1.mean - 1e-12
        else:
            return b2.mean > b1.mean + 1e-12
