from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_numeric_series


@dataclass
class _Block:
    """Internal block used by PAVA (mean-only).

    Stores sufficient statistics so merges are O(1).

    Attributes:
        left: Left (inclusive) edge in x-domain.
        right: Right (exclusive) edge in x-domain.
        n: Count of observations.
        sum: Sum of y.
        sum2: Sum of y**2 (for variance).
        ymin: Minimum y in the block.
        ymax: Maximum y in the block.
    """
    left: float
    right: float
    n: int
    sum: float
    sum2: float
    ymin: float
    ymax: float

    # ---- derived stats ----------------------------------------------------- #

    @property
    def mean(self) -> float:
        return 0.0 if self.n == 0 else self.sum / self.n

    @property
    def var(self) -> float:
        """Unbiased sample variance from aggregated stats (>= 0)."""
        if self.n <= 1:
            return 0.0
        return max((self.sum2 - (self.sum ** 2) / self.n) / (self.n - 1), 0.0)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.var))

    def as_dict(self) -> dict:
        """Safe export: plain-Python dict (no references)."""
        return dict(
            left=float(self.left), right=float(self.right),
            n=int(self.n), sum=float(self.sum), sum2=float(self.sum2),
            ymin=float(self.ymin), ymax=float(self.ymax)
        )

    # ---- merge ------------------------------------------------------------- #

    def merge_with(self, other: "_Block") -> "_Block":
        """Return a new block equal to the union of self and other."""
        n = self.n + other.n
        s = self.sum + other.sum
        s2 = self.sum2 + other.sum2
        ymin = min(self.ymin, other.ymin)
        ymax = max(self.ymax, other.ymax)
        # IMPORTANT: Preserve the LEFT of the left block and RIGHT of the right block.
        # If the first block starts at -inf and/or the last at +inf, that persists.
        return _Block(left=self.left, right=other.right, n=n, sum=s, sum2=s2, ymin=ymin, ymax=ymax)


VALID_SORT_KINDS = (None, "quicksort", "mergesort", "heapsort", "stable")


class PAVA:
    """Pool-Adjacent-Violators Algorithm for isotonic regression (mean-only).

    Pipeline (vectorized + stack-based):

      1) Drop rows with missing x or y.
      2) Sort by x (configurable `sort_kind`) once, then groupby(sort=False).
      3) Build one initial block per unique x with sufficient stats.
         **We expand domain coverage so the first block starts at -inf
         and the last block ends at +inf**.
      4) Run the classic PAVA stack: push each block, while the last two violate
         monotonicity under the resolved sign, merge them.
      5) Record **history frames** after each structural change (push/merge).
         This enables animations / step-by-step visuals.

    Only `metric="mean"` is supported (median/quantiles are future work).

    Args:
        df: Input DataFrame.
        x: Column name of the feature to bin.
        y: Column name of the response/target.
        metric: Must be "mean".
        sign: "+", "-", or "auto" (infer sign from corr between x and group means).
        strict: If True, merge equal-mean plateaus to enforce *strict* monotone.
        sort_kind: pandas sort kind used by `DataFrame.sort_values`.
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
            # Mean-only implementation; median/quantiles are future work.
            raise ValueError("Only metric='mean' is supported at this time.")

        if sort_kind not in VALID_SORT_KINDS:
            raise ValueError(f"sort_kind must be one of {VALID_SORT_KINDS}, got {sort_kind!r}")

        self.df = df
        self.x = x
        self.y = y
        self.metric = metric
        self.sign = sign
        self.strict = strict
        self.sort_kind = sort_kind

        # Fitted artifacts
        self.blocks_: List[_Block] = []           # final monotone blocks (x-domain, first left = -inf, last right = +inf)
        self.groups_: Optional[pd.DataFrame] = None
        self.resolved_sign_: Literal["+", "-"] | None = None

        # History: list of frames; each frame is a list of exported dict blocks
        # representing the *current* stack state (in x-domain).
        self.history_: List[List[dict]] = []

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #

    def fit(self) -> "PAVA":
        """Run PAVA on grouped & ordered data; populate blocks_ and history_.

        Returns:
            Self.

        Raises:
            KeyError: If x or y is missing in df.
            TypeError/ValueError: For non-numeric or non-finite y on clean rows.
            ValueError: If no rows after dropping rows with missing x or y.
        """
        if self.x not in self.df.columns or self.y not in self.df.columns:
            missing = [c for c in (self.x, self.y) if c not in self.df.columns]
            raise KeyError(f"Missing columns in df: {missing}")

        # Clean partition for PAVA (missing x or y are excluded here)
        sub = self.df[[self.x, self.y]].dropna()
        if sub.empty:
            raise ValueError("No rows with non-missing x and y for PAVA.")

        ensure_numeric_series(sub[self.y], self.y)

        # Sort once, then group in that order (stable behavior across pandas)
        if self.sort_kind is None:
            sub_sorted = sub.sort_values(by=self.x, na_position="last")
        else:
            sub_sorted = sub.sort_values(by=self.x, kind=self.sort_kind, na_position="last")
        gb = sub_sorted.groupby(self.x, sort=False)[self.y]

        # Aggregate per unique x (sufficient statistics)
        counts = gb.count().to_numpy(dtype=np.int64)
        sums = gb.sum().to_numpy(dtype=float)
        sums2 = gb.apply(lambda s: np.square(s.to_numpy(dtype=float)).sum()).to_numpy(dtype=float)
        mins = gb.min().to_numpy(dtype=float)
        maxs = gb.max().to_numpy(dtype=float)
        xs = gb.mean().index.to_numpy(dtype=float)  # sorted unique x

        # Cache groups for downstream plotting & sign inference
        self.groups_ = pd.DataFrame(
            {"x": xs, "count": counts, "sum": sums, "sum2": sums2, "ymin": mins, "ymax": maxs}
        )

        # Resolve sign
        if self.sign in {"+", "-"}:
            resolved_sign = self.sign
        else:
            # Infer from corr(x, group mean); default to '+' if len==1 or NaN
            means_by_x = self.groups_["sum"] / self.groups_["count"].clip(lower=1)
            corr = np.corrcoef(self.groups_["x"], means_by_x)[0, 1] if len(self.groups_) > 1 else 1.0
            resolved_sign = "+" if (corr >= 0 or np.isnan(corr)) else "-"
        self.resolved_sign_ = resolved_sign

        # Build initial x-domain blocks with:
        #   - first block LEFT = -inf (domain coverage to the left),
        #   - interior blocks left = unique x,
        #   - last block RIGHT = +inf (domain coverage to the right).
        init_blocks: List[_Block] = []
        k = len(xs)
        if k == 1:
            # Single unique value: one infinite block around it.
            init_blocks.append(_Block(
                left=float("-inf"),
                right=float("inf"),
                n=int(counts[0]), sum=float(sums[0]), sum2=float(sums2[0]),
                ymin=float(mins[0]), ymax=float(maxs[0]),
            ))
        else:
            for i in range(k):
                if i == 0:
                    left = float("-inf")
                    right = float(xs[i + 1])
                elif i == k - 1:
                    left = float(xs[i])
                    right = float("inf")
                else:
                    left = float(xs[i])
                    right = float(xs[i + 1])
                init_blocks.append(_Block(
                    left=left, right=right,
                    n=int(counts[i]), sum=float(sums[i]), sum2=float(sums2[i]),
                    ymin=float(mins[i]), ymax=float(maxs[i]),
                ))

        # PAVA stack with history snapshots
        stack: List[_Block] = []
        self.history_.clear()

        def snapshot():
            """Record a copy of the current stack as dicts (for animation)."""
            self.history_.append([b.as_dict() for b in stack])

        for b in init_blocks:
            stack.append(b)
            snapshot()  # after push, before any merges

            # Merge while the last two violate monotonicity (or strict plateau)
            while len(stack) >= 2:
                b2 = stack[-1]
                b1 = stack[-2]

                if self._violates(b1, b2, resolved_sign):
                    merged = b1.merge_with(b2)
                    stack.pop(); stack.pop()
                    stack.append(merged)
                    snapshot()
                else:
                    # Optional plateau merge to enforce *strict* monotone
                    if self.strict and abs(b2.mean - b1.mean) <= 1e-12:
                        merged = b1.merge_with(b2)
                        stack.pop(); stack.pop()
                        stack.append(merged)
                        snapshot()
                    else:
                        break

        self.blocks_ = stack
        return self

    def export_blocks(self, as_dict: bool = True) -> List[dict] | List[Tuple]:
        """Export final monotone blocks.

        Args:
            as_dict: If True, returns list of dicts (primitive types).
                     If False, returns list of tuples
                     (left,right,n,sum,sum2,ymin,ymax).

        Returns:
            List of exported blocks.
        """
        if as_dict:
            return [b.as_dict() for b in self.blocks_]
        return [(b.left, b.right, b.n, b.sum, b.sum2, b.ymin, b.ymax) for b in self.blocks_]

    def export_history(self, as_dict: bool = True) -> List[List[dict]] | List[List[Tuple]]:
        """Export the recorded PAVA history frames.

        Each frame represents the stack after a push/merge step.

        Args:
            as_dict: If True, each frame is a list of dict blocks;
                     otherwise, lists of tuples.

        Returns:
            List of frames.
        """
        if as_dict:
            # Already stored as dicts; return a deep copy to be safe.
            return [[{**d} for d in frame] for frame in self.history_]
        out: List[List[Tuple]] = []
        for frame in self.history_:
            out.append([(d["left"], d["right"], d["n"], d["sum"], d["sum2"], d["ymin"], d["ymax"]) for d in frame])
        return out

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _violates(b1: _Block, b2: _Block, sign: Literal["+", "-"]) -> bool:
        """Return True if (b1, b2) violates monotonicity for the given sign.

        We allow a tiny tolerance (1e-12) for floating error.
        """
        if sign == "+":
            return b2.mean < b1.mean - 1e-12
        else:
            return b2.mean > b1.mean + 1e-12
