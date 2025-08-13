# src/MOBPY/core/pava.py
"""
Pool Adjacent Violators Algorithm (PAVA) engine.

This module performs isotonic regression by pooling adjacent violators on a
chosen metric computed from an input pair (x, y). It is *purely* responsible for:

1) Grouping rows by unique, ordered `x` and computing sufficient statistics.
2) Enforcing monotonicity on an aggregated *metric* via a stack-based PAVA.
3) Returning left-closed, right-open bins `[left, right)` with final stats.

Supported metrics:
    "count", "mean", "sum", "std", "var", "min", "max", "ptp"

Monotonic direction:
    sign="+"    → non-decreasing metric
    sign="-"    → non-increasing metric
    sign="auto" → inferred via Spearman rank correlation between `x` and metric

Output columns:
    left, right, n, sum, mean, std, min, max

Robustness:
    - Validates presence of columns.
    - Requires numeric `y` (raises TypeError otherwise).
    - Rejects non-finite `y` values (NaN/±inf) with a clear error.
    - Rejects empty post-cleaning dataset and empty grouped result.
    - Validates `metric` and `sign`.

Notes:
    - Right edges are exclusive; the final bin’s `right` is set to `+inf`.
    - Internal attributes avoid masking built-ins (e.g., `y_sum` vs. `sum`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

MetricName = Literal["count", "mean", "sum", "std", "var", "min", "max", "ptp"]
Sign = Literal["+", "-", "auto"]


@dataclass
class _Block:
    """Internal sufficient statistics for a contiguous bin.

    Attributes:
        left: Left x-edge (inclusive).
        right: Right x-edge (exclusive). Finalized after PAVA; last = +inf.
        n: Number of rows in the bin.
        y_sum: Sum of y in the bin.
        y_sum2: Sum of squares of y (∑ y^2), used for stable variance pooling.
        y_min: Minimum y in the bin.
        y_max: Maximum y in the bin.
    """
    left: float
    right: float
    n: int
    y_sum: float
    y_sum2: float
    y_min: float
    y_max: float

    # ---------------- Derived statistics (from sufficient stats) ---------------- #

    def mean(self) -> float:
        """Mean of y in the bin."""
        return self.y_sum / self.n if self.n else float("nan")

    def var(self) -> float:
        """Unbiased sample variance using ∑y and ∑y² (guard when n<=1)."""
        if self.n <= 1:
            return 0.0
        num = self.y_sum2 - (self.y_sum * self.y_sum) / self.n
        return max(num / (self.n - 1), 0.0)

    def std(self) -> float:
        """Sample standard deviation."""
        return np.sqrt(self.var())

    def metric_value(self, name: MetricName) -> float:
        """Return this block's value for the requested metric."""
        if name == "count":
            return float(self.n)
        if name == "mean":
            return self.mean()
        if name == "sum":
            return float(self.y_sum)
        if name == "std":
            return self.std()
        if name == "var":
            return self.var()
        if name == "min":
            return float(self.y_min)
        if name == "max":
            return float(self.y_max)
        if name == "ptp":
            return float(self.y_max - self.y_min)
        raise ValueError(f"Unknown metric: {name!r}")

    # ---------------------------------- Merge ---------------------------------- #

    def merge_with(self, other: "_Block") -> "_Block":
        """Create a new block that merges `self` followed by `other`.

        Left edge comes from the earlier block; right edge from the later block.
        Sufficient statistics are pooled additively.
        """
        return _Block(
            left=self.left,
            right=other.right,  # placeholder; fixed after PAVA
            n=self.n + other.n,
            y_sum=self.y_sum + other.y_sum,
            y_sum2=self.y_sum2 + other.y_sum2,
            y_min=min(self.y_min, other.y_min),
            y_max=max(self.y_max, other.y_max),
        )


class PAVA:
    """Pool Adjacent Violators Algorithm for monotone binning on a chosen metric.

    This class aggregates (x, y) by unique x, enforces monotonicity on a metric,
    and returns contiguous bins satisfying the requested direction.

    Args:
        df: Input DataFrame.
        x: Column name of the ordering variable (sortable; numeric recommended).
        y: Column name of the response variable (must be numeric).
        metric: Metric to enforce monotonicity on. One of:
            {"count","mean","sum","std","var","min","max","ptp"}.
        sign: Monotone direction:
            "+"  → non-decreasing (or strictly increasing if `strict=True`)
            "-"  → non-increasing (or strictly decreasing if `strict=True`)
            "auto" → inferred via Spearman correlation between x and metric.
        strict: If True, merges plateaus (treats equality as a violation).
            If False, allows equal-adjacent metric values to remain.
        sort_kind: Sorting algorithm used when ordering by x. "mergesort" is stable.

    Attributes:
        groups_: Grouped per-x statistics used as PAVA input.
        resolved_sign_: The resolved direction ("+" or "-").
        _blocks_: List of internal blocks after PAVA (and finalized edges).
        _bins_df_: Cached bins DataFrame.

    Raises:
        KeyError: If `x` or `y` columns are missing.
        TypeError: If `y` is not numeric.
        ValueError: If data is empty after cleaning, metric/sign invalid, or
                    `y` contains non-finite values.
    """

    __slots__ = (
        "df",
        "x",
        "y",
        "metric",
        "sign",
        "strict",
        "sort_kind",
        "groups_",
        "resolved_sign_",
        "_blocks_",
        "_bins_df_",
    )

    def __init__(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        metric: MetricName = "mean",
        sign: Sign = "auto",
        strict: bool = True,
        sort_kind: str = "mergesort",
    ) -> None:
        # Basic configuration
        self.df = df
        self.x = x
        self.y = y
        self.metric = metric
        self.sign = sign
        self.strict = strict
        self.sort_kind = sort_kind

        # Internal caches
        self.groups_: Optional[pd.DataFrame] = None
        self.resolved_sign_: Optional[Literal["+", "-"]] = None
        self._blocks_: List[_Block] = []
        self._bins_df_: Optional[pd.DataFrame] = None

        # Early validation for metric/sign
        allowed_metrics = {"count", "mean", "sum", "std", "var", "min", "max", "ptp"}
        if self.metric not in allowed_metrics:
            raise ValueError(f"Unsupported metric: {self.metric!r}. Allowed: {sorted(allowed_metrics)}")
        if self.sign not in {"+", "-", "auto"}:
            raise ValueError("sign must be one of {'+','-','auto'}")

    # ================================ Public API ================================ #

    def fit(self) -> "PAVA":
        """Run PAVA and cache the resulting bins.

        Returns:
            Self (for chaining).
        """
        # 1) Validate & prepare (ensures numeric and finite y)
        sub = self._prepare_xy()

        # 2) Group stats per unique, ordered x (sufficient stats)
        grouped = self._group_stats(sub)  # columns: x_value, count, sum, sum2, min, max
        self.groups_ = grouped

        # 3) PAVA on singleton blocks
        blocks = self._run_pava(grouped)

        # 4) Finalize edges to left-closed/right-open; last right=+inf
        self._blocks_ = self._finalize_edges(blocks)

        # 5) Materialize tidy output for convenience
        self._bins_df_ = self._blocks_to_df(self._blocks_)
        return self

    def bins_(self) -> pd.DataFrame:
        """Return the bins as a DataFrame.

        Returns:
            DataFrame with columns: left, right, n, sum, mean, std, min, max.

        Raises:
            RuntimeError: If `fit()` has not been called yet.
        """
        if self._bins_df_ is None:
            raise RuntimeError("Call fit() before requesting bins.")
        return self._bins_df_.copy()

    def export_blocks(self, as_dict: bool = True):
        """Return a **safe copy** of the fitted blocks for downstream merging.

        This accessor avoids exposing private internal objects. By default it
        returns a list of plain dictionaries that contain only primitive types.

        Args:
            as_dict: If True (default), return ``List[dict]`` with keys:
                ``left, right, n, y_sum, y_sum2, y_min, y_max``.
                If False, returns a list of **new `_Block` copies**. Use this
                only if you need the helper methods (`mean`, `var`, `std`) and
                you will not mutate them.

        Returns:
            List[dict] if `as_dict=True`, else `List[_Block]`.

        Raises:
            RuntimeError: If `fit()` has not been called yet.
        """
        if not self._blocks_:
            raise RuntimeError("No blocks available. Call fit() first.")

        if as_dict:
            return [
                {
                    "left": b.left,
                    "right": b.right,
                    "n": b.n,
                    "y_sum": b.y_sum,
                    "y_sum2": b.y_sum2,
                    "y_min": b.y_min,
                    "y_max": b.y_max,
                }
                for b in self._blocks_
            ]

        # Return brand-new _Block copies (still private type; do not mutate).
        return [
            _Block(
                left=b.left,
                right=b.right,
                n=b.n,
                y_sum=b.y_sum,
                y_sum2=b.y_sum2,
                y_min=b.y_min,
                y_max=b.y_max,
            )
            for b in self._blocks_
        ]

    def transform(
        self,
        x_series: pd.Series,
        assign: Literal["interval", "left", "right"] = "interval",
    ) -> pd.Series:
        """Map each value in `x_series` to its bin.

        Args:
            x_series: Series of x values to assign to bins.
            assign: One of {"interval", "left", "right"}:
                - "interval" → string label like "[a, b)"
                - "left"     → numeric left edge
                - "right"    → numeric right edge

        Returns:
            Series aligned to `x_series` with the chosen assignment.

        Raises:
            RuntimeError: If `fit()` has not been called yet.
        """
        bins = self.bins_()
        right_edges = bins["right"].to_numpy()
        left_edges = bins["left"].to_numpy()

        # Rightmost bin has +inf, so every value will be assigned
        idx = np.searchsorted(right_edges, x_series.to_numpy(), side="right")
        idx = np.clip(idx, 0, len(bins) - 1)

        if assign == "left":
            return pd.Series(left_edges[idx], index=x_series.index, name="left")
        if assign == "right":
            return pd.Series(right_edges[idx], index=x_series.index, name="right")

        labels = [f"[{l}, {r})" for l, r in zip(left_edges, right_edges)]
        return pd.Series(np.array(labels, dtype=object)[idx], index=x_series.index, name="interval")

    # ============================== Internals: I/O ============================== #

    def _prepare_xy(self) -> pd.DataFrame:
        """Validate inputs and return a clean (x, y) subset without NA/inf.

        Returns:
            A copy of `df[[x, y]]` with finite y and no missing rows.

        Raises:
            KeyError: If x or y is missing from df.
            TypeError: If y is not numeric.
            ValueError: If data becomes empty after cleaning, or y contains non-finite values.
        """
        if self.x not in self.df or self.y not in self.df:
            raise KeyError(f"Columns not found: x={self.x!r}, y={self.y!r}")

        sub = self.df[[self.x, self.y]].copy()

        # Require numeric y for statistical operations
        if not pd.api.types.is_numeric_dtype(sub[self.y]):
            raise TypeError(f"Response column {self.y!r} must be numeric.")

        # Drop rows with NA in either x or y
        sub = sub.dropna(subset=[self.x, self.y])
        if sub.empty:
            raise ValueError("No data left after dropping NA in x and/or y.")

        # Reject non-finite y (±inf). NaNs already dropped above.
        if not np.isfinite(sub[self.y].to_numpy()).all():
            raise ValueError("Column y contains non-finite values (±inf). Clean or clip before PAVA.")

        return sub

    def _group_stats(self, sub: pd.DataFrame) -> pd.DataFrame:
        """Group by unique x (sorted) and compute sufficient y-stats.

        Returns:
            DataFrame with columns:
                - x_value: unique sorted x
                - count:   number of rows
                - sum:     sum(y)
                - sum2:    sum(y^2), derived from var/mean/count for numerical stability
                - min:     min(y)
                - max:     max(y)

        Raises:
            ValueError: If the grouped result is empty.
        """
        # Stable ordering by x (mergesort preserves equal-key order)
        sub = sub.sort_values(self.x, kind=self.sort_kind)

        g = (
            sub.groupby(self.x, sort=True)[self.y]
            .agg(["count", "sum", "mean", "var", "std", "min", "max"])
            .reset_index()
        )
        if g.empty:
            raise ValueError("Grouped statistics are empty—no unique x values present.")

        # sum2 = var*(n-1) + n*mean^2  (sample variance, ddof=1 in pandas)
        n = g["count"].to_numpy(dtype=float)
        m = g["mean"].to_numpy(dtype=float)
        v = g["var"].fillna(0.0).to_numpy(dtype=float)  # NaN for singletons → 0
        sum2 = v * (n - 1.0) + n * (m * m)

        g.rename(columns={self.x: "x_value"}, inplace=True)
        g["sum2"] = sum2
        # Keep only columns needed for PAVA
        return g[["x_value", "count", "sum", "sum2", "min", "max"]]

    # =========================== Internals: PAVA core =========================== #

    def _resolve_sign(self, blocks: List[_Block]) -> Literal["+", "-"]:
        """Resolve monotone direction when sign='auto' using Spearman ranks.

        Strategy:
            - Compute metric at each (singleton) block.
            - Rank-transform x and metric (Spearman).
            - If correlation >= 0 → '+', else '-'.
        """
        if self.sign in {"+", "-"}:
            return self.sign

        metric_vals = np.array([b.metric_value(self.metric) for b in blocks], dtype=float)
        x_vals = np.array([b.left for b in blocks], dtype=float)

        r_x = pd.Series(x_vals).rank(method="average").to_numpy()
        r_m = pd.Series(metric_vals).rank(method="average").to_numpy()

        with np.errstate(invalid="ignore"):
            corr = np.corrcoef(r_x, r_m)[0, 1]

        corr = 0.0 if np.isnan(corr) else float(corr)
        return "+" if corr >= 0.0 else "-"

    def _run_pava(self, grouped: pd.DataFrame) -> List[_Block]:
        """Run stack-based PAVA over singleton blocks built from grouped stats."""
        # Create one block per unique x value (right edge is a placeholder for now)
        blocks: List[_Block] = [
            _Block(
                left=float(x),
                right=float(x),
                n=int(n),
                y_sum=float(s),
                y_sum2=float(s2),
                y_min=float(ymin),
                y_max=float(ymax),
            )
            for x, n, s, s2, ymin, ymax in grouped.itertuples(index=False, name=None)
        ]

        sign = self._resolve_sign(blocks)
        self.resolved_sign_ = sign

        # Maintain a stack and merge while the last two violate monotonicity.
        st: List[_Block] = []
        for b in blocks:
            st.append(b)
            while len(st) >= 2:
                b2 = st[-1]
                b1 = st[-2]
                m1 = b1.metric_value(self.metric)
                m2 = b2.metric_value(self.metric)

                # Violation test depends on direction and strictness:
                if sign == "+":
                    # Non-decreasing: expect m2 >= m1 (or > m1 if strict)
                    violates = (m2 <= m1) if self.strict else (m2 < m1)
                else:
                    # Non-increasing: expect m2 <= m1 (or < m1 if strict)
                    violates = (m2 >= m1) if self.strict else (m2 > m1)

                if not violates:
                    break  # monotone holds

                # Merge adjacent violators and continue checking
                merged = b1.merge_with(b2)
                st.pop()
                st.pop()
                st.append(merged)

        return st

    def _finalize_edges(self, blocks: List[_Block]) -> List[_Block]:
        """Set right edges to the next block's left; last one to +inf."""
        for i in range(len(blocks) - 1):
            blocks[i].right = blocks[i + 1].left
        blocks[-1].right = float("inf")
        return blocks

    # ============================== Internals: Out ============================== #

    def _blocks_to_df(self, blocks: List[_Block]) -> pd.DataFrame:
        """Convert internal blocks to a tidy DataFrame."""
        rows = []
        for b in blocks:
            rows.append(
                {
                    "left": b.left,
                    "right": b.right,
                    "n": b.n,
                    "sum": b.y_sum,   # public column name remains intuitive
                    "mean": b.mean(),
                    "std": b.std(),
                    "min": b.y_min,
                    "max": b.y_max,
                }
            )
        return pd.DataFrame(rows)
