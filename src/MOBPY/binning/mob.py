from __future__ import annotations

from typing import Iterable, List, Literal, Optional

import numpy as np
import pandas as pd

from MOBPY.core.constraints import BinningConstraints
from MOBPY.core.merge import Block, merge_adjacent, as_blocks
from MOBPY.core.pava import PAVA
from MOBPY.core.utils import Parts, partition_df, woe_iv, is_binary_series


class MonotonicBinner:
    """End-to-end monotone optimal binning orchestrator (MOB-specialized).

    This class orchestrates:
      1) Partitioning into clean/missing/excluded rows (by x),
      2) PAVA on the clean partition to build monotone "atomic" blocks,
      3) Greedy adjacent merges with simple two-sample tests + constraint penalties,
      4) Emitting half-open bins that **cover the full real line**: (-∞, ...), ..., [..., +∞).

    Notes:
        * Only ``metric='mean'`` is implemented. (Median/quantile support is a
          frequent request and is possible, but requires a different merging/test
          logic and is left as **future work**.)
        * For binary targets, the summary includes WoE/IV.

    Args:
        df: Input DataFrame.
        x: Column to bin.
        y: Target column whose **mean** drives monotonicity/score.
        metric: Only ``'mean'`` is supported.
        sign: Monotone direction: ``'+'``, ``'-'``, or ``'auto'`` (infer).
        strict: If True, PAVA also merges equal-mean plateaus to enforce strictness.
        constraints: User constraints (fractions resolved to absolutes in ``fit()``).
        exclude_values: x-values to be reported separately in the summary (not binned).
        sort_kind: Sorting algorithm hint used by PAVA before grouping.

    Attributes (after ``fit()``):
        resolved_sign_: Final monotone direction ('+' or '-').
        _is_binary_y: Whether y is binary on the clean partition.
        _bins_df: Clean numeric bins as a DataFrame.
        _full_summary_df: Summary including Missing/Excluded rows if present.
        _pava: The fitted PAVA instance (pre-merge blocks available).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        metric: Literal["mean"] = "mean",
        sign: Literal["+", "-", "auto"] = "auto",
        strict: bool = True,
        constraints: Optional[BinningConstraints] = None,
        exclude_values: Optional[Iterable] = None,
        sort_kind: Optional[str] = None,
    ):
        if metric != "mean":
            # Explicitly limit scope to 'mean' for robustness/performance.
            raise ValueError("Only metric='mean' is supported at the moment.")
        self.df = df
        self.x = x
        self.y = y
        self.metric = metric
        self.sign = sign
        self.strict = strict
        self.exclude_values = list(exclude_values) if exclude_values is not None else None
        self.sort_kind = sort_kind
        self.constraints = constraints or BinningConstraints()

        # fitted artifacts
        self._parts: Optional[Parts] = None
        self._pava: Optional[PAVA] = None
        self.resolved_sign_: Optional[str] = None
        self._is_binary_y: Optional[bool] = None
        self._blocks: Optional[List[Block]] = None
        self._bins_df: Optional[pd.DataFrame] = None
        self._full_summary_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    #                             Public API                             #
    # ------------------------------------------------------------------ #

    def fit(self) -> "MonotonicBinner":
        """Run the full pipeline and cache outputs.

        Steps:
            - Partition df into clean / missing / excluded by x.
            - Detect binary vs numeric y on the clean partition.
            - Resolve fractional constraints using clean totals.
            - Run PAVA(mean) on clean → monotone atomic blocks.
            - Greedily merge adjacent blocks until constraints satisfied.
            - Materialize clean bins DataFrame (full real-line coverage).
            - Build the MOB-style summary (adds Missing/Excluded rows, if any).

        Returns:
            Self (for chaining).

        Raises:
            ValueError: If no rows remain for binning after partitioning.
        """
        parts = partition_df(self.df, self.x, self.exclude_values)
        self._parts = parts
        if parts.clean.empty:
            raise ValueError("No rows available for binning after excluding missing/excluded x values.")

        # Infer binary-ness of y from the CLEAN partition
        self._is_binary_y = is_binary_series(parts.clean[self.y])

        # PAVA on the clean subset
        pava = PAVA(
            df=parts.clean[[self.x, self.y]],
            x=self.x,
            y=self.y,
            metric=self.metric,   # mean only
            sign=self.sign,
            strict=self.strict,
            sort_kind=self.sort_kind or "quicksort",
        ).fit()
        self._pava = pava
        self.resolved_sign_ = pava.resolved_sign_

        # Resolve constraints against clean totals (positives used only for binary targets)
        total_n = int(pava.groups_["count"].sum())
        total_pos = int(pava.groups_["sum"].sum()) if self._is_binary_y else 0
        self.constraints.resolve(total_n=total_n, total_pos=total_pos)

        # Export blocks from PAVA safely (dicts) → convert to merge.Block
        blocks_dicts = pava.export_blocks(as_dict=True)  # safe copies, primitives only
        blocks: List[Block] = as_blocks(blocks_dicts)

        # Merge adjacent blocks using statistical tests + penalties
        merged = merge_adjacent(blocks, constraints=self.constraints, is_binary_y=bool(self._is_binary_y))
        if len(merged) == 0:
            raise RuntimeError("Merging produced zero bins; please report with data/constraints.")
        self._blocks = merged

        # Materialize clean bins as DataFrame (first left = -inf, last right = +inf)
        self._bins_df = self._blocks_to_df(merged)

        # Compose full summary (including missing/excluded); MOB columns if binary
        self._full_summary_df = self._build_full_summary()
        return self

    def bins_(self) -> pd.DataFrame:
        """Return clean numeric bins (no Missing/Excluded rows)."""
        if self._bins_df is None:
            raise RuntimeError("Call fit() first.")
        return self._bins_df.copy()

    def summary_(self) -> pd.DataFrame:
        """Return full summary with WoE/IV when y is binary.

        Includes extra rows for Missing and Excluded values (if any).
        """
        if self._full_summary_df is None:
            raise RuntimeError("Call fit() first.")
        return self._full_summary_df.copy()

    def transform(self, x_values: pd.Series, assign: Literal["interval", "left", "right"] = "interval") -> pd.Series:
        """Map raw x to the fitted interval labels or edges.

        Bins are half-open: ``[left, right)``; the first is ``(-∞, right)`` and
        the last is ``[left, +∞)`` — this guarantees full coverage.

        Args:
            x_values: Series of values to transform.
            assign: "interval" → string label "[left, right)",
                    "left" → left edge, "right" → right edge.

        Returns:
            Series of assignments. Missing/excluded get "Missing" or the value repr.

        Raises:
            RuntimeError: If the model is not fitted yet.
        """
        if self._bins_df is None:
            raise RuntimeError("Call fit() first.")

        # Fast vectorized map for clean values
        bins = self._bins_df
        lefts = bins["left"].to_numpy()
        rights = bins["right"].to_numpy()

        def _assign_one(v):
            if pd.isna(v):
                return "Missing"
            if self.exclude_values and v in self.exclude_values:
                return str(v) if assign == "interval" else np.nan
            # Find the bin: index where v < right (since right edges are sorted)
            i = np.searchsorted(rights, v, side="right")
            i = min(i, len(rights) - 1)
            if v < lefts[i]:
                # back up once if searchsorted jumped too far
                i = max(0, i - 1)
            l, r = lefts[i], rights[i]
            if assign == "left":
                return l
            if assign == "right":
                return r
            label = f"[{_format_edge(l)}, {_format_edge(r)})"
            # stylistic: show open paren for (-inf, right)
            if np.isneginf(l):
                label = "(" + label[1:]
            return label

        out = x_values.apply(_assign_one)
        return out

    # ------------------------ Pre-merge PAVA artifacts ------------------------ #

    def pava_blocks_(self, as_dict: bool = False):
        """Return PAVA blocks (before merge-adjacent).

        Args:
            as_dict: If True, return list of primitive dicts; else ``List[Block]``.

        Returns:
            List of dicts or list of :class:`MOBPY.core.merge.Block`.

        Raises:
            RuntimeError: If fit() was not called.
        """
        if self._pava is None:
            raise RuntimeError("Call fit() first.")
        return self._pava.export_blocks(as_dict=as_dict)

    def pava_groups_(self) -> pd.DataFrame:
        """Return the grouped-by-x table used by PAVA (safe copy).

        Returns:
            DataFrame with columns: ['x','count','sum','sum2','ymin','ymax'].

        Raises:
            RuntimeError: If fit() was not called or groups are unavailable.
        """
        if self._pava is None or self._pava.groups_ is None:
            raise RuntimeError("Call fit() first.")
        return self._pava.groups_.copy()

    # ------------------------------------------------------------------ #
    #                             Internals                               #
    # ------------------------------------------------------------------ #

    def _blocks_to_df(self, blocks: List[Block]) -> pd.DataFrame:
        """Materialize contiguous bins from merged blocks with full coverage.

        We output half-open bins where:
          * first left = **-inf**,
          * last right = **+inf**,
          * middle rights = next block's left.

        This keeps assignment stable even if new data extend beyond observed x.

        Args:
            blocks: Merged blocks.

        Returns:
            DataFrame with columns ['left','right','n','sum','mean','std','min','max'].
        """
        if not blocks:
            return pd.DataFrame(columns=["left", "right", "n", "sum", "mean", "std", "min", "max"])

        rows = []
        for i, b in enumerate(blocks):
            # Right edge: use the next block's left; last is +inf
            right = blocks[i + 1].left if i < len(blocks) - 1 else np.inf
            # Left edge: **force the first bin to start at -inf**
            left = -np.inf if i == 0 else b.left

            rows.append(
                dict(
                    left=left,
                    right=right,
                    n=b.n,
                    sum=b.sum,
                    mean=b.mean,
                    std=b.std,
                    min=b.ymin,
                    max=b.ymax,
                )
            )
        return pd.DataFrame(rows)

    def _build_full_summary(self) -> pd.DataFrame:
        """Create a full summary (numeric bins + Missing/Excluded rows)."""
        bins = self._bins_df.copy()

        # Add interval text (use '(' for (-inf, right))
        bins["interval"] = [f"[{_format_edge(l)}, {_format_edge(r)})" for l, r in zip(bins["left"], bins["right"])]
        if not bins.empty and np.isneginf(bins.loc[bins.index[0], "left"]):
            bins.at[bins.index[0], "interval"] = "(" + bins.at[bins.index[0], "interval"][1:]

        # Binary extras: goods/bads, WoE, IV
        if self._is_binary_y:
            bads = bins["sum"].to_numpy()  # here sum == count of positives (y==1)
            ns = bins["n"].to_numpy()
            goods = ns - bads
            w, iv = woe_iv(goods, bads, smoothing=0.5)
            out = bins.assign(
                nsamples=ns,
                bads=bads,
                goods=goods,
                bad_rate=np.divide(bads, ns, out=np.zeros_like(bads, dtype=float), where=ns > 0),
                woe=w,
                iv_grp=iv,
            )
            out = out[["left", "right", "interval", "nsamples", "bads", "goods", "bad_rate", "woe", "iv_grp"]]
        else:
            out = bins.rename(columns={"n": "nsamples"})
            out = out[["left", "right", "interval", "nsamples", "sum", "mean", "std", "min", "max"]]

        # Append Missing / Excluded rows exactly as separate bins (for display)
        parts = self._parts
        assert parts is not None

        rows = [out]
        if not parts.missing.empty:
            if self._is_binary_y:
                m_ns = len(parts.missing)
                m_bads = parts.missing[self.y].sum()
                m_goods = m_ns - m_bads
                m_rate = 0.0 if m_ns == 0 else m_bads / m_ns
                rows.append(pd.DataFrame([{
                    "left": np.nan, "right": np.nan, "interval": "Missing",
                    "nsamples": m_ns, "bads": m_bads, "goods": m_goods,
                    "bad_rate": m_rate, "woe": np.nan, "iv_grp": 0.0
                }]))
            else:
                rows.append(pd.DataFrame([{
                    "left": np.nan, "right": np.nan, "interval": "Missing",
                    "nsamples": len(parts.missing),
                    "sum": parts.missing[self.y].sum(),
                    "mean": parts.missing[self.y].mean(),
                    "std": parts.missing[self.y].std(ddof=1),
                    "min": parts.missing[self.y].min(),
                    "max": parts.missing[self.y].max(),
                }]))

        if self.exclude_values:
            for val in self.exclude_values:
                ex_rows = parts.excluded[parts.excluded[self.x] == val]
                if ex_rows.empty:
                    continue  # only add if present in data
                if self._is_binary_y:
                    n = len(ex_rows)
                    b = ex_rows[self.y].sum()
                    g = n - b
                    r = 0.0 if n == 0 else b / n
                    rows.append(pd.DataFrame([{
                        "left": np.nan, "right": np.nan, "interval": str(val),
                        "nsamples": n, "bads": b, "goods": g,
                        "bad_rate": r, "woe": np.nan, "iv_grp": 0.0
                    }]))
                else:
                    rows.append(pd.DataFrame([{
                        "left": np.nan, "right": np.nan, "interval": str(val),
                        "nsamples": len(ex_rows),
                        "sum": ex_rows[self.y].sum(),
                        "mean": ex_rows[self.y].mean(),
                        "std": ex_rows[self.y].std(ddof=1),
                        "min": ex_rows[self.y].min(),
                        "max": ex_rows[self.y].max(),
                    }]))

        out = pd.concat(rows, ignore_index=True)
        return out


def _format_edge(v: float) -> str:
    """Compact string for an interval edge, including infinities."""
    if np.isneginf(v):
        return "-inf"
    if np.isposinf(v):
        return "inf"
    s = f"{v:.12g}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s
