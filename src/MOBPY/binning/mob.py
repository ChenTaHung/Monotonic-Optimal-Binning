from __future__ import annotations

from typing import Iterable, List, Literal, Optional

import numpy as np
import pandas as pd

from MOBPY.core.constraints import BinningConstraints
from MOBPY.core.merge import Block, merge_adjacent, as_blocks
from MOBPY.core.pava import PAVA
from MOBPY.core.utils import Parts, partition_df, woe_iv, is_binary_series


class MonotonicBinner:
    """End-to-end monotone optimal binning (MOB special case of PAVA).

    This class is the “public” pipeline:

    1) Partition the data into **clean** / **missing** / **excluded** by `x`.
    2) Detect whether `y` is binary on the *clean* part (MOB special case).
    3) Run **PAVA** with `metric="mean"` to get monotone blocks.
    4) Greedily **merge adjacent** blocks with a p-value threshold and
       constraint penalties (min/max samples, min positives, etc.).
    5) Materialize numeric bins and (optionally) produce a MOB-style summary.

    Notes:
      * We treat the last numeric bin as right-open **(+∞)** for clean mapping.
      * Missing and special/excluded values are materialized as extra rows in the
        summary (not in the numeric bins).
      * For binary `y`, `sum` is the count of ones; WoE/IV are computed with
        simple additive smoothing to avoid log(0).

    Args:
        df: Input DataFrame.
        x: Column to bin (feature).
        y: Column whose **mean** drives monotonicity / merge tests.
        metric: Only `"mean"` is supported (MOB case).
        sign: `+`, `-`, or `"auto"` (direction of monotonicity).
        strict: If True, equal-means plateaus are merged inside PAVA.
        constraints: BinningConstraints object (fractions resolved in `fit()`).
        exclude_values: Values in `x` to pull into separate bins in the summary.
        sort_kind: Sorting algorithm used in PAVA’s group-by preparation.

    Raises:
        ValueError: If no clean rows remain to bin.
        RuntimeError: If merging produces zero bins (shouldn’t happen).
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
            raise ValueError("Only metric='mean' is supported in the MOB special case.")
        self.df = df
        self.x = x
        self.y = y
        self.metric = metric
        self.sign = sign
        self.strict = strict
        self.exclude_values = list(exclude_values) if exclude_values is not None else None
        self.sort_kind = sort_kind
        self.constraints = constraints or BinningConstraints()

        # Fitted artifacts
        self._parts: Optional[Parts] = None
        self._pava: Optional[PAVA] = None
        self.resolved_sign_: Optional[str] = None
        self._is_binary_y: Optional[bool] = None
        self._blocks: Optional[List[Block]] = None
        self._bins_df: Optional[pd.DataFrame] = None
        self._full_summary_df: Optional[pd.DataFrame] = None

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def fit(self) -> "MonotonicBinner":
        """Run the full pipeline and cache outputs.

        Steps:
            - Partition df into clean/missing/excluded by x.
            - Detect binary vs. numeric y on the *clean* partition.
            - Resolve fractional constraints to absolute using clean totals.
            - Run PAVA(metric='mean') on clean → monotone blocks.
            - Merge adjacent blocks until constraints satisfied.
            - Materialize bins DataFrame (clean only).
            - Build (optional) MOB-style summary: add missing/excluded rows.

        Returns:
            Self (for chaining).
        """
        # 1) Clean/missing/excluded split
        parts = partition_df(self.df, self.x, self.exclude_values)
        self._parts = parts
        if parts.clean.empty:
            raise ValueError("No rows available after excluding missing/special x values.")

        # 2) Detect binary y on clean
        self._is_binary_y = is_binary_series(parts.clean[self.y])

        # 3) PAVA on the clean subset (monotone means)
        pava = PAVA(
            df=parts.clean[[self.x, self.y]],
            x=self.x,
            y=self.y,
            metric=self.metric,
            sign=self.sign,
            strict=self.strict,
            sort_kind=self.sort_kind,
        ).fit()
        self._pava = pava
        self.resolved_sign_ = pava.resolved_sign_

        # 4) Resolve constraints to absolutes, based on *clean* totals
        total_n = int(pava.groups_["count"].sum())
        total_pos = int(pava.groups_["sum"].sum()) if self._is_binary_y else 0
        self.constraints.resolve(total_n=total_n, total_pos=total_pos)

        # 5) Merge adjacent blocks (accept dicts from PAVA; coerce to Block)
        blocks_dicts = pava.export_blocks(as_dict=True)  # safe primitive copies
        blocks: List[Block] = as_blocks(blocks_dicts)

        merged = merge_adjacent(
            blocks,
            constraints=self.constraints,
            is_binary_y=bool(self._is_binary_y),
        )
        if not merged:
            raise RuntimeError("Merging produced zero bins; please report with data/constraints.")
        self._blocks = merged

        # 6) Materialize clean numeric bins
        self._bins_df = self._blocks_to_df(merged)

        # 7) Full summary (numeric + Missing/Excluded rows)
        self._full_summary_df = self._build_full_summary()
        return self

    def bins_(self) -> pd.DataFrame:
        """Return **clean** numeric bins only (no missing/excluded rows)."""
        if self._bins_df is None:
            raise RuntimeError("Call fit() first.")
        return self._bins_df.copy()

    def summary_(self) -> pd.DataFrame:
        """Return full summary suitable for reporting/plots.

        When `y` is binary, includes WoE/IV; also appends extra rows for
        Missing and any Excluded value that actually appears in the data.
        """
        if self._full_summary_df is None:
            raise RuntimeError("Call fit() first.")
        return self._full_summary_df.copy()

    def transform(
        self,
        x_values: pd.Series,
        assign: Literal["interval", "left", "right"] = "interval",
    ) -> pd.Series:
        """Map raw x-values to the fitted interval labels or edges.

        Args:
            x_values: Series of values to transform.
            assign: One of:
                - "interval": string label like "[a, b)" or "(-inf, b)"
                - "left": left edge value
                - "right": right edge value

        Returns:
            Series of assignments. Missing values map to "Missing".
            Excluded values map to their string repr if `assign="interval"`,
            else NaN (we don't provide numeric edges for special bins).
        """
        if self._bins_df is None:
            raise RuntimeError("Call fit() first.")

        bins = self._bins_df
        lefts = bins["left"].to_numpy()
        rights = bins["right"].to_numpy()

        def _assign_one(v):
            if pd.isna(v):
                return "Missing"
            if self.exclude_values and v in self.exclude_values:
                return str(v) if assign == "interval" else np.nan
            # Half-open membership: left <= v < right
            i = np.searchsorted(rights, v, side="right")
            i = min(i, len(rights) - 1)
            if v < lefts[i]:
                i = max(0, i - 1)
            l, r = lefts[i], rights[i]
            if assign == "left":
                return l
            if assign == "right":
                return r
            label = f"[{_format_edge(l)}, {_format_edge(r)})"
            if np.isneginf(l):
                label = "(" + label[1:]  # (-inf, …)
            return label

        return x_values.apply(_assign_one)

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #

    def _blocks_to_df(self, blocks: List[Block]) -> pd.DataFrame:
        """Materialize contiguous bins from merged blocks.

        We always use **“next-left”** as the right edge, and set the last right
        edge to **+∞** so that transform/searchsorted works cleanly.
        """
        if not blocks:
            return pd.DataFrame(columns=["left", "right", "n", "sum", "mean", "std", "min", "max"])

        rows = []
        for i, b in enumerate(blocks):
            right = blocks[i + 1].left if i < len(blocks) - 1 else np.inf
            rows.append(
                dict(
                    left=b.left,
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
        """Compose a summary DataFrame.

        - For numeric bins: build interval strings and (if binary) WoE/IV.
        - Append rows for Missing and each Excluded value present in data.
        """
        bins = self._bins_df.copy()

        # Interval strings
        bins["interval"] = [f"[{_format_edge(l)}, {_format_edge(r)})" for l, r in zip(bins["left"], bins["right"])]
        # Replace first '[' with '(' for the -inf bin
        bins.at[bins.index[0], "interval"] = "(" + bins.at[bins.index[0], "interval"][1:]

        if self._is_binary_y:
            # Binary case: derive goods/bads and compute WoE/IV
            ns = bins["n"].to_numpy()
            bads = bins["sum"].to_numpy()
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
            # Numeric y: just rename n -> nsamples for readability
            out = bins.rename(columns={"n": "nsamples"})
            out = out[["left", "right", "interval", "nsamples", "sum", "mean", "std", "min", "max"]]

        # Append Missing/Excluded summaries as their own rows (display-only)
        parts = self._parts
        assert parts is not None
        rows = [out]

        # Missing
        if not parts.missing.empty:
            if self._is_binary_y:
                n = len(parts.missing)
                b = parts.missing[self.y].sum()
                g = n - b
                r = 0.0 if n == 0 else b / n
                rows.append(
                    pd.DataFrame(
                        [
                            dict(
                                left=np.nan,
                                right=np.nan,
                                interval="Missing",
                                nsamples=n,
                                bads=b,
                                goods=g,
                                bad_rate=r,
                                woe=np.nan,
                                iv_grp=0.0,
                            )
                        ]
                    )
                )
            else:
                rows.append(
                    pd.DataFrame(
                        [
                            dict(
                                left=np.nan,
                                right=np.nan,
                                interval="Missing",
                                nsamples=len(parts.missing),
                                sum=float(parts.missing[self.y].sum()),
                                mean=float(parts.missing[self.y].mean()),
                                std=float(parts.missing[self.y].std(ddof=1)),
                                min=float(parts.missing[self.y].min()),
                                max=float(parts.missing[self.y].max()),
                            )
                        ]
                    )
                )

        # Excluded (special) values
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
                    rows.append(
                        pd.DataFrame(
                            [
                                dict(
                                    left=np.nan,
                                    right=np.nan,
                                    interval=str(val),
                                    nsamples=n,
                                    bads=b,
                                    goods=g,
                                    bad_rate=r,
                                    woe=np.nan,
                                    iv_grp=0.0,
                                )
                            ]
                        )
                    )
                else:
                    rows.append(
                        pd.DataFrame(
                            [
                                dict(
                                    left=np.nan,
                                    right=np.nan,
                                    interval=str(val),
                                    nsamples=len(ex_rows),
                                    sum=float(ex_rows[self.y].sum()),
                                    mean=float(ex_rows[self.y].mean()),
                                    std=float(ex_rows[self.y].std(ddof=1)),
                                    min=float(ex_rows[self.y].min()),
                                    max=float(ex_rows[self.y].max()),
                                )
                            ]
                        )
                    )

        return pd.concat(rows, ignore_index=True)


def _format_edge(v: float) -> str:
    """Compact, human-friendly tick label for interval edges."""
    if np.isneginf(v):
        return "-inf"
    if np.isposinf(v):
        return "inf"
    s = f"{v:.12g}"  # enough precision w/o sci jitter
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s
