# src/MOBPY/binning/mob.py
"""
High-level orchestrator that unifies PAVA + constraints-aware merging
and (optionally) MOB-style summaries for binary targets.

This module treats **MOB** (monotonic optimal binning for classification)
as a **special case** of the general PAVA workflow:

    1) PAVA enforces monotonicity on a chosen metric across ordered `x`.
    2) Adjacent-merge stage uses statistical tests + penalties to satisfy
       bin-count and bin-size constraints.
    3) (Optional for binary targets) compute WoE/IV and rate summaries.

Key responsibilities:
    - Handle missing and "exclude" values outside of the monotone engine.
    - Configure and resolve binning constraints.
    - Run PAVA, merge, and materialize bins with stable `[left, right)` edges.
    - For binary targets: produce MOB-friendly columns (goods/bads, WoE, IV).

Example:
    >>> from MOBPY.binning.mob import MonotonicBinner
    >>> from MOBPY.core.constraints import BinningConstraints
    >>> import pandas as pd
    >>>
    >>> df = pd.read_csv("/data/german_data_credit_cat.csv")
    >>> df["default"] = df["default"] - 1  # make binary {0,1}
    >>>
    >>> cons = BinningConstraints(max_bins=6, min_bins=4,
    ...                           max_samples=0.4, min_samples=0.05, min_positives=0.05,
    ...                           initial_pvalue=0.4, maximize_bins=True)
    >>>
    >>> mob = MonotonicBinner(df, x="Durationinmonth", y="default",
    ...                       metric="mean", sign="auto", constraints=cons)
    >>> mob.fit()
    MonotonicBinner(...)
    >>> bins = mob.bins_()     # core bins
    >>> out  = mob.summary_()  # MOB-style summary (with WoE/IV for binary y)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from MOBPY.core.constraints import BinningConstraints
from MOBPY.core.pava import PAVA
from MOBPY.core.merge import Block, merge_adjacent, as_blocks


AssignMode = Literal["interval", "left", "right"]


@dataclass
class _Partitions:
    """Holds dataset partitions by `x` cleanliness/exclusion.

    Attributes:
        clean: Rows used by PAVA/merge (non-missing and not excluded).
        missing: Rows where `x` is NA (or None). None if absent.
        excluded: Rows where `x` ∈ exclude_values. None if absent.
    """
    clean: pd.DataFrame
    missing: Optional[pd.DataFrame]
    excluded: Optional[pd.DataFrame]


class MonotonicBinner:
    """Run PAVA → merge, then expose bins and (optionally) MOB-style summary.

    MOB (binary target) is achieved by setting `metric="mean"` and providing a
    binary `y` (values in {0,1}). For numeric `y`, a numeric summary is produced.

    Args:
        df: Input DataFrame.
        x: Column name used for ordering/bucketing (must be sortable).
        y: Response column name. Must be numeric. If binary {0,1}, MOB summary
            (goods/bads, WoE/IV) is available; otherwise only numeric stats.
        metric: PAVA metric to enforce monotonicity on:
            {"count","mean","sum","std","var","min","max","ptp"}.
            For MOB (binary y), use "mean".
        sign: "+", "-", or "auto" for monotone direction (see `PAVA` docs).
        strict: If True, merges plateaus during PAVA; else allows equality.
        constraints: `BinningConstraints`. If None, sensible defaults used.
        exclude_values: Values in `x` to set aside into their own terminal bins
            (strings will be preserved as strings in the final summary).
        sort_kind: Sorting algorithm for ordering by x (default "mergesort").

    Raises:
        KeyError: If `x`/`y` missing in `df`.
        TypeError: If `y` is not numeric.
        ValueError: For invalid inputs or empty partitions.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        metric: str = "mean",
        sign: Literal["+", "-", "auto"] = "auto",
        strict: bool = True,
        constraints: Optional[BinningConstraints] = None,
        exclude_values: Optional[Union[Sequence[Union[int, float, str]], int, float, str]] = None,
        sort_kind: str = "mergesort",
    ) -> None:
        self.df = df
        self.x = x
        self.y = y
        self.metric = metric
        self.sign = sign
        self.strict = strict
        self.sort_kind = sort_kind

        self.constraints = constraints or BinningConstraints()
        self.exclude_values = self._normalize_exclude(exclude_values)

        # Fitted artifacts
        self._is_binary_y: Optional[bool] = None
        self._parts: Optional[_Partitions] = None
        self._bins_df: Optional[pd.DataFrame] = None          # clean bins only
        self._full_summary_df: Optional[pd.DataFrame] = None  # bins + missing + excluded (MOB-style if binary)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def fit(self) -> "MonotonicBinner":
        """Run the full pipeline and cache outputs.

        Steps:
            - Partition df into clean/missing/excluded by x.
            - Detect binary vs. numeric y on the *clean* partition.
            - Resolve fractional constraints to absolute using clean totals.
            - Run PAVA(metric) on clean → monotone blocks.
            - Merge adjacent blocks until constraints satisfied.
            - Materialize bins DataFrame (clean only).
            - Build (optional) MOB-style summary: add missing/excluded rows.

        Returns:
            Self (for chaining).

        Raises:
            ValueError: If no rows remain for binning after partitioning.
        """
        parts = self._partition_df()
        self._parts = parts
        if parts.clean.empty:
            raise ValueError("No rows available for binning after excluding missing/excluded x values.")

        # Infer binary-ness of y from the CLEAN partition
        yvals = parts.clean[self.y].dropna().unique()
        self._is_binary_y = np.isin(yvals, [0, 1]).all() and len(yvals) <= 2

        # PAVA on the clean subset
        pava = PAVA(
            df=parts.clean[[self.x, self.y]],
            x=self.x,
            y=self.y,
            metric=self.metric,
            sign=self.sign,
            strict=self.strict,
            sort_kind=self.sort_kind,
        ).fit()

        # Resolve constraints against clean totals (positives for binary targets)
        total_n = int(pava.groups_["count"].sum())
        total_pos = int(pava.groups_["sum"].sum()) if self._is_binary_y else 0
        self.constraints.resolve(total_n=total_n, total_pos=total_pos)

        # Export blocks from PAVA safely (dicts) → convert to merge.Block
        blocks_dicts = pava.export_blocks(as_dict=True)   # safe copies, primitives only
        blocks: List[Block] = as_blocks(blocks_dicts)

        # Merge adjacent blocks using statistical tests + penalties
        merged = merge_adjacent(blocks, constraints=self.constraints, is_binary_y=bool(self._is_binary_y))

        # Materialize clean bins as DataFrame
        self._bins_df = self._blocks_to_df(merged)

        # Compose full summary (including missing/excluded); MOB columns if binary
        self._full_summary_df = self._build_full_summary()
        return self

    def bins_(self) -> pd.DataFrame:
        """Return the **clean** bins (no missing/excluded rows).

        Columns:
            left, right, n, sum, mean, std, min, max
            (+ for binary y) positives, negatives, rate

        Returns:
            pd.DataFrame

        Raises:
            RuntimeError: If `fit()` has not been called.
        """
        if self._bins_df is None:
            raise RuntimeError("Call fit() before requesting bins.")
        return self._bins_df.copy()

    def summary_(self) -> pd.DataFrame:
        """Return the final output table.

        - For binary `y`: MOB-style table with WoE/IV and interval strings
          (plus extra rows for missing/excluded when present).
        - For numeric `y`: appended missing/excluded rows with basic stats.

        Returns:
            pd.DataFrame

        Raises:
            RuntimeError: If `fit()` has not been called.
        """
        if self._full_summary_df is None:
            raise RuntimeError("Call fit() before requesting summary.")
        return self._full_summary_df.copy()

    def transform(self, x_series: pd.Series, assign: AssignMode = "interval") -> pd.Series:
        """Assign each value in `x_series` to a bin label/edge.

        Missing and excluded values:
            - Missing (NaN) are labeled "Missing" in interval mode.
            - Excluded values are labeled as their exact value (string) bins.

        Args:
            x_series: Series of x values to map.
            assign: "interval" | "left" | "right".

        Returns:
            Series aligned to `x_series`.
        """
        if self._bins_df is None or self._parts is None:
            raise RuntimeError("Call fit() before transform().")

        # Start with clean mapping from numeric bins
        clean_bins = self._bins_df
        rights = clean_bins["right"].to_numpy()
        lefts = clean_bins["left"].to_numpy()
        labels = np.array([f"[{l}, {r})" for l, r in zip(lefts, rights)], dtype=object)

        vals = x_series.to_numpy()
        out = np.empty_like(vals, dtype=object if assign == "interval" else float)

        # Missing → special label or NaN for edges
        is_na = pd.isna(vals)
        if assign == "interval":
            out[is_na] = "Missing"
        else:
            out[is_na] = np.nan

        # Excluded values → show the exact value when assign=="interval"
        is_excl = np.zeros_like(is_na, dtype=bool)
        if self.exclude_values:
            excl_set = set(self.exclude_values)
            is_excl = pd.Series(vals).isin(excl_set).to_numpy()
            if assign == "interval":
                out[is_excl] = pd.Series(vals).astype(str).to_numpy()[is_excl]
            else:
                out[is_excl] = np.nan

        # Clean values → use searchsorted
        mask = ~(is_na | is_excl)
        if mask.any():
            idx = np.searchsorted(rights, vals[mask], side="right")
            idx = np.clip(idx, 0, len(rights) - 1)
            if assign == "left":
                out[mask] = lefts[idx]
            elif assign == "right":
                out[mask] = rights[idx]
            else:
                out[mask] = labels[idx]

        return pd.Series(out, index=x_series.index, name=assign)

    # --------------------------------------------------------------------- #
    # Internals: partitions, blocks → df, summaries
    # --------------------------------------------------------------------- #

    @staticmethod
    def _normalize_exclude(exclude: Optional[Union[Sequence, int, float, str]]) -> List[Union[int, float, str]]:
        """Return a flat list of exclude values (empty list if None)."""
        if exclude is None:
            return []
        if isinstance(exclude, (list, tuple, set, np.ndarray, pd.Series)):
            return list(exclude)
        return [exclude]

    def _partition_df(self) -> _Partitions:
        """Split df into clean/missing/excluded based on `x`.

        Returns:
            _Partitions with clean, missing (or None), excluded (or None).

        Raises:
            KeyError: If `x` or `y` columns are absent.
            TypeError: If `y` is not numeric.
        """
        if self.x not in self.df or self.y not in self.df:
            raise KeyError(f"Columns not found: x={self.x!r}, y={self.y!r}")

        # y must be numeric for stats
        if not pd.api.types.is_numeric_dtype(self.df[self.y]):
            raise TypeError(f"Response column '{self.y}' must be numeric.")

        dfx = self.df[[self.x, self.y]].copy()

        df_missing = dfx[dfx[self.x].isna()]
        df_excl = None
        df_clean = dfx[dfx[self.x].notna()]

        if self.exclude_values:
            mask_excl = df_clean[self.x].isin(self.exclude_values)
            df_excl = df_clean[mask_excl]
            df_clean = df_clean[~mask_excl]

        return _Partitions(
            clean=df_clean.reset_index(drop=True),
            missing=None if df_missing.empty else df_missing.reset_index(drop=True),
            excluded=None if (df_excl is None or df_excl.empty) else df_excl.reset_index(drop=True),
        )

    def _blocks_to_df(self, blocks: List[Block]) -> pd.DataFrame:
        """Convert merged blocks to a tidy DataFrame (clean bins only)."""
        rows = []
        for b in blocks:
            row = {
                "left": b.left,
                "right": b.right,
                "n": b.n,
                "sum": b.y_sum,
                "mean": b.mean(),
                "std": b.std(),
                "min": b.y_min,
                "max": b.y_max,
            }
            if self._is_binary_y:
                pos = int(round(b.y_sum))
                row.update(
                    {
                        "positives": pos,
                        "negatives": int(b.n - pos),
                        "rate": row["mean"],
                    }
                )
            rows.append(row)
        out = pd.DataFrame(rows)
        # Ensure numeric dtype and stable ordering by left
        out = out.sort_values("left", kind=self.sort_kind).reset_index(drop=True)
        return out

    def _build_full_summary(self) -> pd.DataFrame:
        """Return final summary table, adding missing/excluded if present.

        For binary y:
            - Adds WoE/IV columns and distribution columns.
            - Adds human-readable interval strings.
            - Adds rows for missing and each excluded value (one row per value).

        For numeric y:
            - Returns base stats with appended missing/excluded rows.
        """
        clean_bins = self._bins_df.copy()
        parts = self._parts
        assert parts is not None

        if not self._is_binary_y:
            # Numeric y: attach missing/excluded counts only (no WoE/IV)
            extra_rows = []
            if parts.missing is not None:
                extra_rows.append(self._numeric_row_from_df(parts.missing, label="Missing"))
            if parts.excluded is not None:
                for val, grp in parts.excluded.groupby(self.x, dropna=False):
                    extra_rows.append(self._numeric_row_from_df(grp, label=str(val)))

            if extra_rows:
                add_df = pd.DataFrame(extra_rows)
                return pd.concat([clean_bins, add_df], ignore_index=True, sort=False)
            return clean_bins

        # --- Binary y: compute MOB-friendly columns (goods/bads, WoE/IV) ---

        mob = clean_bins.copy()
        mob.rename(columns={"positives": "bads"}, inplace=True)
        mob["goods"] = mob["n"] - mob["bads"]

        # Prepare interval labels (for plotting/interpretation)
        mob["[intervalStart"] = mob["left"]
        mob["intervalEnd)"] = mob["right"]
        mob["interval"] = mob.apply(lambda r: f"[{r['left']}, {r['right']})", axis=1)

        # Distributions
        mob["dist_obs"] = mob["n"] / mob["n"].sum()
        mob["dist_bads"] = mob["bads"] / mob["bads"].sum() if mob["bads"].sum() > 0 else np.nan
        mob["dist_goods"] = mob["goods"] / mob["goods"].sum() if mob["goods"].sum() > 0 else np.nan

        # WoE with 0.5 smoothing where needed (avoid ±inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            mob["woe"] = np.log(mob["dist_goods"] / mob["dist_bads"])
        zero_mask = (mob["bads"] == 0) | (mob["goods"] == 0)
        if zero_mask.any():
            adj_goods = mob.loc[zero_mask, "goods"] + 0.5
            adj_bads = mob.loc[zero_mask, "bads"] + 0.5
            adj_dist_goods = adj_goods / mob["goods"].sum()
            adj_dist_bads = adj_bads / mob["bads"].sum()
            mob.loc[zero_mask, "woe"] = np.log(adj_dist_goods / adj_dist_bads)

        # IV contribution
        with np.errstate(invalid="ignore"):
            mob["iv_grp"] = (mob["dist_goods"] - mob["dist_bads"]) * mob["woe"]

        # Extend with missing/excluded
        extras = []
        if parts.missing is not None:
            miss = parts.missing
            extras.append(self._binary_row_from_df(miss, label="Missing"))
        if parts.excluded is not None:
            for val, grp in parts.excluded.groupby(self.x, dropna=False):
                extras.append(self._binary_row_from_df(grp, label=str(val)))

        if extras:
            add_df = pd.DataFrame(extras)
            # Align columns (fill missing with NaN), keep clean bins first
            cols = list(mob.columns)
            add_df = add_df.reindex(columns=cols, fill_value=np.nan)
            out = pd.concat([mob, add_df], ignore_index=True, sort=False)
        else:
            out = mob

        return out

    # -------------------------- helpers: summary rows -------------------------- #

    def _numeric_row_from_df(self, dfsub: pd.DataFrame, label: str) -> dict:
        """Build a numeric-y summary row from a small df slice for output."""
        y = dfsub[self.y].to_numpy()
        n = int(y.size)
        y_sum = float(y.sum())
        mean = float(y_sum / n) if n else np.nan
        std = float(np.sqrt(np.var(y, ddof=1))) if n > 1 else 0.0
        return {
            "left": np.nan,
            "right": np.nan,
            "n": n,
            "sum": y_sum,
            "mean": mean,
            "std": std,
            "min": float(np.min(y)) if n else np.nan,
            "max": float(np.max(y)) if n else np.nan,
            # consistent label columns for plotting/helper functions
            "[intervalStart": label,
            "intervalEnd)": label,
            "interval": label,
        }

    def _binary_row_from_df(self, dfsub: pd.DataFrame, label: str) -> dict:
        """Build a MOB-style summary row (for missing/excluded slices)."""
        y = dfsub[self.y].to_numpy()
        n = int(y.size)
        bads = int(y.sum())
        goods = n - bads
        rate = bads / n if n else np.nan
        return {
            "left": np.nan,
            "right": np.nan,
            "n": n,
            "sum": float(bads),
            "mean": float(rate),
            "std": float(np.sqrt(rate * (1 - rate))) if n > 1 else 0.0,
            "min": float(np.min(y)) if n else np.nan,
            "max": float(np.max(y)) if n else np.nan,
            "bads": bads,
            "goods": goods,
            "rate": rate,
            "[intervalStart": label,
            "intervalEnd)": label,
            "interval": label,
            "dist_obs": np.nan,
            "dist_bads": np.nan,
            "dist_goods": np.nan,
            "woe": np.nan,
            "iv_grp": np.nan,
        }

    # --------------------------------------------------------------------- #
    # Representation
    # --------------------------------------------------------------------- #

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        bins = None if self._bins_df is None else len(self._bins_df)
        return f"{cls}(x={self.x!r}, y={self.y!r}, metric={self.metric!r}, sign={self.sign!r}, bins={bins})"
