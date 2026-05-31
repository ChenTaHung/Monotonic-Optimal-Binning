"""Monotonic Optimal Binning (MOB) orchestrator.

This module provides the main user-facing API for monotonic binning. It orchestrates
the complete pipeline: data partitioning, PAVA fitting, constraint-based merging,
and final bin creation with optional WoE/IV calculations for binary targets.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union, Any
import warnings

import numpy as np
import pandas as pd

from MOBPY.core.categorical_merge import CategoryBlock, merge_categorical
from MOBPY.core.constraints import BinningConstraints
from MOBPY.core.merge import Block, merge_adjacent, MergeStrategy
from MOBPY.core.pava import PAVA
from MOBPY.core.utils import (
    Parts, partition_df, woe_iv, is_binary_series,
    ensure_numeric_series, ensure_categorical_series, validate_column_exists
)
from MOBPY.exceptions import DataError, NotFittedError, FittingError
from MOBPY.config import get_config
from MOBPY.logging_utils import get_logger, BinningProgressLogger

logger = get_logger(__name__)


def _format_edge(val: float) -> str:
    """Format bin edge for display.
    
    Args:
        val: Edge value (may be ±inf).
        
    Returns:
        str: Formatted string representation.
    """
    if np.isneginf(val):
        return "-inf"
    elif np.isposinf(val):
        return "+inf"
    else:
        # Use appropriate precision based on magnitude
        if abs(val) < 0.01 or abs(val) > 1000:
            return f"{val:.2e}"
        else:
            return f"{val:.4g}"


class MonotonicBinner:
    """End-to-end monotonic optimal binning orchestrator.

    Supports both **numeric** x (PAVA + adjacent merging via Welch's t-test)
    and **categorical** x (chi-square merging with multiple comparison correction).
    The x_type parameter defaults to ``'numeric'``; pass ``x_type='categorical'``
    (or ``'auto'``) to activate the categorical path.

    **Numeric pipeline** (x_type='numeric' or auto-detected):

    1. Partition data into clean/missing/excluded subsets.
    2. Apply PAVA to create initial monotonic blocks.
    3. Merge adjacent blocks using statistical tests and constraints.
    4. Generate final bins with full real-line coverage (-∞ … +∞).
    5. Calculate WoE/IV for binary targets.

    **Categorical pipeline** (x_type='categorical' or auto-detected):

    1. Partition data into clean/missing/excluded subsets.
    2. Build one block per unique category value.
    3. Merge any-pair blocks via chi-square + Holm correction.
    4. Generate final bins as groups of categories.
    5. Calculate WoE/IV for binary targets.

    Args:
        df: Input DataFrame containing feature and target columns.
        x: Name of the feature column to bin.
        y: Name of the target column.
        metric: Aggregation metric. Only 'mean' is currently supported.
        sign: Monotonicity direction for numeric x: '+' (increasing),
              '-' (decreasing), or 'auto' (infer from data).
              Ignored for categorical x.
        strict: If True, enforce strict monotonicity (no plateaus).
              Ignored for categorical x.
        constraints: Binning constraints. If None, uses defaults.
        exclude_values: Feature values to exclude from binning.
                        These are reported separately in the summary.
        sort_kind: Pandas sorting algorithm for PAVA (numeric only).
        merge_strategy: Strategy for selecting blocks to merge (numeric only).
        x_type: Feature type routing. Default ``'numeric'`` always uses the
            numeric (PAVA) path, preserving backward compatibility. Set to
            ``'categorical'`` to use chi-square merging, or ``'auto'`` to
            detect based on the column dtype.
        categorical_alpha: Significance level for categorical merging.
            Pairs with adjusted p-value ≥ alpha are merge candidates.
            Default 0.05.
        categorical_correction: Multiple comparison correction for categorical
            merging. Options: ``'holm'`` (default), ``'bonferroni'``,
            ``'fdr_bh'``.
        unseen_categories: Behaviour when transform() encounters a category
            not seen during fit(). ``'error'`` (default) raises ValueError.
            ``'unknown'`` returns the label ``'Unknown'`` (or NaN for woe assign).
        max_label_cats: Maximum number of category names to include in the
            bin label string. When a bin holds more categories than this,
            the label is truncated: ``{A, B, C, ...+5}``. Applies to
            ``bins_()``, ``summary_()``, and ``transform()`` output.
            ``None`` (default) uses the full label.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        x: str,
        y: str,
        *,
        metric: Literal["mean"] = "mean",
        sign: Literal["+", "-", "auto"] = "auto",
        strict: bool = True,
        constraints: Optional[BinningConstraints] = None,
        exclude_values: Optional[Iterable] = None,
        sort_kind: Optional[str] = "quicksort",
        merge_strategy: Union[MergeStrategy, str] = MergeStrategy.HIGHEST_PVALUE,
        x_type: Literal["auto", "numeric", "categorical"] = "numeric",
        categorical_alpha: float = 0.05,
        categorical_correction: Literal["bonferroni", "holm", "fdr_bh"] = "holm",
        unseen_categories: Literal["unknown", "error"] = "error",
        max_label_cats: Optional[int] = None,
    ):
        """Initialize the binner with configuration.

        Raises:
            ValueError: If metric is not 'mean' or parameters are invalid.
            DataError: If required columns are missing.
        """
        if metric != "mean":
            raise ValueError(
                f"Only metric='mean' is supported in this version, got '{metric}'. "
                f"Median/quantile support is planned for future releases."
            )

        # Validate columns exist
        validate_column_exists(df, [x, y])

        # Store configuration
        self.df = df
        self.x = x
        self.y = y
        self.metric = metric
        self.sign = sign
        self.strict = strict
        self.constraints = constraints or BinningConstraints()
        self.exclude_values = (
            set(exclude_values) if exclude_values is not None else None
        )
        self.sort_kind = sort_kind
        self.x_type = x_type
        self._categorical_alpha = categorical_alpha
        self._categorical_correction = categorical_correction
        self._unseen_categories = unseen_categories
        self._max_label_cats = max_label_cats

        # Handle merge strategy
        if isinstance(merge_strategy, str):
            try:
                self.merge_strategy = MergeStrategy(merge_strategy)
            except ValueError:
                valid = [s.value for s in MergeStrategy]
                raise ValueError(
                    f"Invalid merge_strategy '{merge_strategy}'. "
                    f"Valid options: {valid}"
                )
        else:
            self.merge_strategy = merge_strategy

        # Results (populated by fit())
        self._resolved_x_type: Optional[Literal["numeric", "categorical"]] = None
        self.resolved_sign_: Optional[Literal["+", "-"]] = None
        self._is_fitted: bool = False
        self._is_binary_y: bool = False
        self._parts: Optional[Parts] = None

        # Numeric path results
        self._pava: Optional[PAVA] = None
        self._merged_blocks: Optional[List[Block]] = None
        self._bins_df: Optional[pd.DataFrame] = None

        # Categorical path results
        self._cat_merged_blocks: Optional[List[CategoryBlock]] = None
        self._cat_bins_df: Optional[pd.DataFrame] = None
        self._cat_map: Dict[Any, str] = {}

        # Shared
        self._full_summary_df: Optional[pd.DataFrame] = None

        # Diagnostics
        self._fit_diagnostics: Dict[str, Any] = {}
    
    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def _finalize_summary(self, rows: List[Dict]) -> pd.DataFrame:
        """Append Missing/Excluded rows, add count_pct, and add WoE/IV if binary.

        This shared helper is called by both the numeric and categorical summary
        builders to avoid code duplication.

        Args:
            rows: Pre-built rows for the main bins (numeric intervals or
                categorical group labels). Each row must have keys:
                bucket, count, sum, mean, std, min, max.

        Returns:
            Complete summary DataFrame with all columns.
        """
        assert self._parts is not None

        # Add Missing row if present
        if len(self._parts.missing) > 0:
            y_missing = self._parts.missing[self.y]
            rows.append({
                "bucket": "Missing",
                "count": len(y_missing),
                "sum": float(y_missing.sum()),
                "mean": float(y_missing.mean()) if len(y_missing) > 0 else 0.0,
                "std": float(y_missing.std()) if len(y_missing) > 0 else 0.0,
                "min": float(y_missing.min()) if len(y_missing) > 0 else np.nan,
                "max": float(y_missing.max()) if len(y_missing) > 0 else np.nan,
            })

        # Add Excluded rows if present
        if len(self._parts.excluded) > 0:
            for val, group in self._parts.excluded.groupby(self.x):
                y_group = group[self.y]
                rows.append({
                    "bucket": f"Excluded:{val}",
                    "count": len(y_group),
                    "sum": float(y_group.sum()),
                    "mean": float(y_group.mean()),
                    "std": float(y_group.std()),
                    "min": float(y_group.min()),
                    "max": float(y_group.max()),
                })

        summary = pd.DataFrame(rows)

        # Percentage column
        total_count = summary["count"].sum()
        summary["count_pct"] = summary["count"] / total_count * 100

        # WoE / IV for binary targets
        if self._is_binary_y:
            summary["bads"] = summary["sum"].astype(float)
            summary["goods"] = summary["count"] - summary["bads"]

            woe_components = woe_iv(
                summary["goods"].to_numpy(),
                summary["bads"].to_numpy(),
                smoothing=0.5,
                return_components=True,
            )

            summary["woe"] = woe_components["woe"]
            summary["iv"] = woe_components["iv"]

            total_iv = summary["iv"].sum()
            main_mask = ~summary["bucket"].str.contains("Missing|Excluded", regex=True)
            main_iv = summary.loc[main_mask, "iv"].sum()
            missing_iv = (
                summary.loc[summary["bucket"] == "Missing", "iv"].sum()
                if "Missing" in summary["bucket"].values
                else 0.0
            )
            excluded_iv = summary.loc[
                summary["bucket"].str.startswith("Excluded:"), "iv"
            ].sum()

            logger.info(f"Total Information Value: {total_iv:.4f}")
            logger.info(f"  Main bins IV: {main_iv:.4f}")
            logger.info(f"  Missing bin IV: {missing_iv:.4f}")
            logger.info(f"  Excluded bins IV: {excluded_iv:.4f}")

            summary = summary.drop(columns=["bads", "goods"])

        base_cols = ["bucket", "count", "count_pct", "sum", "mean", "std", "min", "max"]
        if self._is_binary_y:
            base_cols.extend(["woe", "iv"])

        return summary[base_cols]

    def _build_full_summary(self) -> pd.DataFrame:
        """Build the full summary DataFrame for numeric x."""
        assert self._bins_df is not None
        rows = []
        for _, bin_row in self._bins_df.iterrows():
            left = bin_row["left"]
            right = bin_row["right"]
            label = f"[{_format_edge(left)}, {_format_edge(right)})"
            if np.isneginf(left):
                label = "(" + label[1:]
            rows.append({
                "bucket": label,
                "count": int(bin_row["n"]),
                "sum": bin_row["sum"],
                "mean": bin_row["mean"],
                "std": bin_row["std"],
                "min": bin_row["min"],
                "max": bin_row["max"],
            })
        return self._finalize_summary(rows)

    def _build_categorical_summary(self) -> pd.DataFrame:
        """Build the full summary DataFrame for categorical x."""
        assert self._cat_bins_df is not None
        rows = []
        for _, bin_row in self._cat_bins_df.iterrows():
            rows.append({
                "bucket": bin_row["label"],
                "count": int(bin_row["n"]),
                "sum": bin_row["sum"],
                "mean": bin_row["mean"],
                "std": bin_row["std"],
                "min": bin_row["min"],
                "max": bin_row["max"],
            })
        return self._finalize_summary(rows)

    # ------------------------------------------------------------------
    # Fit: public dispatcher + two private paths
    # ------------------------------------------------------------------

    def fit(self) -> "MonotonicBinner":
        """Run the complete binning pipeline.

        Automatically routes to the numeric or categorical fitting path based
        on ``x_type``. Default ``x_type='numeric'`` always uses the numeric
        (PAVA) path. Set ``x_type='categorical'`` for chi-square merging, or
        ``x_type='auto'`` to detect from the column dtype.

        Returns:
            Self for method chaining.

        Raises:
            DataError: If data has issues (e.g., no clean rows, wrong dtype).
            FittingError: If fitting fails (e.g., PAVA convergence, zero bins).
        """
        if self.x_type == "auto":
            self._resolved_x_type = (
                "numeric"
                if pd.api.types.is_numeric_dtype(self.df[self.x])
                else "categorical"
            )
        else:
            self._resolved_x_type = self.x_type  # type: ignore[assignment]

        if self._resolved_x_type == "categorical":
            return self._fit_categorical()
        else:
            return self._fit_numeric()

    def _fit_numeric(self) -> "MonotonicBinner":
        """Run the numeric (PAVA + adjacent merge) fitting pipeline."""
        config = get_config()

        with BinningProgressLogger("MOB fitting", logger) as progress:

            # Step 1: Partition data
            progress.update("Partitioning data")
            self._parts = partition_df(
                self.df, self.x, self.exclude_values, validate=False
            )

            if len(self._parts.clean) == 0:
                raise DataError(
                    f"No clean rows after removing missing/excluded values. "
                    f"Partition summary: {self._parts.summary()}"
                )

            # Validate y column on clean partition
            ensure_numeric_series(self._parts.clean[self.y], self.y)

            # Step 2: Check if binary target
            progress.update("Checking target type")
            self._is_binary_y = is_binary_series(
                self._parts.clean[self.y], strict=False
            )

            if self._is_binary_y:
                y_clean = self._parts.clean[self.y]
                unique_vals = y_clean.dropna().unique()
                if set(unique_vals) != {0, 1} and set(unique_vals) != {0.0, 1.0}:
                    if len(unique_vals) == 2:
                        val_map = {min(unique_vals): 0, max(unique_vals): 1}
                        self._parts.clean[self.y] = y_clean.map(val_map)
                        logger.info(
                            f"Converted binary target to 0/1 using mapping {val_map}"
                        )

            # Step 3: Resolve constraints
            progress.update("Resolving constraints")
            total_n = len(self._parts.clean)
            total_pos = (
                int(self._parts.clean[self.y].sum()) if self._is_binary_y else 0
            )

            self.constraints = self.constraints.copy()
            self.constraints.resolve(total_n=total_n, total_pos=total_pos)

            logger.info(
                f"Resolved constraints: {self.constraints} "
                f"(n={total_n}, pos={total_pos})"
            )

            # Step 4: Run PAVA
            progress.update("Running PAVA algorithm")
            self._pava = PAVA(
                df=self._parts.clean,
                x=self.x,
                y=self.y,
                metric=self.metric,
                sign=self.sign,
                strict=self.strict,
                sort_kind=self.sort_kind,
            )
            self._pava.fit()

            self.resolved_sign_ = self._pava.resolved_sign_
            pava_blocks = self._pava.export_blocks(as_dict=True)

            if not self._pava.validate_monotonicity():
                raise FittingError("PAVA failed to produce monotonic blocks")

            # Step 5: Merge adjacent blocks
            progress.update("Merging adjacent blocks")
            merge_history: List[List[Dict]] = []

            self._merged_blocks = merge_adjacent(
                blocks=pava_blocks,
                constraints=self.constraints,
                is_binary_y=self._is_binary_y,
                strategy=self.merge_strategy,
                history=merge_history,
            )

            logger.info(
                f"Merged {len(pava_blocks)} PAVA blocks -> "
                f"{len(self._merged_blocks)} final bins"
            )

            # Step 6: Build bins DataFrame
            progress.update("Building bins DataFrame")
            self._bins_df = self._blocks_to_df(self._merged_blocks)

            # Step 7: Build full summary with WoE/IV
            progress.update("Creating summary with WoE/IV")
            self._full_summary_df = self._build_full_summary()

            self._fit_diagnostics = {
                "x_type": "numeric",
                "partition_summary": self._parts.summary(),
                "is_binary": self._is_binary_y,
                "resolved_sign": self.resolved_sign_,
                "pava_diagnostics": self._pava.get_diagnostics(),
                "n_pava_blocks": len(pava_blocks),
                "n_final_bins": len(self._merged_blocks),
                "constraints_satisfied": self._check_constraints_satisfied(),
            }

            self._is_fitted = True
            logger.info("MOB numeric fitting complete")

        return self

    def _fit_categorical(self) -> "MonotonicBinner":
        """Run the categorical (chi-square merge) fitting pipeline."""
        with BinningProgressLogger("MOB categorical fitting", logger) as progress:

            # Step 1: Partition data
            progress.update("Partitioning data")
            self._parts = partition_df(
                self.df, self.x, self.exclude_values, validate=False
            )

            if len(self._parts.clean) == 0:
                raise DataError(
                    f"No clean rows after removing missing/excluded values. "
                    f"Partition summary: {self._parts.summary()}"
                )

            # Validate x column is non-numeric
            ensure_categorical_series(self._parts.clean[self.x], self.x)

            # Validate y column
            ensure_numeric_series(self._parts.clean[self.y], self.y)

            # Step 2: Check if binary target (required for chi-square)
            progress.update("Checking target type")
            self._is_binary_y = is_binary_series(
                self._parts.clean[self.y], strict=False
            )

            if not self._is_binary_y:
                raise DataError(
                    "Categorical binning requires a binary target (y ∈ {0, 1}). "
                    "Non-binary continuous y is not supported for categorical x."
                )

            y_clean = self._parts.clean[self.y]
            unique_vals = y_clean.dropna().unique()
            if set(unique_vals) != {0, 1} and set(unique_vals) != {0.0, 1.0}:
                if len(unique_vals) == 2:
                    val_map = {min(unique_vals): 0, max(unique_vals): 1}
                    self._parts.clean[self.y] = y_clean.map(val_map)
                    logger.info(
                        f"Converted binary target to 0/1 using mapping {val_map}"
                    )

            # Step 3: Resolve constraints
            progress.update("Resolving constraints")
            total_n = len(self._parts.clean)
            total_pos = int(self._parts.clean[self.y].sum())

            self.constraints = self.constraints.copy()
            self.constraints.resolve(total_n=total_n, total_pos=total_pos)

            logger.info(
                f"Resolved constraints: {self.constraints} "
                f"(n={total_n}, pos={total_pos})"
            )

            # Step 4: Build one CategoryBlock per unique category
            progress.update("Building initial category blocks")
            clean = self._parts.clean
            cat_blocks: List[CategoryBlock] = []

            for cat_val, group in clean.groupby(self.x, sort=False):
                y_vals = group[self.y].to_numpy(dtype=float)
                cat_blocks.append(
                    CategoryBlock(
                        n=len(y_vals),
                        sum=float(y_vals.sum()),
                        sum2=float((y_vals ** 2).sum()),
                        ymin=float(y_vals.min()),
                        ymax=float(y_vals.max()),
                        categories=frozenset([cat_val]),
                    )
                )

            if not cat_blocks:
                raise DataError("No categories found in clean data")

            logger.info(f"Created {len(cat_blocks)} initial category blocks")

            # Step 5: Merge categorical blocks
            progress.update("Merging categorical blocks")
            merge_history: List[List[Dict]] = []

            self._cat_merged_blocks = merge_categorical(
                blocks=cat_blocks,
                constraints=self.constraints,
                is_binary_y=self._is_binary_y,
                alpha=self._categorical_alpha,
                correction=self._categorical_correction,
                history=merge_history,
            )

            logger.info(
                f"Merged {len(cat_blocks)} category blocks -> "
                f"{len(self._cat_merged_blocks)} final bins"
            )

            # Step 6: Build categorical bins DataFrame
            progress.update("Building bins DataFrame")
            self._cat_bins_df = self._cat_blocks_to_df(self._cat_merged_blocks)

            # Build category -> bin label mapping for transform()
            self._cat_map = {}
            for _, row in self._cat_bins_df.iterrows():
                for cat in row["categories"]:
                    self._cat_map[cat] = row["label"]

            # Step 7: Build full summary with WoE/IV
            progress.update("Creating summary with WoE/IV")
            self._full_summary_df = self._build_categorical_summary()

            self._fit_diagnostics = {
                "x_type": "categorical",
                "partition_summary": self._parts.summary(),
                "is_binary": self._is_binary_y,
                "n_initial_categories": len(cat_blocks),
                "n_final_bins": len(self._cat_merged_blocks),
                "constraints_satisfied": self._check_categorical_constraints_satisfied(),
            }

            self._is_fitted = True
            logger.info("MOB categorical fitting complete")

        return self
    
    def _blocks_to_df(self, blocks: List[Block]) -> pd.DataFrame:
        """Convert blocks to bins DataFrame with proper edges.
        
        Args:
            blocks: List of Block objects from merging.
            
        Returns:
            DataFrame with bin information.
        """
        rows = []
        for i, block in enumerate(blocks):
            # Determine bin edges
            if i == 0:
                left = -np.inf
            else:
                # Use midpoint between this block's max and previous block's max
                prev_right = blocks[i-1].right
                curr_left = block.left
                left = (prev_right + curr_left) / 2
            
            if i == len(blocks) - 1:
                right = np.inf
            else:
                # Use midpoint between this block's max and next block's min
                curr_right = block.right
                next_left = blocks[i+1].left
                right = (curr_right + next_left) / 2
            
            rows.append({
                'left': left,
                'right': right,
                'n': block.n,
                'sum': block.sum,
                'mean': block.mean,
                'std': block.std,
                'min': block.ymin,
                'max': block.ymax,
            })
        
        return pd.DataFrame(rows)
    
    def _make_cat_label(self, cats_sorted: List[str]) -> str:
        """Build the display label for a categorical bin.

        When ``max_label_cats`` is set and the bin contains more categories,
        the label is truncated: ``{A, B, C, ...+N}``.

        Args:
            cats_sorted: Sorted list of category name strings.

        Returns:
            str: Display label, e.g. ``{A, B, C}`` or ``{A, B, ...+5}``.
        """
        m = self._max_label_cats
        if m is not None and len(cats_sorted) > m:
            shown = cats_sorted[:m]
            remaining = len(cats_sorted) - m
            return "{" + ", ".join(shown) + f", ...+{remaining}" + "}"
        return "{" + ", ".join(cats_sorted) + "}"

    def _cat_blocks_to_df(self, cat_blocks: List[CategoryBlock]) -> pd.DataFrame:
        """Convert CategoryBlock list to a bins DataFrame for categorical x.

        Args:
            cat_blocks: Merged CategoryBlock objects.

        Returns:
            DataFrame with columns: categories (frozenset), label (str),
            n, sum, mean, std, min, max.
        """
        rows = []
        for block in cat_blocks:
            cats_sorted = sorted(str(c) for c in block.categories)
            label = self._make_cat_label(cats_sorted)
            rows.append({
                "categories": frozenset(block.categories),
                "label": label,
                "n": block.n,
                "sum": block.sum,
                "mean": block.mean,
                "std": block.std,
                "min": block.ymin,
                "max": block.ymax,
            })
        return pd.DataFrame(rows)

    def _check_constraints_satisfied(self) -> Dict[str, bool]:
        """Check which constraints were satisfied in the final numeric binning."""
        if self._merged_blocks is None:
            return {}

        results = {}
        results["max_bins"] = len(self._merged_blocks) <= self.constraints.max_bins
        results["min_bins"] = len(self._merged_blocks) >= self.constraints.min_bins
        min_n = min(block.n for block in self._merged_blocks)
        results["min_samples"] = min_n >= self.constraints.abs_min_samples

        if self._is_binary_y:
            min_pos = min(block.sum for block in self._merged_blocks)
            results["min_positives"] = min_pos >= self.constraints.abs_min_positives
            min_neg = min(block.n - block.sum for block in self._merged_blocks)
            results["min_negatives"] = min_neg >= self.constraints.abs_min_negatives

        return results

    def _check_categorical_constraints_satisfied(self) -> Dict[str, bool]:
        """Check which constraints were satisfied in the final categorical binning."""
        if self._cat_merged_blocks is None:
            return {}

        results = {}
        results["max_bins"] = len(self._cat_merged_blocks) <= self.constraints.max_bins
        results["min_bins"] = len(self._cat_merged_blocks) >= self.constraints.min_bins
        min_n = min(b.n for b in self._cat_merged_blocks)
        results["min_samples"] = min_n >= self.constraints.abs_min_samples

        if self._is_binary_y:
            min_pos = min(b.positives for b in self._cat_merged_blocks)
            results["min_positives"] = min_pos >= self.constraints.abs_min_positives
            min_neg = min(b.negatives for b in self._cat_merged_blocks)
            results["min_negatives"] = min_neg >= self.constraints.abs_min_negatives

        return results
    
    def bins_(self) -> pd.DataFrame:
        """Get the fitted bins DataFrame.

        **Numeric x** returns one row per bin with half-open interval edges
        (-∞…+∞ coverage):

        - ``left``, ``right``: Bin edges (first is -inf, last is +inf)
        - ``n``, ``sum``, ``mean``, ``std``, ``min``, ``max``

        **Categorical x** returns one row per merged category group:

        - ``categories``: Sorted list of category values in this bin
        - ``n``, ``sum``, ``mean``, ``std``, ``min``, ``max``

        Missing and Excluded rows are not included; use ``summary_()`` for those.

        Raises:
            NotFittedError: If called before fit().
        """
        if not self._is_fitted:
            raise NotFittedError("Call fit() before accessing bins")

        if self._resolved_x_type == "categorical":
            if self._cat_bins_df is None:
                raise NotFittedError("Call fit() before accessing bins")
            df = self._cat_bins_df.copy()
            # Convert frozenset to sorted list for user-friendly output
            df["categories"] = df["categories"].apply(
                lambda s: sorted(str(c) for c in s)
            )
            return df.drop(columns=["label"])

        if self._bins_df is None:
            raise NotFittedError("Call fit() before accessing bins")
        return self._bins_df.copy()
    
    def summary_(self) -> pd.DataFrame:
        """Get the full binning summary including WoE/IV for binary targets.
        
        Includes separate rows for Missing and Excluded values if present.
        For binary targets, adds Weight of Evidence and Information Value columns.
        WoE and IV are now calculated for ALL bins including Missing and Excluded.
        
        Returns:
            DataFrame with columns:
            - bucket: Bin label (e.g., "[-inf, 25.5)", "Missing", "Excluded:-999")
            - count: Number of samples
            - count_pct: Percentage of total samples
            - sum: Sum of y values (events for binary)
            - mean: Mean of y (event rate for binary)
            - std: Standard deviation
            - min/max: Range of y values
            - woe: Weight of Evidence (binary only, calculated for all bins)
            - iv: Information Value contribution (binary only, calculated for all bins)
            
        Raises:
            NotFittedError: If called before fit().
            
        Examples:
            >>> summary = binner.summary_()
            >>> print(f"Total IV: {summary['iv'].sum():.4f}")
            >>> # Check IV contribution from missing values
            >>> missing_iv = summary[summary['bucket'] == 'Missing']['iv'].sum()
            >>> print(f"Missing bin IV: {missing_iv:.4f}")
        """
        if not self._is_fitted or self._full_summary_df is None:
            raise NotFittedError("Call fit() before accessing summary")
        
        return self._full_summary_df.copy()
    
    def transform(
        self,
        x_values: pd.Series,
        assign: Literal["interval", "left", "right", "woe"] = "interval",
    ) -> pd.Series:
        """Transform raw x values to bin assignments.

        Maps each value to its corresponding bin using the fitted model.
        Missing values map to ``"Missing"``; excluded values to
        ``"Excluded:<value>"``. For ``assign='woe'``, those special bins
        return their calculated WoE values.

        **Categorical x notes:**

        - ``assign='interval'`` returns the merged group label, e.g.
          ``"{A, B, D}"``.
        - ``assign='left'`` / ``assign='right'`` raise ``ValueError`` (no
          numeric edges for categorical bins).
        - Unseen categories (not seen during fit) are handled by the
          ``unseen_categories`` constructor parameter (``'error'`` (default)
          raises ``ValueError``; ``'unknown'`` returns ``"Unknown"`` / NaN for woe).

        Args:
            x_values: Series of values to transform.
            assign: Type of assignment:

                - ``"interval"``: Bin label (interval string for numeric,
                  category group label for categorical).
                - ``"left"``: Left numeric edge (numeric only).
                - ``"right"``: Right numeric edge (numeric only).
                - ``"woe"``: Weight of Evidence (binary targets only).

        Returns:
            Series with assigned values, same index as ``x_values``.

        Raises:
            NotFittedError: If called before fit().
            ValueError: If assign='woe' but target is not binary, or if
                assign='left'/'right' for categorical x.
        """
        if not self._is_fitted:
            raise NotFittedError("Call fit() before transform")

        if assign == "woe" and not self._is_binary_y:
            raise ValueError("WoE assignment requires binary target")

        if self._resolved_x_type == "categorical" and assign in ("left", "right"):
            raise ValueError(
                f"assign='{assign}' is not supported for categorical x "
                f"(no numeric bin edges). Use assign='interval' or assign='woe'."
            )

        result = pd.Series(index=x_values.index, dtype=object)

        # ---- Missing values ----
        missing_mask = x_values.isna()
        if missing_mask.any():
            if assign == "interval":
                result.loc[missing_mask] = "Missing"
            elif assign == "woe" and self._is_binary_y:
                missing_row = self._full_summary_df[
                    self._full_summary_df["bucket"] == "Missing"
                ]
                result.loc[missing_mask] = (
                    missing_row["woe"].iloc[0] if len(missing_row) > 0 else 0.0
                )
            else:
                result.loc[missing_mask] = np.nan

        # ---- Excluded values ----
        if self.exclude_values is not None:
            for exc_val in self.exclude_values:
                exc_mask = x_values == exc_val
                if exc_mask.any():
                    if assign == "interval":
                        result.loc[exc_mask] = f"Excluded:{exc_val}"
                    elif assign == "woe" and self._is_binary_y:
                        exc_row = self._full_summary_df[
                            self._full_summary_df["bucket"] == f"Excluded:{exc_val}"
                        ]
                        result.loc[exc_mask] = (
                            exc_row["woe"].iloc[0] if len(exc_row) > 0 else 0.0
                        )
                    else:
                        result.loc[exc_mask] = np.nan

        # ---- Clean values ----
        exclude_list = (
            list(self.exclude_values) if self.exclude_values is not None else []
        )
        clean_mask = ~missing_mask & ~x_values.isin(exclude_list)

        if not clean_mask.any():
            return result

        clean_vals = x_values[clean_mask]

        if self._resolved_x_type == "categorical":
            self._transform_categorical(result, clean_mask, clean_vals, assign)
        else:
            self._transform_numeric(result, clean_mask, clean_vals, assign)

        return result

    def _transform_numeric(
        self,
        result: pd.Series,
        clean_mask: pd.Series,
        clean_vals: pd.Series,
        assign: str,
    ) -> None:
        """Fill result for the numeric path (in-place)."""
        assert self._bins_df is not None

        if assign == "interval":
            for _, bin_row in self._bins_df.iterrows():
                bin_mask = (clean_vals >= bin_row["left"]) & (
                    clean_vals < bin_row["right"]
                )
                if bin_mask.any():
                    label = (
                        f"[{_format_edge(bin_row['left'])}, "
                        f"{_format_edge(bin_row['right'])})"
                    )
                    if np.isneginf(bin_row["left"]):
                        label = "(" + label[1:]
                    result.loc[clean_mask & bin_mask] = label

        elif assign in ("left", "right"):
            for _, bin_row in self._bins_df.iterrows():
                bin_mask = (clean_vals >= bin_row["left"]) & (
                    clean_vals < bin_row["right"]
                )
                if bin_mask.any():
                    result.loc[clean_mask & bin_mask] = bin_row[assign]

        elif assign == "woe":
            woe_map = dict(
                zip(
                    self._full_summary_df["bucket"],
                    self._full_summary_df["woe"],
                )
            )
            for _, bin_row in self._bins_df.iterrows():
                bin_mask = (clean_vals >= bin_row["left"]) & (
                    clean_vals < bin_row["right"]
                )
                if bin_mask.any():
                    label = (
                        f"[{_format_edge(bin_row['left'])}, "
                        f"{_format_edge(bin_row['right'])})"
                    )
                    if np.isneginf(bin_row["left"]):
                        label = "(" + label[1:]
                    if label in woe_map:
                        result.loc[clean_mask & bin_mask] = woe_map[label]

    def _transform_categorical(
        self,
        result: pd.Series,
        clean_mask: pd.Series,
        clean_vals: pd.Series,
        assign: str,
    ) -> None:
        """Fill result for the categorical path (in-place)."""
        if assign == "interval":
            mapped = clean_vals.map(self._cat_map)
            if mapped.isna().any():
                unseen = clean_vals[mapped.isna()].unique().tolist()
                if self._unseen_categories == "error":
                    raise ValueError(
                        f"Unseen categories encountered during transform: {unseen}. "
                        f"Use unseen_categories='unknown' to handle gracefully."
                    )
                mapped = mapped.fillna("Unknown")
            result.loc[clean_mask] = mapped

        elif assign == "woe":
            woe_map = dict(
                zip(
                    self._full_summary_df["bucket"],
                    self._full_summary_df["woe"],
                )
            )
            bin_labels = clean_vals.map(self._cat_map)
            if bin_labels.isna().any():
                unseen = clean_vals[bin_labels.isna()].unique().tolist()
                if self._unseen_categories == "error":
                    raise ValueError(
                        f"Unseen categories encountered during transform: {unseen}. "
                        f"Use unseen_categories='unknown' to handle gracefully."
                    )
                # Unseen -> NaN WoE
                bin_labels = bin_labels.fillna("__unseen__")
            result.loc[clean_mask] = bin_labels.map(woe_map)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics from the fitting process.
        
        Returns:
            Dict containing:
            - partition_summary: Counts for clean/missing/excluded
            - is_binary: Whether target was binary
            - resolved_sign: Final monotonicity direction
            - pava_diagnostics: PAVA algorithm metrics
            - n_pava_blocks: Number of blocks after PAVA
            - n_final_bins: Number of bins after merging
            - constraints_satisfied: Which constraints were met
            
        Raises:
            NotFittedError: If called before fit().
            
        Examples:
            >>> diag = binner.get_diagnostics()
            >>> print(f"Compression: {diag['n_pava_blocks']} -> {diag['n_final_bins']}")
        """
        if not self._is_fitted:
            raise NotFittedError("Call fit() before getting diagnostics")
        
        return self._fit_diagnostics.copy()
    
    # ---- Categorical bin membership (categorical path only) ----

    def bin_assignment(self) -> "pd.Series":
        """Return a Series mapping every category to its 0-based bin index.

        The bin index directly corresponds to the **row index** of
        ``binner.bins_()`` and ``binner.summary_()``, so you can cross-reference
        any value with::

            ba = binner.bin_assignment()
            # which categories are in bin 2?
            ba[ba == 2].index.tolist()
            # inspect that bin's statistics
            binner.bins_().loc[2]

        Only available after fitting a categorical binner
        (``x_type='categorical'``).

        Returns:
            ``pd.Series`` with name ``'bin'`` and index named ``'category'``,
            mapping each original category string to its 0-based bin index.

        Raises:
            NotFittedError: If called before :meth:`fit`.
            ValueError: If the binner was not fitted on a categorical x.

        Examples:
            >>> binner = MonotonicBinner(df, x='merchant', y='is_fraud',
            ...                         x_type='categorical')
            >>> binner.fit()
            >>> ba = binner.bin_assignment()
            >>> ba.head()
            category
            auto_parts    2
            baby_products  0
            ...
            Name: bin, dtype: int64
        """
        if not self._is_fitted:
            raise NotFittedError("Call fit() before bin_assignment().")
        if self._resolved_x_type != "categorical":
            raise ValueError(
                "bin_assignment() is only available for categorical binners "
                "(x_type='categorical')."
            )
        mapping: Dict[Any, int] = {}
        for bin_idx, block in enumerate(self._cat_merged_blocks):
            for cat in block.categories:
                mapping[cat] = bin_idx
        return pd.Series(mapping, name="bin", dtype=int).rename_axis("category").sort_index()

    # ---- Pre-merge PAVA artifacts (numeric path only) ----

    def pava_blocks_(self, as_dict: bool = True) -> Union[List[Dict], List[Block]]:
        """Get the PAVA blocks before merging (numeric x only).

        Useful for understanding the initial monotonic structure before
        constraint-based merging.

        Args:
            as_dict: If True, return list of dicts. If False, return Block objects.

        Returns:
            List of blocks from PAVA (before merge-adjacent).

        Raises:
            NotFittedError: If called before fit() or if x is categorical
                (PAVA is not used in the categorical path).
        """
        if not self._is_fitted:
            raise NotFittedError("Call fit() before accessing PAVA blocks")
        if self._resolved_x_type == "categorical":
            raise NotFittedError(
                "pava_blocks_() is not available for categorical x. "
                "PAVA is only used in the numeric binning path."
            )
        assert self._pava is not None
        return self._pava.export_blocks(as_dict=as_dict)

    def pava_groups_(self) -> pd.DataFrame:
        """Get the grouped statistics used by PAVA (numeric x only).

        Returns the DataFrame of unique x values with their aggregated
        statistics before PAVA pooling.

        Returns:
            DataFrame with columns: x, count, sum, sum2, ymin, ymax,
            cum_count, cum_sum, cum_mean, group_mean.

        Raises:
            NotFittedError: If called before fit() or if x is categorical.
        """
        if not self._is_fitted:
            raise NotFittedError("Call fit() before accessing PAVA groups")
        if self._resolved_x_type == "categorical":
            raise NotFittedError(
                "pava_groups_() is not available for categorical x. "
                "PAVA is only used in the numeric binning path."
            )
        assert self._pava is not None
        return self._pava.groups_.copy()