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

from MOBPY.core.constraints import BinningConstraints
from MOBPY.core.merge import Block, merge_adjacent, MergeStrategy
from MOBPY.core.pava import PAVA
from MOBPY.core.utils import (
    Parts, partition_df, woe_iv, is_binary_series, 
    ensure_numeric_series, validate_column_exists
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
    
    This class orchestrates the complete MOB pipeline:
    1. Partition data into clean/missing/excluded subsets
    2. Apply PAVA to create initial monotonic blocks
    3. Merge adjacent blocks based on statistical tests and constraints
    4. Generate final bins with full real-line coverage
    5. Calculate WoE/IV for binary targets
    
    The algorithm ensures that bins cover the entire real line using half-open
    intervals: the first bin is (-∞, c₁), middle bins are [cᵢ, cᵢ₊₁), and the
    last bin is [cₙ, +∞). This guarantees any future value can be assigned.
    
    Args:
        df: Input DataFrame containing feature and target columns.
        x: Name of the feature column to bin.
        y: Name of the target column.
        metric: Aggregation metric. Only 'mean' is currently supported.
        sign: Monotonicity direction: '+' (increasing), '-' (decreasing),
              or 'auto' (infer from data).
        strict: If True, enforce strict monotonicity (no plateaus).
        constraints: Binning constraints. If None, uses defaults.
        exclude_values: Feature values to exclude from binning (e.g., special codes).
                        These are reported separately in the summary.
        sort_kind: Pandas sorting algorithm for PAVA. None uses pandas default.
        merge_strategy: Strategy for selecting adjacent blocks to merge.
        
    Attributes:
        resolved_sign_: Actual monotonicity direction used ('+' or '-').
        
    Examples:
        >>> # Basic usage with binary target
        >>> binner = MonotonicBinner(df, x='age', y='default')
        >>> binner.fit()
        >>> bins = binner.bins_()
        >>> summary = binner.summary_()  # Includes WoE/IV
        
        >>> # Custom constraints
        >>> constraints = BinningConstraints(
        ...     max_bins=5,
        ...     min_samples=0.05,  # 5% of data per bin
        ...     min_positives=0.01  # 1% of positives per bin
        ... )
        >>> binner = MonotonicBinner(
        ...     df, x='income', y='approved',
        ...     constraints=constraints,
        ...     exclude_values=[-999, -1]  # Special codes
        ... )
        
        >>> # Transform new data
        >>> new_bins = binner.transform(new_df['income'])
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
        self.resolved_sign_: Optional[Literal["+", "-"]] = None
        self._is_fitted: bool = False
        self._is_binary_y: bool = False
        self._parts: Optional[Parts] = None
        self._pava: Optional[PAVA] = None
        self._merged_blocks: Optional[List[Block]] = None
        self._bins_df: Optional[pd.DataFrame] = None
        self._full_summary_df: Optional[pd.DataFrame] = None
        
        # Diagnostics
        self._fit_diagnostics: Dict[str, Any] = {}
    
    def fit(self) -> "MonotonicBinner":
        """Fit the monotonic binner to the data.
        
        Main pipeline:
        1. Partition data by x values (clean/missing/excluded)
        2. Check if y is binary on clean partition
        3. Resolve constraints based on actual data size
        4. Run PAVA to create initial monotonic blocks
        5. Merge adjacent blocks to satisfy constraints
        6. Build final bins and summary DataFrame
        
        Returns:
            Self for method chaining.
            
        Raises:
            DataError: If data has issues (e.g., no clean rows).
            FittingError: If fitting fails (e.g., PAVA convergence).
        """
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
                # Ensure it's coded as 0/1 for WoE calculations
                y_clean = self._parts.clean[self.y]
                unique_vals = y_clean.dropna().unique()
                if set(unique_vals) != {0, 1} and set(unique_vals) != {0.0, 1.0}:
                    # Convert to 0/1
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
            
            # Work with a copy to avoid modifying user's constraints
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
            # Export blocks as dictionaries for merge_adjacent
            pava_blocks = self._pava.export_blocks(as_dict=True)
            
            # Validate PAVA result
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
            
            # Step 7: Build full summary
            progress.update("Creating summary with WoE/IV")
            self._full_summary_df = self._build_full_summary()
            
            # Store diagnostics
            self._fit_diagnostics = {
                'partition_summary': self._parts.summary(),
                'is_binary': self._is_binary_y,
                'resolved_sign': self.resolved_sign_,
                'pava_diagnostics': self._pava.get_diagnostics(),
                'n_pava_blocks': len(pava_blocks),
                'n_final_bins': len(self._merged_blocks),
                'constraints_satisfied': self._check_constraints_satisfied(),
            }
            
            self._is_fitted = True
            logger.info("MOB fitting complete")
        
        return self
    
    def bins_(self) -> pd.DataFrame:
        """Get the fitted bins DataFrame.
        
        Returns only the numeric bins (excludes Missing/Excluded rows).
        Bins use half-open intervals [left, right) with full real-line coverage.
        
        Returns:
            DataFrame with columns:
            - left: Left bin edge (first is -inf)
            - right: Right bin edge (last is +inf)
            - n: Number of samples
            - sum: Sum of y values
            - mean: Mean of y values
            - std: Standard deviation
            - min: Minimum y value
            - max: Maximum y value
            
        Raises:
            NotFittedError: If called before fit().
            
        Examples:
            >>> binner.fit()
            >>> bins = binner.bins_()
            >>> print(bins[['left', 'right', 'n', 'mean']])
        """
        if not self._is_fitted or self._bins_df is None:
            raise NotFittedError("Call fit() before accessing bins")
        
        return self._bins_df.copy()
    
    def summary_(self) -> pd.DataFrame:
        """Get the full binning summary including WoE/IV for binary targets.
        
        Includes separate rows for Missing and Excluded values if present.
        For binary targets, adds Weight of Evidence and Information Value columns.
        
        Returns:
            DataFrame with columns:
            - bucket: Bin label (e.g., "[-inf, 25.5)", "Missing", "Excluded:-999")
            - count: Number of samples
            - count_pct: Percentage of total samples
            - sum: Sum of y values (events for binary)
            - mean: Mean of y (event rate for binary)
            - std: Standard deviation
            - min/max: Range of y values
            - woe: Weight of Evidence (binary only)
            - iv: Information Value contribution (binary only)
            
        Raises:
            NotFittedError: If called before fit().
            
        Examples:
            >>> summary = binner.summary_()
            >>> print(f"Total IV: {summary['iv'].sum():.4f}")
        """
        if not self._is_fitted or self._full_summary_df is None:
            raise NotFittedError("Call fit() before accessing summary")
        
        return self._full_summary_df.copy()
    
    def transform(
        self,
        x_values: pd.Series,
        assign: Literal["interval", "left", "right", "woe"] = "interval"
    ) -> pd.Series:
        """Transform raw x values to bin assignments.
        
        Maps each value to its corresponding bin using the fitted boundaries.
        Missing values map to "Missing", excluded values to their string repr.
        
        Args:
            x_values: Series of values to transform.
            assign: Type of assignment:
                - "interval": Bin label like "[10, 20)" or "(-inf, 5)"
                - "left": Left edge of the bin
                - "right": Right edge of the bin
                - "woe": Weight of Evidence (binary targets only)
                
        Returns:
            Series with assigned values. For "left"/"right", returns float.
            For "interval", returns string labels. For "woe", returns float
            (NaN for Missing/Excluded).
            
        Raises:
            NotFittedError: If called before fit().
            ValueError: If assign='woe' but target is not binary.
            
        Examples:
            >>> # Get bin labels
            >>> bins = binner.transform(new_df['age'])
            >>> 
            >>> # Get WoE values for scoring
            >>> woe_values = binner.transform(new_df['age'], assign='woe')
        """
        if not self._is_fitted or self._bins_df is None:
            raise NotFittedError("Call fit() before transforming")
        
        if assign == "woe" and not self._is_binary_y:
            raise ValueError("assign='woe' requires binary target")
        
        # Prepare bin edges and WoE mapping if needed
        bins_df = self._bins_df
        lefts = bins_df["left"].to_numpy()
        rights = bins_df["right"].to_numpy()
        
        if assign == "woe":
            # Build WoE lookup from summary
            summary = self._full_summary_df
            # Extract numeric bins only
            numeric_mask = ~summary["bucket"].str.contains("Missing|Excluded")
            woe_map = dict(zip(
                summary.loc[numeric_mask, "bucket"],
                summary.loc[numeric_mask, "woe"]
            ))
        
        def _assign_one(val):
            """Assign a single value to its bin."""
            # Handle missing
            if pd.isna(val):
                if assign in ("left", "right", "woe"):
                    return np.nan
                return "Missing"
            
            # Handle excluded
            if self.exclude_values and val in self.exclude_values:
                if assign in ("left", "right", "woe"):
                    return np.nan
                return f"Excluded:{val}"
            
            # Find the bin containing this value
            # Since bins are [left, right), use searchsorted on rights
            idx = np.searchsorted(rights, val, side="right")
            idx = min(idx, len(rights) - 1)
            
            # Verify value is actually in this bin
            if idx > 0 and val < lefts[idx]:
                idx -= 1
            
            # Extract bin info
            left_edge = lefts[idx]
            right_edge = rights[idx]
            
            # Return requested assignment
            if assign == "left":
                return left_edge
            elif assign == "right":
                return right_edge
            elif assign == "interval":
                label = f"[{_format_edge(left_edge)}, {_format_edge(right_edge)})"
                # Special case: first bin uses open left parenthesis
                if np.isneginf(left_edge):
                    label = "(" + label[1:]
                return label
            elif assign == "woe":
                # Build the interval label to look up WoE
                label = f"[{_format_edge(left_edge)}, {_format_edge(right_edge)})"
                if np.isneginf(left_edge):
                    label = "(" + label[1:]
                return woe_map.get(label, np.nan)
        
        # Apply to all values
        return x_values.apply(_assign_one)
    
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
    
    # ---- Pre-merge PAVA artifacts ----
    
    def pava_blocks_(self, as_dict: bool = True) -> Union[List[Dict], List[Block]]:
        """Get the PAVA blocks before merging.
        
        Useful for understanding the initial monotonic structure before
        constraint-based merging.
        
        Args:
            as_dict: If True, return list of dicts. If False, return Block objects.
            
        Returns:
            List of blocks from PAVA (before merge-adjacent).
            
        Raises:
            NotFittedError: If called before fit().
            
        Examples:
            >>> pava_blocks = binner.pava_blocks_()
            >>> print(f"PAVA created {len(pava_blocks)} initial blocks")
        """
        if not self._is_fitted or self._pava is None:
            raise NotFittedError("Call fit() before accessing PAVA blocks")
        
        return self._pava.export_blocks(as_dict=as_dict)
    
    def pava_groups_(self) -> pd.DataFrame:
        """Get the grouped statistics used by PAVA.
        
        Returns the DataFrame of unique x values with their aggregated
        statistics before PAVA pooling.
        
        Returns:
            DataFrame with columns: x, count, sum, sum2, ymin, ymax,
            cum_count, cum_sum, cum_mean, group_mean.
            
        Raises:
            NotFittedError: If called before fit().
            
        Examples:
            >>> groups = binner.pava_groups_()
            >>> print(f"Data has {len(groups)} unique {binner.x} values")
        """
        if not self._is_fitted or self._pava is None:
            raise NotFittedError("Call fit() before accessing PAVA groups")
        
        return self._pava.groups_.copy()
    
    # ---- Private methods ----
    
    def _blocks_to_df(self, blocks: List[Block]) -> pd.DataFrame:
        """Convert blocks to bins DataFrame with proper edge handling.
        
        Creates half-open intervals where:
        - First bin: (-inf, c₁)
        - Middle bins: [cᵢ, cᵢ₊₁)
        - Last bin: [cₙ, +inf)
        
        Args:
            blocks: Merged blocks from the algorithm.
            
        Returns:
            DataFrame with bin information.
        """
        if not blocks:
            return pd.DataFrame(
                columns=["left", "right", "n", "sum", "mean", "std", "min", "max"]
            )
        
        rows = []
        for i, block in enumerate(blocks):
            # Determine edges
            if i == 0:
                # First bin starts at -inf
                left = float("-inf")
            else:
                left = block.left
            
            if i == len(blocks) - 1:
                # Last bin ends at +inf
                right = float("inf")
            else:
                # Use next block's left as this bin's right
                right = blocks[i + 1].left
            
            rows.append({
                "left": left,
                "right": right,
                "n": block.n,
                "sum": block.sum,
                "mean": block.mean,
                "std": block.std,
                "min": block.ymin,
                "max": block.ymax,
            })
        
        return pd.DataFrame(rows)
    
    def _build_full_summary(self) -> pd.DataFrame:
        """Build complete summary including Missing/Excluded rows.
        
        For binary targets, calculates WoE and IV using proper smoothing.
        
        Returns:
            DataFrame with full binning summary.
        """
        if self._bins_df is None or self._parts is None:
            raise RuntimeError("Internal error: bins or parts not available")
        
        config = get_config()
        rows = []
        
        # Add numeric bins
        for _, bin_row in self._bins_df.iterrows():
            left = bin_row["left"]
            right = bin_row["right"]
            
            # Format bucket label
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
        
        # Add Missing row if present
        if len(self._parts.missing) > 0:
            y_missing = self._parts.missing[self.y]
            rows.append({
                "bucket": "Missing",
                "count": len(y_missing),
                "sum": y_missing.sum(),
                "mean": y_missing.mean() if len(y_missing) > 0 else 0,
                "std": y_missing.std() if len(y_missing) > 0 else 0,
                "min": y_missing.min() if len(y_missing) > 0 else np.nan,
                "max": y_missing.max() if len(y_missing) > 0 else np.nan,
            })
        
        # Add Excluded rows if present
        if len(self._parts.excluded) > 0:
            # Group by excluded value
            for val, group in self._parts.excluded.groupby(self.x):
                y_group = group[self.y]
                rows.append({
                    "bucket": f"Excluded:{val}",
                    "count": len(y_group),
                    "sum": y_group.sum(),
                    "mean": y_group.mean(),
                    "std": y_group.std(),
                    "min": y_group.min(),
                    "max": y_group.max(),
                })
        
        # Create summary DataFrame
        summary = pd.DataFrame(rows)
        
        # Add percentage column
        total_count = summary["count"].sum()
        summary["count_pct"] = summary["count"] / total_count * 100
        
        # Add WoE/IV for binary targets
        if self._is_binary_y:
            # Calculate goods (y=0) and bads (y=1)
            summary["bads"] = summary["sum"].astype(float)
            summary["goods"] = summary["count"] - summary["bads"]
            
            # Get WoE/IV for numeric bins only
            numeric_mask = ~summary["bucket"].str.contains("Missing|Excluded")
            numeric_indices = summary.index[numeric_mask]
            
            if len(numeric_indices) > 0:
                goods = summary.loc[numeric_indices, "goods"].to_numpy()
                bads = summary.loc[numeric_indices, "bads"].to_numpy()
                
                # Calculate WoE/IV with smoothing
                woe_components = woe_iv(
                    goods, bads, 
                    smoothing=0.5,
                    return_components=True
                )
                
                # Assign to numeric bins
                summary.loc[numeric_indices, "woe"] = woe_components["woe"]
                summary.loc[numeric_indices, "iv"] = woe_components["iv"]
                
                # Non-numeric bins get NaN
                summary.loc[~numeric_mask, "woe"] = np.nan
                summary.loc[~numeric_mask, "iv"] = 0.0
            else:
                summary["woe"] = np.nan
                summary["iv"] = 0.0
            
            # Log total IV
            total_iv = summary["iv"].sum()
            logger.info(f"Total Information Value: {total_iv:.4f}")
            
            # Drop intermediate columns
            summary = summary.drop(columns=["bads", "goods"])
        
        # Reorder columns
        base_cols = ["bucket", "count", "count_pct", "sum", "mean", "std", "min", "max"]
        if self._is_binary_y:
            base_cols.extend(["woe", "iv"])
        
        return summary[base_cols]
    
    def _check_constraints_satisfied(self) -> Dict[str, bool]:
        """Check which constraints were satisfied in the final binning.
        
        Returns:
            Dict mapping constraint name to satisfaction status.
        """
        if not self._merged_blocks:
            return {}
        
        constraints = self.constraints
        blocks = self._merged_blocks
        
        satisfied = {
            "max_bins": len(blocks) <= constraints.max_bins,
            "min_bins": len(blocks) >= constraints.min_bins,
        }
        
        # Check sample constraints
        if constraints.abs_min_samples > 0:
            satisfied["min_samples"] = all(
                b.n >= constraints.abs_min_samples for b in blocks
            )
        
        if constraints.abs_max_samples is not None:
            satisfied["max_samples"] = all(
                b.n <= constraints.abs_max_samples for b in blocks
            )
        
        # Check positives constraint (binary only)
        if self._is_binary_y and constraints.abs_min_positives > 0:
            satisfied["min_positives"] = all(
                b.sum >= constraints.abs_min_positives for b in blocks
            )
        
        return satisfied
    
    def __repr__(self) -> str:
        """String representation showing configuration and fit status."""
        status = "fitted" if self._is_fitted else "not fitted"
        n_bins = len(self._merged_blocks) if self._merged_blocks else "N/A"
        
        return (
            f"MonotonicBinner(x='{self.x}', y='{self.y}', "
            f"sign='{self.sign}', status={status}, n_bins={n_bins})"
        )