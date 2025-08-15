"""Core utility functions for MOBPY.

This module provides essential helper functions for data validation,
partitioning, and statistical calculations used throughout the package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, Union, Any, Dict
import warnings

import numpy as np
import pandas as pd

from MOBPY.exceptions import DataError
from MOBPY.config import get_config
from MOBPY.logging_utils import get_logger

logger = get_logger(__name__)


def ensure_numeric_series(s: pd.Series, name: str) -> None:
    """Validate that a pandas Series is numeric and finite.
    
    Performs comprehensive checks to ensure the series is suitable for
    numerical operations in the binning pipeline.
    
    Args:
        s: Series to validate.
        name: Human-readable name for error messages.
        
    Raises:
        DataError: If series is not numeric or contains non-finite values.
        
    Notes:
        This function checks the non-null values only. NaN values are
        acceptable as they indicate missing data to be handled separately.
        
    Examples:
        >>> s = pd.Series([1, 2, 3, np.nan, 5])
        >>> ensure_numeric_series(s, "target")  # OK
        
        >>> s = pd.Series([1, 2, np.inf])
        >>> ensure_numeric_series(s, "feature")  # Raises DataError
    """
    if not pd.api.types.is_numeric_dtype(s):
        dtype_info = f"dtype={s.dtype}"
        sample_values = s.dropna().head(3).tolist() if not s.dropna().empty else []
        raise DataError(
            f"Column '{name}' must be numeric, but got {dtype_info}. "
            f"Sample values: {sample_values}"
        )
    
    # Check for infinity or other non-finite values (excluding NaN)
    non_null = s.dropna()
    if non_null.empty:
        logger.warning(f"Column '{name}' contains only null values")
        return
    
    finite_mask = np.isfinite(non_null.to_numpy())
    if not finite_mask.all():
        n_inf = (~finite_mask).sum()
        inf_indices = non_null.index[~finite_mask].tolist()[:5]  # Show first 5
        raise DataError(
            f"Column '{name}' contains {n_inf} non-finite values "
            f"(inf or -inf) at indices {inf_indices}..."
        )


def is_binary_series(s: pd.Series, strict: bool = False) -> bool:
    """Check if a Series represents binary data (0/1 values).
    
    Args:
        s: Series to check.
        strict: If True, requires exactly {0, 1} values.
                If False, allows any two unique values that can be
                coerced to {0, 1}.
                
    Returns:
        bool: True if series is binary, False otherwise.
        
    Notes:
        - NaN values are ignored in the check
        - Empty series returns False
        - Single unique value returns False (not truly binary)
        
    Examples:
        >>> s = pd.Series([0, 1, 1, 0, np.nan])
        >>> is_binary_series(s)  # True
        
        >>> s = pd.Series([True, False, True])  
        >>> is_binary_series(s, strict=False)  # True
        >>> is_binary_series(s, strict=True)   # False (not exactly 0/1)
    """
    # Remove nulls for checking
    clean = s.dropna()
    if clean.empty:
        return False
    
    unique_vals = pd.Series(clean.unique())
    
    if strict:
        # Strict mode: must be exactly {0, 1}
        if len(unique_vals) != 2:
            return False
        return set(unique_vals) == {0, 1} or set(unique_vals) == {0.0, 1.0}
    else:
        # Flexible mode: any two values that look like 0/1
        if len(unique_vals) > 2:
            return False
        
        try:
            # Try to convert to int and check if {0, 1}
            as_int = unique_vals.astype(int)
            if not np.array_equal(unique_vals, as_int):
                # Values changed during conversion, not integer-like
                return False
            return set(as_int.tolist()) == {0, 1} or len(unique_vals) == 1
        except (ValueError, TypeError):
            return False


def validate_column_exists(df: pd.DataFrame, columns: Union[str, list]) -> None:
    """Validate that columns exist in DataFrame.
    
    Args:
        df: DataFrame to check.
        columns: Single column name or list of column names.
        
    Raises:
        DataError: If any column is missing.
        
    Examples:
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> validate_column_exists(df, 'a')  # OK
        >>> validate_column_exists(df, ['a', 'b'])  # OK
        >>> validate_column_exists(df, 'c')  # Raises DataError
    """
    if isinstance(columns, str):
        columns = [columns]
    
    missing = [col for col in columns if col not in df.columns]
    if missing:
        available = list(df.columns)[:10]  # Show first 10 available
        raise DataError(
            f"Missing columns in DataFrame: {missing}. "
            f"Available columns: {available}{'...' if len(df.columns) > 10 else ''}"
        )


def woe_iv(
    goods: np.ndarray,
    bads: np.ndarray,
    smoothing: float = 0.5,
    return_components: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Dict[str, np.ndarray]]:
    """Calculate Weight of Evidence (WoE) and Information Value (IV).
    
    WoE measures the predictive power of a binned feature with respect
    to the target variable. IV aggregates this across all bins.
    
    Args:
        goods: Number of good outcomes (y=0) per bin.
        bads: Number of bad outcomes (y=1) per bin.
        smoothing: Laplace smoothing parameter to avoid log(0).
                   Common values: 0.5 (default), 1.0 (Laplace), 0.001 (minimal).
        return_components: If True, return detailed components as dict.
        
    Returns:
        If return_components=False: Tuple of (woe_array, iv_array)
        If return_components=True: Dict with keys 'woe', 'iv', 'good_rate', 'bad_rate'
        
    Notes:
        WoE = ln(good_rate / bad_rate)
        IV = Σ (good_rate - bad_rate) * WoE
        
        Smoothing prevents division by zero when a bin has no goods or bads.
        
    Examples:
        >>> goods = np.array([100, 200, 150])
        >>> bads = np.array([20, 30, 40])
        >>> woe, iv = woe_iv(goods, bads)
        
        >>> # Get detailed components
        >>> components = woe_iv(goods, bads, return_components=True)
        >>> print(f"Total IV: {components['iv'].sum():.4f}")
    """
    config = get_config()
    
    # Ensure float arrays for computation
    goods = np.asarray(goods, dtype=float)
    bads = np.asarray(bads, dtype=float)
    
    # Total goods and bads
    total_goods = goods.sum()
    total_bads = bads.sum()
    
    # Handle edge case: no events
    if total_goods == 0 or total_bads == 0:
        logger.warning(
            f"No variation in target: total_goods={total_goods}, total_bads={total_bads}. "
            f"Returning zero WoE/IV."
        )
        n_bins = len(goods)
        zeros = np.zeros(n_bins)
        if return_components:
            return {
                'woe': zeros,
                'iv': zeros,
                'good_rate': zeros if total_goods == 0 else goods / total_goods,
                'bad_rate': zeros if total_bads == 0 else bads / total_bads
            }
        return zeros, zeros
    
    # Calculate rates with smoothing
    # Add smoothing to both numerator and denominator
    good_rate = (goods + smoothing) / (total_goods + smoothing * len(goods))
    bad_rate = (bads + smoothing) / (total_bads + smoothing * len(bads))
    
    # WoE calculation with numerical stability
    # Use clip to prevent extreme values
    woe = np.log(np.clip(good_rate / bad_rate, config.epsilon, 1 / config.epsilon))
    
    # IV calculation for each group
    iv_groups = (good_rate - bad_rate) * woe
    
    if return_components:
        return {
            'woe': woe,
            'iv': iv_groups,
            'good_rate': good_rate,
            'bad_rate': bad_rate,
            'total_iv': iv_groups.sum()
        }
    
    return woe, iv_groups


@dataclass(frozen=True)
class Parts:
    """Container for partitioned dataset by x-column conditions.
    
    Immutable container that holds the three partitions of data based on
    the feature column: clean (valid), missing, and excluded values.
    
    Attributes:
        clean: DataFrame with valid, non-missing, non-excluded x values.
        missing: DataFrame where x is NaN/null.
        excluded: DataFrame where x matches user-specified exclusion values.
        
    Examples:
        >>> parts = partition_df(df, x='feature', exclude_values=[-999])
        >>> print(f"Clean rows: {len(parts.clean)}")
        >>> print(f"Missing rows: {len(parts.missing)}")
        >>> print(f"Excluded rows: {len(parts.excluded)}")
    """
    clean: pd.DataFrame
    missing: pd.DataFrame
    excluded: pd.DataFrame
    
    def summary(self) -> Dict[str, int]:
        """Get partition sizes summary.
        
        Returns:
            Dict with counts for each partition.
        """
        return {
            'clean': len(self.clean),
            'missing': len(self.missing),
            'excluded': len(self.excluded),
            'total': len(self.clean) + len(self.missing) + len(self.excluded)
        }
    
    def validate(self) -> bool:
        """Validate that partitions don't overlap.
        
        Returns:
            bool: True if partitions are valid (no overlapping indices).
        """
        # Check that indices don't overlap
        clean_idx = set(self.clean.index)
        missing_idx = set(self.missing.index)
        excluded_idx = set(self.excluded.index)
        
        return (len(clean_idx & missing_idx) == 0 and
                len(clean_idx & excluded_idx) == 0 and
                len(missing_idx & excluded_idx) == 0)


def partition_df(
    df: pd.DataFrame,
    x: str,
    exclude_values: Optional[Iterable] = None,
    validate: bool = True
) -> Parts:
    """Partition DataFrame into clean, missing, and excluded subsets.
    
    Splits the input DataFrame based on the values in column x:
    - Clean: Valid numeric values not in exclude_values
    - Missing: Rows where x is NaN/null  
    - Excluded: Rows where x matches any exclude_values
    
    Args:
        df: Input DataFrame to partition.
        x: Column name to partition on.
        exclude_values: Values to exclude from clean partition.
                        Common examples: [-999, -1, 999999] for special codes.
        validate: Whether to validate the input column exists.
        
    Returns:
        Parts: Container with clean, missing, and excluded DataFrames.
        
    Raises:
        DataError: If x column doesn't exist (when validate=True).
        
    Examples:
        >>> df = pd.DataFrame({
        ...     'feature': [1, 2, np.nan, -999, 5],
        ...     'target': [0, 1, 1, 0, 1]
        ... })
        >>> parts = partition_df(df, 'feature', exclude_values=[-999])
        >>> print(parts.summary())
        {'clean': 3, 'missing': 1, 'excluded': 1, 'total': 5}
        
    Notes:
        - The three partitions maintain the original DataFrame index
        - Excluded values are matched exactly (using pd.Series.isin)
        - If exclude_values is None or empty, excluded partition will be empty
    """
    if validate:
        validate_column_exists(df, x)
    
    # Missing partition: x is null
    missing_mask = df[x].isna()
    missing = df[missing_mask]
    
    # Excluded partition: x in exclude_values (and not null)
    if exclude_values is not None:
        exclude_list = list(exclude_values) if not isinstance(exclude_values, list) else exclude_values
        if exclude_list:  # Only if list is not empty
            excluded_mask = df[x].notna() & df[x].isin(exclude_list)
            excluded = df[excluded_mask]
            
            # Clean partition: not missing and not excluded
            clean_mask = df[x].notna() & ~df[x].isin(exclude_list)
            clean = df[clean_mask]
            
            # Log partition info
            if len(excluded) > 0:
                logger.info(
                    f"Excluded {len(excluded)} rows with values in {exclude_list[:5]}"
                    f"{'...' if len(exclude_list) > 5 else ''}"
                )
        else:
            excluded = df.iloc[0:0]  # Empty with same structure
            clean = df[df[x].notna()]
    else:
        excluded = df.iloc[0:0]  # Empty with same structure
        clean = df[df[x].notna()]
    
    # Create parts
    parts = Parts(clean=clean, missing=missing, excluded=excluded)
    
    # Validate partitions don't overlap
    if not parts.validate():
        raise DataError("Partitioning error: overlapping indices detected")
    
    # Log summary
    summary = parts.summary()
    logger.debug(
        f"Partitioned {summary['total']} rows: "
        f"clean={summary['clean']}, missing={summary['missing']}, "
        f"excluded={summary['excluded']}"
    )
    
    # Warn if clean partition is very small
    if summary['clean'] < 100 and summary['total'] > 100:
        warnings.warn(
            f"Clean partition has only {summary['clean']} rows out of {summary['total']}. "
            f"This may lead to unstable binning results.",
            UserWarning
        )
    
    return parts


def calculate_correlation(x: pd.Series, y: pd.Series, method: str = 'pearson') -> float:
    """Calculate correlation between two series with proper handling of edge cases.
    
    Args:
        x: First series.
        y: Second series.
        method: Correlation method ('pearson', 'spearman', 'kendall').
        
    Returns:
        float: Correlation coefficient, or 0.0 if undefined.
        
    Examples:
        >>> x = pd.Series([1, 2, 3, 4, 5])
        >>> y = pd.Series([2, 4, 6, 8, 10])
        >>> corr = calculate_correlation(x, y)  # Returns 1.0
    """
    # Remove rows where either x or y is null
    valid_mask = x.notna() & y.notna()
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    # Need at least 2 observations for correlation
    if len(x_clean) < 2:
        logger.warning(f"Insufficient data for correlation: {len(x_clean)} valid pairs")
        return 0.0
    
    # Check for zero variance
    if x_clean.std() == 0 or y_clean.std() == 0:
        logger.warning("Zero variance in one or both series, returning 0 correlation")
        return 0.0
    
    try:
        if method == 'pearson':
            corr = x_clean.corr(y_clean, method='pearson')
        elif method == 'spearman':
            corr = x_clean.corr(y_clean, method='spearman')
        elif method == 'kendall':
            corr = x_clean.corr(y_clean, method='kendall')
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        # Handle NaN result
        if pd.isna(corr):
            logger.warning(f"Correlation calculation returned NaN for method={method}")
            return 0.0
        
        return float(corr)
        
    except Exception as e:
        logger.warning(f"Correlation calculation failed: {e}")
        return 0.0