"""Pool-Adjacent-Violators Algorithm (PAVA) implementation.

This module provides a fast, numerically stable implementation of PAVA
for isotonic regression on grouped data. It's optimized for the monotonic
binning use case where we need to ensure monotone relationships between
features and target means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Dict, Any
import warnings

import numpy as np
import pandas as pd

from MOBPY.exceptions import DataError, FittingError
from MOBPY.core.utils import ensure_numeric_series, calculate_correlation
from MOBPY.config import get_config
from MOBPY.logging_utils import get_logger, BinningProgressLogger

logger = get_logger(__name__)


@dataclass
class _Block:
    """Internal block representation for PAVA algorithm.
    
    Maintains sufficient statistics for O(1) merge operations and
    variance calculations. All statistics are tracked incrementally
    to avoid numerical issues with large datasets.
    
    Attributes:
        left: Left edge (inclusive) of the block's x-range.
        right: Right edge (exclusive) of the block's x-range.
        n: Number of observations in the block.
        sum: Sum of y values.
        sum2: Sum of y² values (for variance calculation).
        ymin: Minimum y value in the block.
        ymax: Maximum y value in the block.
    """
    left: float
    right: float
    n: int
    sum: float
    sum2: float
    ymin: float
    ymax: float
    
    # Optional fields for tracking merge history
    merge_count: int = field(default=0, compare=False)
    original_groups: List[float] = field(default_factory=list, compare=False)
    
    @property
    def mean(self) -> float:
        """Calculate block mean with zero-division protection.
        
        Returns:
            float: Mean of y values, or 0.0 if empty.
        """
        return self.sum / self.n if self.n > 0 else 0.0
    
    @property
    def var(self) -> float:
        """Calculate unbiased sample variance using Welford's method.
        
        Uses the computational formula: var = (sum2 - sum²/n) / (n-1)
        
        Returns:
            float: Sample variance, guaranteed non-negative.
        """
        if self.n <= 1:
            return 0.0
        
        # Numerical stability: use Welford's formula
        mean_of_squares = self.sum2 / self.n
        square_of_mean = (self.sum / self.n) ** 2
        
        # Ensure non-negative (numerical errors can make this slightly negative)
        var = max(0.0, (mean_of_squares - square_of_mean) * self.n / (self.n - 1))
        return var
    
    @property
    def std(self) -> float:
        """Calculate sample standard deviation.
        
        Returns:
            float: Square root of variance.
        """
        return float(np.sqrt(self.var))
    
    def merge_with(self, other: "_Block") -> "_Block":
        """Merge with another block, pooling all statistics.
        
        Args:
            other: Block to merge with (must be adjacent).
            
        Returns:
            _Block: New merged block with combined statistics.
            
        Notes:
            This creates a new block rather than modifying in place,
            which helps with debugging and maintaining history.
        """
        merged = _Block(
            left=self.left,
            right=other.right,
            n=self.n + other.n,
            sum=self.sum + other.sum,
            sum2=self.sum2 + other.sum2,
            ymin=min(self.ymin, other.ymin),
            ymax=max(self.ymax, other.ymax),
            merge_count=self.merge_count + other.merge_count + 1,
            original_groups=self.original_groups + other.original_groups
        )
        return merged
    
    def as_dict(self) -> Dict[str, Any]:
        """Export block as dictionary for serialization.
        
        Returns:
            Dict containing all block statistics and derived values.
        """
        return {
            'left': float(self.left),
            'right': float(self.right),
            'n': int(self.n),
            'sum': float(self.sum),
            'sum2': float(self.sum2),
            'ymin': float(self.ymin),
            'ymax': float(self.ymax),
            'mean': float(self.mean),
            'std': float(self.std),
            'var': float(self.var),
            'merge_count': int(self.merge_count)
        }


# Valid sorting algorithms for pandas
VALID_SORT_KINDS = {None, "quicksort", "mergesort", "heapsort", "stable"}


class PAVA:
    """Pool-Adjacent-Violators Algorithm for isotonic regression.
    
    PAVA creates a monotone (isotonic) step function by pooling adjacent
    groups that violate the monotonicity constraint. This implementation:
    
    - Groups data by x values before applying PAVA (efficient for discrete x)
    - Uses a stack-based approach for O(n) complexity
    - Maintains full sufficient statistics for downstream analysis
    - Handles both increasing (+) and decreasing (-) monotonicity
    - Optionally enforces strict monotonicity (no plateaus)
    
    Args:
        df: Input DataFrame containing x and y columns.
        x: Name of the feature column to group by.
        y: Name of the target column to aggregate.
        metric: Aggregation metric (only 'mean' currently supported).
        sign: Monotonicity direction: '+' (increasing), '-' (decreasing),
              or 'auto' (infer from data).
        strict: If True, merge equal-mean blocks to ensure strict monotonicity.
        sort_kind: Pandas sorting algorithm. None uses pandas default.
        
    Attributes:
        blocks_: List of monotone blocks after fitting.
        groups_: DataFrame of grouped statistics with cumulative columns.
        resolved_sign_: Actual monotonicity direction used ('+' or '-').
        
    Examples:
        >>> pava = PAVA(df=data, x='feature', y='target', sign='auto')
        >>> pava.fit()
        >>> blocks = pava.export_blocks(as_dict=True)
        >>> print(f"Created {len(blocks)} monotone blocks")
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
        """Initialize PAVA algorithm.
        
        Raises:
            ValueError: If metric is not 'mean' or sort_kind is invalid.
        """
        if metric != "mean":
            # Explicitly limit to mean for now; median would require different algorithm
            raise ValueError(
                f"Only metric='mean' is supported in this version, got '{metric}'. "
                f"Median/quantile support is planned for future releases."
            )
        
        if sort_kind not in VALID_SORT_KINDS:
            raise ValueError(
                f"sort_kind must be one of {VALID_SORT_KINDS}, got {sort_kind!r}"
            )
        
        self.df = df
        self.x = x
        self.y = y
        self.metric = metric
        self.sign = sign
        self.strict = strict
        self.sort_kind = sort_kind
        
        # Results (populated by fit())
        self.blocks_: List[_Block] = []
        self.groups_: Optional[pd.DataFrame] = None
        self.resolved_sign_: Optional[Literal["+", "-"]] = None
        
        # Statistics for diagnostics
        self._n_merges: int = 0
        self._n_initial_groups: int = 0
    
    def fit(self) -> "PAVA":
        """Run PAVA to create monotone blocks.
        
        Main algorithm:
        1. Group data by x and compute statistics
        2. Determine monotonicity direction (if auto)
        3. Apply PAVA using stack-based pooling
        4. Optionally merge equal-mean blocks (strict mode)
        5. Store results with cumulative statistics
        
        Returns:
            Self for method chaining.
            
        Raises:
            DataError: If x/y columns missing or invalid.
            FittingError: If PAVA fails to converge or produces invalid results.
        """
        config = get_config()
        
        with BinningProgressLogger("PAVA fitting", logger) as progress:
            
            # Validate inputs
            progress.update("Validating inputs")
            if self.x not in self.df.columns or self.y not in self.df.columns:
                missing = [c for c in (self.x, self.y) if c not in self.df.columns]
                raise DataError(f"Missing columns in DataFrame: {missing}")
            
            # Work with clean rows only
            sub = self.df[[self.x, self.y]].dropna()
            if sub.empty:
                raise DataError(
                    f"No rows with non-missing {self.x} and {self.y} for PAVA"
                )
            
            ensure_numeric_series(sub[self.y], self.y)
            
            # Sort then group (maintains stable order within groups)
            progress.update("Sorting and grouping data")
            if self.sort_kind is None:
                sub_sorted = sub.sort_values(by=self.x, na_position="last")
            else:
                sub_sorted = sub.sort_values(
                    by=self.x, kind=self.sort_kind, na_position="last"
                )
            
            gb = sub_sorted.groupby(self.x, sort=False)[self.y]
            
            # Compute group statistics efficiently
            progress.update("Computing group statistics")
            groups_data = []
            for x_val, y_group in gb:
                y_array = y_group.to_numpy(dtype=float)
                groups_data.append({
                    'x': float(x_val),
                    'count': len(y_array),
                    'sum': float(y_array.sum()),
                    'sum2': float((y_array ** 2).sum()),
                    'ymin': float(y_array.min()),
                    'ymax': float(y_array.max())
                })
            
            groups = pd.DataFrame(groups_data)
            self._n_initial_groups = len(groups)
            
            # Add cumulative columns for CSD-style visualizations
            groups['cum_count'] = groups['count'].cumsum().astype(float)
            groups['cum_sum'] = groups['sum'].cumsum()
            groups['cum_mean'] = groups['cum_sum'] / groups['cum_count']
            groups['group_mean'] = groups['sum'] / groups['count']
            
            self.groups_ = groups
            
            # Determine monotonicity direction
            progress.update("Determining monotonicity direction")
            if self.sign in {"+", "-"}:
                resolved_sign = self.sign
            else:
                # Infer from correlation
                corr = calculate_correlation(
                    groups['x'], 
                    groups['group_mean'],
                    method='pearson'
                )
                resolved_sign = "+" if corr >= 0 else "-"
                logger.info(
                    f"Inferred monotonicity sign='{resolved_sign}' "
                    f"from correlation={corr:.4f}"
                )
            
            self.resolved_sign_ = resolved_sign
            
            # Initialize blocks from groups
            progress.update("Initializing blocks")
            blocks: List[_Block] = []
            xs = groups['x'].to_numpy(dtype=float)
            
            for i, row in enumerate(groups.itertuples(index=False)):
                # Set right edge to next x value (will adjust last one later)
                right = xs[i + 1] if i < len(xs) - 1 else xs[i]
                
                block = _Block(
                    left=float(row.x),
                    right=right,
                    n=int(row.count),
                    sum=float(row.sum),
                    sum2=float(row.sum2),
                    ymin=float(row.ymin),
                    ymax=float(row.ymax),
                    original_groups=[float(row.x)]
                )
                blocks.append(block)
            
            # CRITICAL: Ensure full real-line coverage
            # This guarantees any future x value can be assigned to a bin
            if blocks:
                blocks[0].left = float('-inf')
                blocks[-1].right = float('inf')
            
            # Apply PAVA pooling with stack
            progress.update("Applying PAVA pooling")
            blocks = self._apply_pava(blocks, resolved_sign)
            
            # Optional: enforce strict monotonicity
            if self.strict and len(blocks) > 1:
                progress.update("Enforcing strict monotonicity")
                blocks = self._enforce_strict_monotonicity(blocks, resolved_sign)
            
            self.blocks_ = blocks
            logger.info(
                f"PAVA complete: {self._n_initial_groups} groups -> "
                f"{len(blocks)} blocks ({self._n_merges} merges)"
            )
        
        return self
    
    def _apply_pava(self, blocks: List[_Block], sign: str) -> List[_Block]:
        """Apply PAVA pooling using stack-based algorithm.
        
        Args:
            blocks: Initial blocks (one per unique x).
            sign: Monotonicity direction ('+' or '-').
            
        Returns:
            List of monotone blocks after pooling.
            
        Notes:
            Stack-based PAVA has O(n) complexity as each block is pushed
            and popped at most once.
        """
        if not blocks:
            return []
        
        config = get_config()
        stack: List[_Block] = []
        
        for block in blocks:
            stack.append(block)
            
            # Keep merging while top two blocks violate monotonicity
            while len(stack) >= 2:
                b2 = stack[-1]  # Current block
                b1 = stack[-2]  # Previous block
                
                # Check for violation based on sign
                violates = False
                if sign == "+":
                    # For increasing: merge if b2.mean < b1.mean
                    violates = b2.mean < b1.mean - config.epsilon
                else:
                    # For decreasing: merge if b2.mean > b1.mean
                    violates = b2.mean > b1.mean + config.epsilon
                
                if violates:
                    # Merge the two blocks
                    merged = b1.merge_with(b2)
                    stack.pop()
                    stack.pop()
                    stack.append(merged)
                    self._n_merges += 1
                else:
                    # No violation, move to next block
                    break
        
        return stack
    
    def _enforce_strict_monotonicity(
        self, blocks: List[_Block], sign: str
    ) -> List[_Block]:
        """Merge blocks with equal means to ensure strict monotonicity.
        
        Args:
            blocks: Blocks after PAVA (weakly monotone).
            sign: Monotonicity direction.
            
        Returns:
            Blocks with strict monotonicity (no plateaus).
        """
        if not blocks:
            return blocks
        
        config = get_config()
        result: List[_Block] = [blocks[0]]
        
        for block in blocks[1:]:
            last = result[-1]
            
            # Check if means are effectively equal
            if abs(block.mean - last.mean) <= config.epsilon:
                # Merge to remove plateau
                merged = last.merge_with(block)
                result[-1] = merged
                self._n_merges += 1
                logger.debug(f"Merged equal-mean blocks: {last.mean:.6f}")
            else:
                result.append(block)
        
        return result
    
    def export_blocks(self, as_dict: bool = True) -> List[Any]:
        """Export fitted blocks in specified format.
        
        Args:
            as_dict: If True, return list of dicts. If False, return
                     list of tuples (for backward compatibility).
                     
        Returns:
            List of blocks in requested format.
            
        Raises:
            FittingError: If called before fit().
            
        Examples:
            >>> pava.fit()
            >>> blocks_dict = pava.export_blocks(as_dict=True)
            >>> blocks_tuple = pava.export_blocks(as_dict=False)
        """
        if not self.blocks_:
            raise FittingError("No blocks available. Call fit() first.")
        
        if as_dict:
            return [b.as_dict() for b in self.blocks_]
        else:
            # Legacy tuple format: (left, right, n, sum, sum2, ymin, ymax)
            return [
                (b.left, b.right, b.n, b.sum, b.sum2, b.ymin, b.ymax)
                for b in self.blocks_
            ]
    
    def validate_monotonicity(self, tolerance: float = 1e-10) -> bool:
        """Validate that blocks satisfy monotonicity constraint.
        
        Args:
            tolerance: Numerical tolerance for comparison.
            
        Returns:
            bool: True if monotonicity is satisfied.
            
        Examples:
            >>> pava.fit()
            >>> assert pava.validate_monotonicity(), "Monotonicity violated!"
        """
        if len(self.blocks_) <= 1:
            return True
        
        means = [b.mean for b in self.blocks_]
        
        if self.resolved_sign_ == "+":
            # Check non-decreasing
            for i in range(1, len(means)):
                if means[i] < means[i-1] - tolerance:
                    logger.warning(
                        f"Monotonicity violation at block {i}: "
                        f"{means[i]} < {means[i-1]}"
                    )
                    return False
        else:
            # Check non-increasing
            for i in range(1, len(means)):
                if means[i] > means[i-1] + tolerance:
                    logger.warning(
                        f"Monotonicity violation at block {i}: "
                        f"{means[i]} > {means[i-1]}"
                    )
                    return False
        
        return True
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about the PAVA fit.
        
        Returns:
            Dict containing diagnostic metrics.
            
        Examples:
            >>> diag = pava.get_diagnostics()
            >>> print(f"Compression ratio: {diag['compression_ratio']:.2f}")
        """
        if not self.blocks_:
            return {
                'fitted': False,
                'n_initial_groups': 0,
                'n_final_blocks': 0,
                'n_merges': 0,
                'compression_ratio': 0.0
            }
        
        return {
            'fitted': True,
            'n_initial_groups': self._n_initial_groups,
            'n_final_blocks': len(self.blocks_),
            'n_merges': self._n_merges,
            'compression_ratio': (
                self._n_initial_groups / len(self.blocks_)
                if self.blocks_ else 0.0
            ),
            'resolved_sign': self.resolved_sign_,
            'strict_monotone': self.strict,
            'total_samples': sum(b.n for b in self.blocks_),
            'mean_block_size': (
                sum(b.n for b in self.blocks_) / len(self.blocks_)
                if self.blocks_ else 0.0
            )
        }