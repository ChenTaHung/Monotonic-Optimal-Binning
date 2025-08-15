"""Adjacent block merging with statistical tests and constraints.

This module implements the merging phase that follows PAVA, using statistical
tests to decide which adjacent blocks to combine while respecting user-defined
constraints on bin sizes and counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, Any
import math
import warnings
from enum import Enum

import numpy as np
import scipy.stats

from MOBPY.core.constraints import BinningConstraints
from MOBPY.exceptions import FittingError
from MOBPY.config import get_config
from MOBPY.logging_utils import get_logger

logger = get_logger(__name__)


class MergeStrategy(Enum):
    """Strategy for selecting which blocks to merge."""
    HIGHEST_PVALUE = "highest_pvalue"  # Merge most similar blocks
    SMALLEST_LOSS = "smallest_loss"    # Merge with minimum information loss
    BALANCED_SIZE = "balanced_size"    # Prefer merging small blocks


@dataclass
class Block:
    """Contiguous block with sufficient statistics for merging.
    
    Represents a range of x values with aggregated y statistics.
    Designed for O(1) merge operations and variance calculations.
    
    Attributes:
        left: Left boundary (inclusive) of x range.
        right: Right boundary (exclusive) of x range.
        n: Number of samples in the block.
        sum: Sum of y values.
        sum2: Sum of y² values (for variance).
        ymin: Minimum y value.
        ymax: Maximum y value.
        
    Properties:
        mean: Sample mean of y values.
        var: Unbiased sample variance.
        std: Sample standard deviation.
    """
    
    left: float
    right: float
    n: int
    sum: float
    sum2: float
    ymin: float
    ymax: float
    
    # Optional metadata for tracking
    merge_history: List[Tuple[float, float]] = field(default_factory=list, compare=False)
    pvalue_history: List[float] = field(default_factory=list, compare=False)
    
    @property
    def mean(self) -> float:
        """Calculate sample mean safely.
        
        Returns:
            float: Mean value, or 0.0 for empty blocks.
        """
        return self.sum / self.n if self.n > 0 else 0.0
    
    @property
    def var(self) -> float:
        """Calculate unbiased sample variance.
        
        Uses the computational formula with numerical stability checks.
        
        Returns:
            float: Variance, guaranteed non-negative.
        """
        if self.n <= 1:
            return 0.0
        
        # Use stable computation
        mean_sq = (self.sum / self.n) ** 2
        mean_of_sq = self.sum2 / self.n
        
        # Ensure non-negative (handles numerical errors)
        raw_var = (mean_of_sq - mean_sq) * self.n / (self.n - 1)
        return max(0.0, raw_var)
    
    @property
    def std(self) -> float:
        """Calculate sample standard deviation.
        
        Returns:
            float: Square root of variance.
        """
        return math.sqrt(self.var)
    
    @property
    def cv(self) -> float:
        """Calculate coefficient of variation.
        
        Returns:
            float: std/mean ratio, or 0 if mean is 0.
        """
        if abs(self.mean) < 1e-10:
            return 0.0
        return self.std / abs(self.mean)
    
    def merge_with(self, other: "Block") -> "Block":
        """Merge with another block, pooling statistics.
        
        Args:
            other: Block to merge with (should be adjacent).
            
        Returns:
            Block: New merged block with combined statistics.
            
        Notes:
            Creates a new block rather than modifying in place.
            Tracks merge history for debugging.
        """
        # Combine statistics
        merged = Block(
            left=self.left,
            right=other.right,
            n=self.n + other.n,
            sum=self.sum + other.sum,
            sum2=self.sum2 + other.sum2,
            ymin=min(self.ymin, other.ymin),
            ymax=max(self.ymax, other.ymax)
        )
        
        # Track merge history
        merged.merge_history = (
            self.merge_history + 
            [(self.left, self.right)] +
            other.merge_history + 
            [(other.left, other.right)]
        )
        
        return merged
    
    def as_dict(self) -> Dict[str, Any]:
        """Export block as dictionary.
        
        Returns:
            Dict with all block statistics and derived values.
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
            'var': float(self.var),
            'std': float(self.std),
            'cv': float(self.cv)
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Block([{self.left:.3f}, {self.right:.3f}), "
            f"n={self.n}, mean={self.mean:.4f}, std={self.std:.4f})"
        )


class MergeScorer:
    """Scoring system for evaluating potential merges.
    
    Combines statistical tests with constraint-based penalties to
    prioritize which adjacent blocks should be merged.
    """
    
    def __init__(
        self,
        constraints: BinningConstraints,
        is_binary_y: bool,
        strategy: MergeStrategy = MergeStrategy.HIGHEST_PVALUE
    ):
        """Initialize scorer with constraints and strategy.
        
        Args:
            constraints: Resolved binning constraints.
            is_binary_y: Whether target is binary.
            strategy: Merge selection strategy.
        """
        self.constraints = constraints
        self.is_binary_y = is_binary_y
        self.strategy = strategy
        self.config = get_config()
    
    def score_pair(self, a: Block, b: Block) -> float:
        """Score a potential merge between adjacent blocks.
        
        Higher scores indicate more desirable merges.
        
        Args:
            a: First block.
            b: Second block (must be adjacent).
            
        Returns:
            float: Merge score in [0, ∞), higher is better.
        """
        if self.strategy == MergeStrategy.HIGHEST_PVALUE:
            base_score = self._two_sample_pvalue(a, b)
        elif self.strategy == MergeStrategy.SMALLEST_LOSS:
            base_score = 1.0 / (1.0 + self._information_loss(a, b))
        elif self.strategy == MergeStrategy.BALANCED_SIZE:
            base_score = self._size_balance_score(a, b)
        else:
            base_score = self._two_sample_pvalue(a, b)
        
        # Apply constraint-based adjustments
        adjusted = self._apply_penalties(base_score, a, b)
        
        return adjusted
    
    def _two_sample_pvalue(self, a: Block, b: Block) -> float:
        """Two-sample test for difference in means.
        
        Uses Welch's t-test for unequal variances.
        
        Args:
            a: First block.
            b: Second block.
            
        Returns:
            float: P-value in [0, 1], higher means more similar.
        """
        na, nb = a.n, b.n
        
        # Handle edge cases
        if na == 0 or nb == 0:
            return 1.0
        
        if na == 1 and nb == 1:
            # Not enough data for test
            return 1.0 if abs(a.mean - b.mean) < self.config.epsilon else 0.5
        
        # Welch's t-test (doesn't assume equal variances)
        va, vb = a.var, b.var
        
        # Standard error of difference
        se_diff_sq = va / na + vb / nb
        
        if se_diff_sq <= 0:
            # No variance, check if means are equal
            return 1.0 if abs(a.mean - b.mean) < self.config.epsilon else 0.0
        
        se_diff = math.sqrt(se_diff_sq)
        
        # Test statistic
        t_stat = abs(a.mean - b.mean) / se_diff
        
        # Degrees of freedom (Welch-Satterthwaite)
        if va > 0 and vb > 0:
            df_num = se_diff_sq ** 2
            df_denom = (va/na)**2/(na-1) + (vb/nb)**2/(nb-1)
            df = df_num / df_denom
            df = max(1, min(df, na + nb - 2))  # Bound df reasonably
        else:
            df = na + nb - 2
        
        # Two-tailed p-value
        try:
            p_value = 2 * (1 - scipy.stats.t.cdf(abs(t_stat), df=df))
        except Exception as e:
            logger.warning(f"P-value calculation failed: {e}, using fallback")
            # Fallback: normal approximation
            p_value = 2 * (1 - scipy.stats.norm.cdf(abs(t_stat)))
        
        return float(np.clip(p_value, 0, 1))
    
    def _information_loss(self, a: Block, b: Block) -> float:
        """Calculate information loss from merging.
        
        Uses within-block sum of squares as loss metric.
        
        Args:
            a: First block.
            b: Second block.
            
        Returns:
            float: Information loss, lower is better.
        """
        # Current within-block SS
        ss_a = a.sum2 - a.sum**2 / a.n if a.n > 0 else 0
        ss_b = b.sum2 - b.sum**2 / b.n if b.n > 0 else 0
        current_ss = ss_a + ss_b
        
        # Merged within-block SS
        merged_n = a.n + b.n
        merged_sum = a.sum + b.sum
        merged_sum2 = a.sum2 + b.sum2
        
        if merged_n > 0:
            merged_ss = merged_sum2 - merged_sum**2 / merged_n
        else:
            merged_ss = 0
        
        # Loss is increase in within-block SS
        loss = max(0, merged_ss - current_ss)
        
        # Normalize by total variance for scale invariance
        total_var = (a.var * a.n + b.var * b.n) / (a.n + b.n) if (a.n + b.n) > 0 else 1
        
        return loss / max(total_var, self.config.epsilon)
    
    def _size_balance_score(self, a: Block, b: Block) -> float:
        """Score based on creating balanced bin sizes.
        
        Prefers merging small blocks over large ones.
        
        Args:
            a: First block.
            b: Second block.
            
        Returns:
            float: Size balance score in [0, 1].
        """
        # Smaller blocks get higher score
        max_n = max(a.n, b.n)
        min_n = min(a.n, b.n)
        
        if max_n == 0:
            return 1.0
        
        # Ratio of sizes (1 = perfect balance, 0 = very imbalanced)
        balance = min_n / max_n
        
        # Penalty for large blocks
        total_n = a.n + b.n
        if self.constraints.abs_max_samples:
            size_penalty = min(1.0, self.constraints.abs_max_samples / total_n)
        else:
            # Use a reasonable default
            size_penalty = min(1.0, 1000 / total_n)
        
        return balance * size_penalty
    
    def _apply_penalties(self, base_score: float, a: Block, b: Block) -> float:
        """Apply constraint-based adjustments to base score.
        
        Args:
            base_score: Initial score from statistical test.
            a: First block.
            b: Second block.
            
        Returns:
            float: Adjusted score with penalties/bonuses applied.
        """
        score = base_score
        
        # Penalty factors based on constraints
        constraints = self.constraints
        
        # 1. Small bin penalty - encourage merging undersized bins
        if constraints.abs_min_samples > 0:
            if a.n < constraints.abs_min_samples:
                score *= 1.5  # 50% bonus for merging small bin
            if b.n < constraints.abs_min_samples:
                score *= 1.5
        
        # 2. Binary extremes - merge bins with 0% or 100% event rate
        if self.is_binary_y:
            a_rate = a.mean
            b_rate = b.mean
            
            if a_rate <= 0.001 or a_rate >= 0.999:
                score *= 1.3  # 30% bonus
            if b_rate <= 0.001 or b_rate >= 0.999:
                score *= 1.3
        
        # 3. Max samples penalty - discourage creating oversized bins
        if constraints.abs_max_samples:
            merged_n = a.n + b.n
            if merged_n > constraints.abs_max_samples:
                # Strong penalty that scales with violation
                violation_ratio = merged_n / constraints.abs_max_samples
                score *= max(0.1, 1.0 / violation_ratio)
        
        # 4. Minimum positives (binary only)
        if self.is_binary_y and constraints.abs_min_positives > 0:
            a_positives = a.sum
            b_positives = b.sum
            
            if a_positives < constraints.abs_min_positives:
                score *= 1.4
            if b_positives < constraints.abs_min_positives:
                score *= 1.4
        
        return score


def merge_adjacent(
    blocks: Union[List[Block], List[Dict[str, Any]]],
    constraints: BinningConstraints,
    is_binary_y: bool,
    *,
    strategy: MergeStrategy = MergeStrategy.HIGHEST_PVALUE,
    history: Optional[List[List[Dict]]] = None,
    max_iterations: Optional[int] = None
) -> List[Block]:
    """Merge adjacent blocks using statistical tests and constraints.
    
    Main merging algorithm that:
    1. Greedily merges best-scoring adjacent pairs
    2. Respects max_bins constraint
    3. Enforces min_samples through a final sweep
    
    Args:
        blocks: Input blocks from PAVA (as Block objects or dicts).
        constraints: Resolved binning constraints.
        is_binary_y: Whether target is binary.
        strategy: Strategy for selecting merges.
        history: Optional list to append merge snapshots to.
        max_iterations: Maximum merge iterations (default: unlimited).
        
    Returns:
        List[Block]: Merged blocks satisfying constraints.
        
    Raises:
        FittingError: If merging produces invalid results.
        
    Examples:
        >>> blocks = pava.export_blocks(as_dict=True)
        >>> merged = merge_adjacent(blocks, constraints, is_binary_y=True)
        >>> print(f"Merged {len(blocks)} -> {len(merged)} blocks")
    """
    # Convert input to Block objects
    blocks_typed = as_blocks(blocks)
    
    if not blocks_typed:
        return []
    
    # Work on a copy
    current = list(blocks_typed)
    
    # Initialize scorer
    scorer = MergeScorer(constraints, is_binary_y, strategy)
    
    # Track iterations for convergence
    iteration = 0
    max_iter = max_iterations or constraints.max_bins * 100  # Reasonable default
    
    logger.info(
        f"Starting merge: {len(current)} blocks, "
        f"target range [{constraints.min_bins}, {constraints.max_bins}]"
    )
    
    # Phase 1: Statistical merging
    current = _statistical_merge_phase(
        current, constraints, scorer, history, max_iter
    )
    
    # Phase 2: Enforce minimum samples
    if constraints.abs_min_samples > 0:
        current = _enforce_min_samples(
            current, constraints, scorer, history
        )
    
    # Validate result
    if len(current) == 0:
        raise FittingError("Merging produced zero blocks")
    
    # Final validation
    _validate_merge_result(current, constraints)
    
    logger.info(f"Merge complete: {len(blocks_typed)} -> {len(current)} blocks")
    
    return current


def _statistical_merge_phase(
    blocks: List[Block],
    constraints: BinningConstraints,
    scorer: MergeScorer,
    history: Optional[List[List[Dict]]],
    max_iterations: int
) -> List[Block]:
    """Phase 1: Merge based on statistical similarity.
    
    Args:
        blocks: Current blocks.
        constraints: Binning constraints.
        scorer: Merge scorer.
        history: Optional history list.
        max_iterations: Maximum iterations.
        
    Returns:
        Merged blocks.
    """
    current = list(blocks)
    iteration = 0
    
    while iteration < max_iterations and len(current) > 1:
        # Check if we need to merge
        should_continue = False
        
        if constraints.maximize_bins:
            # Must stay at or below max_bins
            should_continue = len(current) > constraints.max_bins
        else:
            # Try to maintain min_bins unless scores are very high
            should_continue = True  # Will check scores below
        
        if not should_continue and constraints.maximize_bins:
            break
        
        # Find best merge candidate
        best_idx, best_score = _find_best_merge(current, scorer)
        
        if best_idx is None:
            break
        
        # Decide whether to merge
        should_merge = False
        
        if constraints.maximize_bins and len(current) > constraints.max_bins:
            # Must merge to satisfy max_bins
            should_merge = True
        elif best_score >= constraints.initial_pvalue:
            # Score exceeds threshold
            should_merge = True
        elif not constraints.maximize_bins and len(current) > constraints.min_bins:
            # In min_bins mode, merge if score is reasonable
            should_merge = best_score >= constraints.initial_pvalue * 0.5
        
        if should_merge:
            # Perform merge
            current = _merge_at(current, best_idx)
            
            if history is not None:
                history.append(_snapshot(current))
            
            logger.debug(
                f"Iteration {iteration}: Merged blocks {best_idx},{best_idx+1} "
                f"(score={best_score:.4f}), {len(current)} blocks remain"
            )
        else:
            # No more merges warranted
            break
        
        iteration += 1
    
    if iteration >= max_iterations:
        warnings.warn(
            f"Min-samples enforcement reached max iterations ({max_iterations})",
            UserWarning
        )
    
    # Log result
    still_undersized = sum(1 for b in current if b.n < constraints.abs_min_samples)
    if still_undersized > 0:
        logger.warning(
            f"Could not satisfy min_samples for {still_undersized} bins "
            f"(reached min_bins={constraints.min_bins} limit)"
        )
    
    return current


def _find_best_merge(
    blocks: List[Block],
    scorer: MergeScorer
) -> Tuple[Optional[int], float]:
    """Find the best adjacent pair to merge.
    
    Args:
        blocks: Current blocks.
        scorer: Merge scorer.
        
    Returns:
        Tuple of (index of first block to merge, score).
        Returns (None, -1) if no valid merges.
    """
    if len(blocks) < 2:
        return None, -1.0
    
    best_idx = None
    best_score = -1.0
    
    for i in range(len(blocks) - 1):
        score = scorer.score_pair(blocks[i], blocks[i + 1])
        
        if score > best_score:
            best_score = score
            best_idx = i
    
    return best_idx, best_score


def _merge_at(blocks: List[Block], idx: int) -> List[Block]:
    """Merge blocks at index idx and idx+1.
    
    Args:
        blocks: Current blocks.
        idx: Index of first block to merge.
        
    Returns:
        New list with merged blocks.
        
    Raises:
        IndexError: If idx is out of range.
    """
    if idx < 0 or idx >= len(blocks) - 1:
        raise IndexError(f"Invalid merge index {idx} for {len(blocks)} blocks")
    
    merged_block = blocks[idx].merge_with(blocks[idx + 1])
    
    # Build new list
    result = blocks[:idx] + [merged_block] + blocks[idx + 2:]
    
    return result


def _snapshot(blocks: Sequence[Block]) -> List[Dict]:
    """Create a snapshot of current blocks for history.
    
    Args:
        blocks: Current blocks.
        
    Returns:
        List of block dictionaries.
    """
    return [b.as_dict() for b in blocks]


def _validate_merge_result(blocks: List[Block], constraints: BinningConstraints) -> None:
    """Validate that merge result satisfies constraints.
    
    Args:
        blocks: Final blocks.
        constraints: Binning constraints.
        
    Raises:
        FittingError: If critical constraints are violated.
    """
    n_blocks = len(blocks)
    
    # Check bin count constraints
    if constraints.maximize_bins:
        if n_blocks > constraints.max_bins:
            raise FittingError(
                f"Merge failed: {n_blocks} blocks exceeds max_bins={constraints.max_bins}"
            )
    
    # Check for edge coverage
    if blocks:
        if not np.isneginf(blocks[0].left):
            logger.warning(
                f"First block does not start at -inf (starts at {blocks[0].left})"
            )
        if not np.isposinf(blocks[-1].right):
            logger.warning(
                f"Last block does not end at +inf (ends at {blocks[-1].right})"
            )
    
    # Report on constraint satisfaction
    undersized = [b for b in blocks if b.n < constraints.abs_min_samples]
    if undersized and n_blocks > constraints.min_bins:
        logger.warning(
            f"{len(undersized)} blocks have fewer than {constraints.abs_min_samples} samples"
        )
    
    oversized = [
        b for b in blocks 
        if constraints.abs_max_samples and b.n > constraints.abs_max_samples
    ]
    if oversized:
        logger.warning(
            f"{len(oversized)} blocks exceed max_samples={constraints.abs_max_samples}"
        )


# Helper functions for type conversion

def blocks_from_dicts(rows: Iterable[Dict[str, Any]]) -> List[Block]:
    """Convert dictionaries to Block objects.
    
    Args:
        rows: Iterable of dictionaries with block data.
        
    Returns:
        List of Block objects.
        
    Raises:
        KeyError: If required fields are missing.
        ValueError: If values cannot be converted to expected types.
        
    Examples:
        >>> dicts = [{'left': 0, 'right': 1, 'n': 10, ...}]
        >>> blocks = blocks_from_dicts(dicts)
    """
    blocks: List[Block] = []
    
    for i, row in enumerate(rows):
        try:
            # Handle both field naming conventions
            block = Block(
                left=float(row['left']),
                right=float(row['right']),
                n=int(row['n']),
                sum=float(row['sum']),
                sum2=float(row['sum2']),
                ymin=float(row.get('ymin', row.get('min', float('inf')))),
                ymax=float(row.get('ymax', row.get('max', float('-inf'))))
            )
            blocks.append(block)
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(
                f"Error converting row {i} to Block: {e}. "
                f"Row data: {row}"
            )
    
    return blocks


def as_blocks(rows: Union[List[Dict[str, Any]], List[Block]]) -> List[Block]:
    """Convert input to list of Block objects.
    
    Handles both dictionary and Block inputs for flexibility.
    
    Args:
        rows: List of dicts or Block objects.
        
    Returns:
        List of Block objects.
        
    Examples:
        >>> # From dicts
        >>> blocks = as_blocks([{'left': 0, 'right': 1, ...}])
        >>> 
        >>> # From Blocks (returns as-is)
        >>> blocks = as_blocks([Block(...)])
    """
    if not rows:
        return []
    
    # Check first element to determine type
    first = rows[0]
    
    if isinstance(first, Block):
        # Already Block objects
        return list(rows)  # type: ignore
    elif isinstance(first, dict):
        # Convert from dictionaries
        return blocks_from_dicts(rows)  # type: ignore
    else:
        raise TypeError(
            f"Expected list of Block or dict, got list of {type(first).__name__}"
        )


# Monotonicity validation utilities

def validate_monotonicity(
    blocks: List[Block],
    sign: str,
    tolerance: float = 1e-10
) -> bool:
    """Check if blocks satisfy monotonicity constraint.
    
    Args:
        blocks: Blocks to validate.
        sign: '+' for non-decreasing, '-' for non-increasing.
        tolerance: Numerical tolerance for comparisons.
        
    Returns:
        bool: True if monotonicity is satisfied.
        
    Examples:
        >>> blocks = merge_adjacent(...)
        >>> assert validate_monotonicity(blocks, '+'), "Not monotone!"
    """
    if len(blocks) <= 1:
        return True
    
    means = [b.mean for b in blocks]
    
    for i in range(1, len(means)):
        if sign == '+':
            # Check non-decreasing
            if means[i] < means[i-1] - tolerance:
                logger.error(
                    f"Monotonicity violation at {i}: "
                    f"{means[i]:.6f} < {means[i-1]:.6f}"
                )
                return False
        else:
            # Check non-increasing  
            if means[i] > means[i-1] + tolerance:
                logger.error(
                    f"Monotonicity violation at {i}: "
                    f"{means[i]:.6f} > {means[i-1]:.6f}"
                )
                return False
    
    return True


def get_merge_summary(
    original_blocks: List[Block],
    merged_blocks: List[Block]
) -> Dict[str, Any]:
    """Generate summary statistics about the merge process.
    
    Args:
        original_blocks: Blocks before merging.
        merged_blocks: Blocks after merging.
        
    Returns:
        Dict with merge statistics.
        
    Examples:
        >>> summary = get_merge_summary(pava_blocks, merged_blocks)
        >>> print(f"Compression: {summary['compression_ratio']:.2f}x")
    """
    if not original_blocks or not merged_blocks:
        return {
            'original_count': len(original_blocks),
            'merged_count': len(merged_blocks),
            'compression_ratio': 0.0,
            'merges_performed': 0
        }
    
    # Basic counts
    n_original = len(original_blocks)
    n_merged = len(merged_blocks)
    n_merges = n_original - n_merged
    
    # Size statistics
    original_sizes = [b.n for b in original_blocks]
    merged_sizes = [b.n for b in merged_blocks]
    
    # Mean statistics
    original_means = [b.mean for b in original_blocks]
    merged_means = [b.mean for b in merged_blocks]
    
    return {
        'original_count': n_original,
        'merged_count': n_merged,
        'merges_performed': n_merges,
        'compression_ratio': n_original / n_merged if n_merged > 0 else 0.0,
        
        'original_size_stats': {
            'min': min(original_sizes),
            'max': max(original_sizes),
            'mean': np.mean(original_sizes),
            'std': np.std(original_sizes)
        },
        
        'merged_size_stats': {
            'min': min(merged_sizes),
            'max': max(merged_sizes),
            'mean': np.mean(merged_sizes),
            'std': np.std(merged_sizes)
        },
        
        'original_mean_range': (min(original_means), max(original_means)),
        'merged_mean_range': (min(merged_means), max(merged_means)),
        
        'size_balance': min(merged_sizes) / max(merged_sizes) if merged_sizes else 0.0
    }


# Public API exports
__all__ = [
    'Block',
    'MergeStrategy',
    'MergeScorer',
    'merge_adjacent',
    'blocks_from_dicts',
    'as_blocks',
    'validate_monotonicity',
    'get_merge_summary'
]


def _enforce_min_samples(
    blocks: List[Block],
    constraints: BinningConstraints,
    scorer: MergeScorer,
    history: Optional[List[List[Dict]]]
) -> List[Block]:
    """Phase 2: Enforce minimum samples per bin.
    
    Args:
        blocks: Current blocks.
        constraints: Binning constraints.
        scorer: Merge scorer.
        history: Optional history list.
        
    Returns:
        Blocks with min_samples enforced where possible.
    """
    current = list(blocks)
    
    if constraints.abs_min_samples <= 0:
        return current
    
    logger.debug(f"Enforcing min_samples={constraints.abs_min_samples}")
    
    max_iterations = len(blocks) * 2  # Prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        # Stop if we hit min_bins
        if len(current) <= max(1, constraints.min_bins):
            break
        
        # Find undersized bins
        undersized = [
            i for i, b in enumerate(current)
            if b.n < constraints.abs_min_samples
        ]
        
        if not undersized:
            break
        
        # Process first undersized bin
        idx = undersized[0]
        
        # Determine merge direction
        if idx == 0:
            # First block - can only merge right
            merge_idx = 0
        elif idx == len(current) - 1:
            # Last block - can only merge left
            merge_idx = idx - 1
        else:
            # Middle block - choose better neighbor
            left_score = scorer.score_pair(current[idx-1], current[idx])
            right_score = scorer.score_pair(current[idx], current[idx+1])
            merge_idx = idx if right_score >= left_score else idx - 1
        
        # Perform merge
        current = _merge_at(current, merge_idx)
        
        if history is not None:
            history.append(_snapshot(current))
        
        logger.debug(
            f"Min-samples merge: blocks {merge_idx},{merge_idx+1}, "
            f"{len(current)} blocks remain"
        )
        
        iteration += 1
        if iteration >= max_iterations:
            warnings.warn(
                f"Merge phase reached maximum iterations ({max_iterations})",
                UserWarning
            )
    
    return current