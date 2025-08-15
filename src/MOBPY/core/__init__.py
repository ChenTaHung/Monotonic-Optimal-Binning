"""Core algorithms and utilities for monotonic optimal binning.

This module contains the fundamental building blocks of the MOBPY library:
- PAVA (Pool-Adjacent-Violators Algorithm) for isotonic regression
- Constraint specification and resolution
- Statistical merging of adjacent blocks
- Utility functions for data handling and calculations

Main Components:
    BinningConstraints: Specifies and resolves constraints for the binning process.
                       Supports both fractional (0-1) and absolute specifications.
    
    PAVA: Implements the Pool-Adjacent-Violators Algorithm for creating
          monotonic blocks from grouped data.
    
    Block: Data structure representing a contiguous block with sufficient
           statistics for O(1) merge operations.
    
    merge_adjacent: Algorithm for merging adjacent blocks using statistical
                    tests while respecting constraints.
    
    MergeStrategy: Enum defining strategies for selecting blocks to merge
                   (highest_pvalue, smallest_loss, balanced_size).
    
    utils: Collection of helper functions for data partitioning, validation,
           and WoE/IV calculations.

Example:
    >>> from MOBPY.core import BinningConstraints, PAVA, merge_adjacent
    >>> 
    >>> # Define constraints
    >>> constraints = BinningConstraints(
    ...     max_bins=6,
    ...     min_samples=0.05,  # 5% of data per bin
    ...     min_positives=0.01  # 1% of positives per bin
    ... )
    >>> 
    >>> # Run PAVA
    >>> pava = PAVA(df=data, x='feature', y='target', sign='auto')
    >>> pava.fit()
    >>> 
    >>> # Merge blocks
    >>> blocks = pava.export_blocks(as_dict=False)
    >>> merged = merge_adjacent(blocks, constraints, is_binary_y=True)
"""

from .constraints import BinningConstraints
from .pava import PAVA
from .merge import Block, merge_adjacent, as_blocks, MergeStrategy, MergeScorer
from . import utils

__all__ = [
    # Constraints
    "BinningConstraints",
    
    # PAVA algorithm
    "PAVA",
    
    # Merging components
    "Block",
    "merge_adjacent",
    "as_blocks",
    "MergeStrategy",
    "MergeScorer",
    
    # Utilities module
    "utils",
]