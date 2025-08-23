"""Unit tests for block merging module.

This module tests the adjacent block merging algorithm including
statistical tests, constraint satisfaction, and merge strategies.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from typing import List, Dict

from MOBPY.core.merge import (
    Block, merge_adjacent, as_blocks, 
    MergeStrategy, MergeScorer, 
    blocks_from_dicts, _compute_pvalue
)
from MOBPY.core.constraints import BinningConstraints
from MOBPY.exceptions import FittingError


class TestBlock:
    """Test suite for Block dataclass.
    
    Tests block statistics, merge operations, and property calculations.
    """
    
    def test_block_initialization(self):
        """Test basic block creation."""
        block = Block(
            left=0.0,
            right=1.0,
            n=10,
            sum=50.0,
            sum2=300.0,
            ymin=3.0,
            ymax=7.0
        )
        
        assert block.left == 0.0
        assert block.right == 1.0
        assert block.n == 10
        assert block.sum == 50.0
        assert block.sum2 == 300.0
        assert block.ymin == 3.0
        assert block.ymax == 7.0
    
    def test_block_mean_property(self):
        """Test mean calculation."""
        block = Block(
            left=0.0, right=1.0,
            n=5, sum=15.0, sum2=50.0,
            ymin=2.0, ymax=4.0
        )
        
        assert block.mean == 3.0  # 15.0 / 5
    
    def test_block_variance_property(self):
        """Test variance calculation with numerical stability."""
        # Values: [1, 2, 3, 4, 5], mean=3, var=2.5
        block = Block(
            left=0.0, right=1.0,
            n=5, sum=15.0, sum2=55.0,
            ymin=1.0, ymax=5.0
        )
        
        expected_var = 2.5
        assert abs(block.var - expected_var) < 1e-10
    
    def test_block_std_property(self):
        """Test standard deviation calculation."""
        block = Block(
            left=0.0, right=1.0,
            n=5, sum=15.0, sum2=55.0,
            ymin=1.0, ymax=5.0
        )
        
        assert block.std == np.sqrt(block.var)
    
    def test_block_merge_statistics(self):
        """Test merging blocks pools statistics correctly."""
        block1 = Block(
            left=0.0, right=1.0,
            n=3, sum=6.0, sum2=14.0,
            ymin=1.0, ymax=3.0
        )
        
        block2 = Block(
            left=1.0, right=2.0,
            n=2, sum=7.0, sum2=25.0,
            ymin=3.0, ymax=4.0
        )
        
        merged = block1.merge(block2)
        
        assert merged.left == 0.0
        assert merged.right == 2.0
        assert merged.n == 5
        assert merged.sum == 13.0
        assert merged.sum2 == 39.0
        assert merged.ymin == 1.0
        assert merged.ymax == 4.0
    
    def test_block_merge_history_tracking(self):
        """Test merge history is tracked."""
        block1 = Block(
            left=0.0, right=1.0,
            n=5, sum=10.0, sum2=25.0,
            ymin=1.0, ymax=3.0
        )
        
        block2 = Block(
            left=1.0, right=2.0,
            n=5, sum=15.0, sum2=50.0,
            ymin=2.0, ymax=4.0
        )
        
        merged = block1.merge(block2)
        
        # Should track merge in history
        assert len(merged.merge_history) > 0
        assert (0.0, 2.0) in merged.merge_history or \
               (block1.left, block2.right) in merged.merge_history
    
    def test_block_empty_merge(self):
        """Test merging with empty block."""
        normal_block = Block(
            left=0.0, right=1.0,
            n=5, sum=10.0, sum2=25.0,
            ymin=1.0, ymax=3.0
        )
        
        empty_block = Block(
            left=1.0, right=2.0,
            n=0, sum=0.0, sum2=0.0,
            ymin=float('inf'), ymax=float('-inf')
        )
        
        merged = normal_block.merge(empty_block)
        
        # Should preserve normal block's statistics
        assert merged.n == 5
        assert merged.sum == 10.0
        assert merged.sum2 == 25.0


class TestMergeAdjacent:
    """Test suite for merge_adjacent function.
    
    Tests the main merging algorithm with various strategies and constraints.
    """
    
    def _create_test_blocks(self) -> List[Dict]:
        """Create standard test blocks for merging tests.
        
        Returns:
            List of block dictionaries with known statistics.
        """
        return [
            {'left': 0.0, 'right': 1.0, 'n': 10, 'sum': 10.0, 
             'sum2': 12.0, 'ymin': 0.8, 'ymax': 1.2},
            {'left': 1.0, 'right': 2.0, 'n': 10, 'sum': 20.0, 
             'sum2': 42.0, 'ymin': 1.8, 'ymax': 2.2},
            {'left': 2.0, 'right': 3.0, 'n': 10, 'sum': 30.0, 
             'sum2': 92.0, 'ymin': 2.8, 'ymax': 3.2},
        ]
    
    def test_merge_no_constraints(self):
        """Test merging without constraints merges everything."""
        blocks = self._create_test_blocks()
        
        constraints = BinningConstraints(
            max_bins=1,  # Force merge to single bin
            min_bins=1
        )
        constraints.resolve(total_n=30, total_pos=15)
        
        merged = merge_adjacent(blocks, constraints, is_binary_y=False)
        
        # Should merge to single block
        assert len(merged) == 1
        assert merged[0].n == 30
    
    def test_merge_respects_max_bins(self):
        """Test merging respects maximum bins constraint."""
        blocks = self._create_test_blocks()
        
        constraints = BinningConstraints(
            max_bins=2,
            min_bins=1
        )
        constraints.resolve(total_n=30, total_pos=15)
        
        merged = merge_adjacent(blocks, constraints, is_binary_y=False)
        
        # Should have at most 2 bins
        assert len(merged) <= 2
    
    def test_merge_respects_min_samples(self):
        """Test merging respects minimum samples constraint."""
        # Create blocks with varying sizes
        blocks = [
            {'left': 0.0, 'right': 1.0, 'n': 5, 'sum': 5.0, 
             'sum2': 6.0, 'ymin': 0.8, 'ymax': 1.2},
            {'left': 1.0, 'right': 2.0, 'n': 3, 'sum': 6.0,  # Small block
             'sum2': 13.0, 'ymin': 1.8, 'ymax': 2.2},
            {'left': 2.0, 'right': 3.0, 'n': 20, 'sum': 60.0, 
             'sum2': 182.0, 'ymin': 2.8, 'ymax': 3.2},
        ]
        
        constraints = BinningConstraints(
            max_bins=10,
            min_bins=1,
            min_samples=10  # Absolute minimum
        )
        constraints.resolve(total_n=28, total_pos=14)
        
        merged = merge_adjacent(blocks, constraints, is_binary_y=False)
        
        # All blocks should have at least 10 samples
        for block in merged:
            assert block.n >= 10
    
    def test_merge_strategy_highest_pvalue(self):
        """Test highest p-value merge strategy."""
        blocks = self._create_test_blocks()
        
        constraints = BinningConstraints(
            max_bins=2,
            min_bins=1
        )
        constraints.resolve(total_n=30, total_pos=15)
        
        merged = merge_adjacent(
            blocks, constraints, 
            is_binary_y=False,
            strategy=MergeStrategy.HIGHEST_PVALUE
        )
        
        assert len(merged) <= 2
    
    def test_merge_strategy_smallest_loss(self):
        """Test smallest loss merge strategy."""
        blocks = self._create_test_blocks()
        
        constraints = BinningConstraints(
            max_bins=2,
            min_bins=1
        )
        constraints.resolve(total_n=30, total_pos=15)
        
        merged = merge_adjacent(
            blocks, constraints,
            is_binary_y=False,
            strategy=MergeStrategy.SMALLEST_LOSS
        )
        
        assert len(merged) <= 2
    
    def test_merge_strategy_balanced_size(self):
        """Test balanced size merge strategy."""
        # Create unbalanced blocks
        blocks = [
            {'left': 0.0, 'right': 1.0, 'n': 100, 'sum': 100.0,
             'sum2': 102.0, 'ymin': 0.8, 'ymax': 1.2},
            {'left': 1.0, 'right': 2.0, 'n': 5, 'sum': 10.0,  # Very small
             'sum2': 21.0, 'ymin': 1.8, 'ymax': 2.2},
            {'left': 2.0, 'right': 3.0, 'n': 95, 'sum': 285.0,
             'sum2': 857.0, 'ymin': 2.8, 'ymax': 3.2},
        ]
        
        constraints = BinningConstraints(
            max_bins=2,
            min_bins=1
        )
        constraints.resolve(total_n=200, total_pos=100)
        
        merged = merge_adjacent(
            blocks, constraints,
            is_binary_y=False,
            strategy=MergeStrategy.BALANCED_SIZE
        )
        
        # Should prefer merging the small block
        assert len(merged) <= 2
    
    def test_merge_with_history_tracking(self):
        """Test merge history is recorded."""
        blocks = self._create_test_blocks()
        history = []
        
        constraints = BinningConstraints(
            max_bins=2,
            min_bins=1
        )
        constraints.resolve(total_n=30, total_pos=15)
        
        merged = merge_adjacent(
            blocks, constraints,
            is_binary_y=False,
            history=history
        )
        
        # History should be populated
        assert len(history) > 0
        
        # Each history entry should be a list of blocks
        assert all(isinstance(h, list) for h in history)
    
    def test_merge_binary_target(self):
        """Test merging with binary target variable."""
        # Binary blocks (0s and 1s)
        blocks = [
            {'left': 0.0, 'right': 1.0, 'n': 10, 'sum': 2.0,  # 20% event rate
             'sum2': 2.0, 'ymin': 0.0, 'ymax': 1.0},
            {'left': 1.0, 'right': 2.0, 'n': 10, 'sum': 5.0,  # 50% event rate
             'sum2': 5.0, 'ymin': 0.0, 'ymax': 1.0},
            {'left': 2.0, 'right': 3.0, 'n': 10, 'sum': 8.0,  # 80% event rate
             'sum2': 8.0, 'ymin': 0.0, 'ymax': 1.0},
        ]
        
        constraints = BinningConstraints(
            max_bins=2,
            min_bins=1,
            min_positives=3  # Minimum positives per bin
        )
        constraints.resolve(total_n=30, total_pos=15)
        
        merged = merge_adjacent(blocks, constraints, is_binary_y=True)
        
        # Should respect min_positives constraint
        for block in merged:
            n_positives = block.sum  # For binary, sum = count of 1s
            assert n_positives >= 3
    
    def test_merge_single_block(self):
        """Test merging with single input block."""
        blocks = [
            {'left': 0.0, 'right': 1.0, 'n': 100, 'sum': 50.0,
             'sum2': 30.0, 'ymin': 0.0, 'ymax': 1.0}
        ]
        
        constraints = BinningConstraints()
        constraints.resolve(total_n=100, total_pos=50)
        
        merged = merge_adjacent(blocks, constraints, is_binary_y=False)
        
        # Should return single block unchanged
        assert len(merged) == 1
        assert merged[0].n == 100
    
    def test_merge_empty_blocks_list(self):
        """Test merging with empty blocks list."""
        blocks = []
        
        constraints = BinningConstraints()
        constraints.resolve(total_n=0, total_pos=0)
        
        merged = merge_adjacent(blocks, constraints, is_binary_y=False)
        
        # Should return empty list
        assert len(merged) == 0
    
    def test_merge_maximize_bins_mode(self):
        """Test maximize_bins mode tries to keep more bins."""
        blocks = self._create_test_blocks()
        
        # Test with maximize_bins=True
        constraints_max = BinningConstraints(
            max_bins=3,
            min_bins=2,
            maximize_bins=True
        )
        constraints_max.resolve(total_n=30, total_pos=15)
        
        merged_max = merge_adjacent(
            blocks, constraints_max,
            is_binary_y=False
        )
        
        # Test with maximize_bins=False
        constraints_min = BinningConstraints(
            max_bins=3,
            min_bins=2,
            maximize_bins=False
        )
        constraints_min.resolve(total_n=30, total_pos=15)
        
        merged_min = merge_adjacent(
            blocks, constraints_min,
            is_binary_y=False
        )
        
        # maximize_bins=True should try to keep more bins
        assert len(merged_max) >= len(merged_min)


class TestMergeHelperFunctions:
    """Test suite for merge module helper functions."""
    
    def test_as_blocks_conversion(self):
        """Test as_blocks converts various input formats."""
        # Test with dictionaries
        dicts = [
            {'left': 0.0, 'right': 1.0, 'n': 10, 'sum': 10.0,
             'sum2': 12.0, 'ymin': 0.8, 'ymax': 1.2}
        ]
        
        blocks = as_blocks(dicts)
        assert len(blocks) == 1
        assert isinstance(blocks[0], Block)
        assert blocks[0].n == 10
        
        # Test with Block objects (should pass through)
        block_objs = [Block(
            left=0.0, right=1.0, n=10, sum=10.0,
            sum2=12.0, ymin=0.8, ymax=1.2
        )]
        
        blocks = as_blocks(block_objs)
        assert len(blocks) == 1
        assert isinstance(blocks[0], Block)
    
    def test_blocks_from_dicts(self):
        """Test blocks_from_dicts conversion function."""
        dicts = [
            {'left': 0.0, 'right': 1.0, 'n': 5, 'sum': 5.0,
             'sum2': 6.0, 'ymin': 0.8, 'ymax': 1.2},
            {'left': 1.0, 'right': 2.0, 'n': 5, 'sum': 10.0,
             'sum2': 21.0, 'ymin': 1.8, 'ymax': 2.2}
        ]
        
        blocks = blocks_from_dicts(dicts)
        
        assert len(blocks) == 2
        assert all(isinstance(b, Block) for b in blocks)
        assert blocks[0].mean == 1.0
        assert blocks[1].mean == 2.0
    
    def test_compute_pvalue_continuous(self):
        """Test p-value computation for continuous data."""
        # Create blocks with different means
        block1 = Block(
            left=0.0, right=1.0, n=100, sum=100.0,
            sum2=110.0, ymin=0.8, ymax=1.2
        )
        
        block2 = Block(
            left=1.0, right=2.0, n=100, sum=200.0,
            sum2=410.0, ymin=1.8, ymax=2.2
        )
        
        pvalue = _compute_pvalue(block1, block2, is_binary=False)
        
        # Should return valid p-value
        assert 0 <= pvalue <= 1
        
        # Different means should give low p-value
        assert pvalue < 0.05
    
    def test_compute_pvalue_binary(self):
        """Test p-value computation for binary data."""
        # Binary blocks with different proportions
        block1 = Block(
            left=0.0, right=1.0, n=100, sum=20.0,  # 20% rate
            sum2=20.0, ymin=0.0, ymax=1.0
        )
        
        block2 = Block(
            left=1.0, right=2.0, n=100, sum=80.0,  # 80% rate
            sum2=80.0, ymin=0.0, ymax=1.0
        )
        
        pvalue = _compute_pvalue(block1, block2, is_binary=True)
        
        # Should return valid p-value
        assert 0 <= pvalue <= 1
        
        # Very different proportions should give very low p-value
        assert pvalue < 0.001
    
    def test_compute_pvalue_edge_cases(self):
        """Test p-value computation edge cases."""
        # Identical blocks should give high p-value
        block1 = Block(
            left=0.0, right=1.0, n=100, sum=100.0,
            sum2=102.0, ymin=0.98, ymax=1.02
        )
        
        block2 = Block(
            left=1.0, right=2.0, n=100, sum=100.0,
            sum2=102.0, ymin=0.98, ymax=1.02
        )
        
        pvalue = _compute_pvalue(block1, block2, is_binary=False)
        
        # Very similar blocks should give high p-value
        assert pvalue > 0.5
    
    def test_merge_scorer_initialization(self):
        """Test MergeScorer class initialization."""
        blocks = [
            Block(left=0.0, right=1.0, n=10, sum=10.0,
                  sum2=12.0, ymin=0.8, ymax=1.2),
            Block(left=1.0, right=2.0, n=10, sum=20.0,
                  sum2=42.0, ymin=1.8, ymax=2.2)
        ]
        
        scorer = MergeScorer(
            blocks=blocks,
            is_binary=False,
            strategy=MergeStrategy.HIGHEST_PVALUE
        )
        
        assert scorer.n_blocks == 2
        assert scorer.strategy == MergeStrategy.HIGHEST_PVALUE
    
    def test_merge_scorer_compute_scores(self):
        """Test MergeScorer score computation."""
        blocks = [
            Block(left=0.0, right=1.0, n=10, sum=10.0,
                  sum2=12.0, ymin=0.8, ymax=1.2),
            Block(left=1.0, right=2.0, n=10, sum=15.0,
                  sum2=24.0, ymin=1.3, ymax=1.7),
            Block(left=2.0, right=3.0, n=10, sum=30.0,
                  sum2=92.0, ymin=2.8, ymax=3.2)
        ]
        
        scorer = MergeScorer(
            blocks=blocks,
            is_binary=False,
            strategy=MergeStrategy.HIGHEST_PVALUE
        )
        
        scores = scorer.compute_scores()
        
        # Should have n-1 scores for adjacent pairs
        assert len(scores) == 2
        
        # Scores should be valid
        assert all(isinstance(s, (int, float)) for s in scores)