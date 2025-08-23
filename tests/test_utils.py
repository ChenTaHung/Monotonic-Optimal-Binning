"""Unit tests for utility functions module.

This module tests helper functions for data validation, partitioning,
WoE/IV calculations, and other utilities.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import warnings

from MOBPY.core.utils import (
    ensure_numeric_series, is_binary_series, validate_column_exists,
    partition_df, Parts, woe_iv, calculate_correlation,
    safe_log, clip_values, format_number
)
from MOBPY.exceptions import DataError


class TestDataValidation:
    """Test suite for data validation functions."""
    
    def test_ensure_numeric_series_valid(self):
        """Test validation passes for valid numeric series."""
        s = pd.Series([1, 2, 3, 4, 5])
        
        # Should not raise
        ensure_numeric_series(s, "test_column")
    
    def test_ensure_numeric_series_with_nan(self):
        """Test validation allows NaN values."""
        s = pd.Series([1, 2, np.nan, 4, 5])
        
        # Should not raise - NaN is acceptable
        ensure_numeric_series(s, "test_column")
    
    def test_ensure_numeric_series_non_numeric(self):
        """Test validation fails for non-numeric data."""
        s = pd.Series(['a', 'b', 'c'])
        
        with pytest.raises(DataError, match="must be numeric"):
            ensure_numeric_series(s, "test_column")
    
    def test_ensure_numeric_series_infinite(self):
        """Test validation fails for infinite values."""
        s = pd.Series([1, 2, np.inf, 4, 5])
        
        with pytest.raises(DataError, match="non-finite"):
            ensure_numeric_series(s, "test_column")
        
        s = pd.Series([1, 2, -np.inf, 4, 5])
        
        with pytest.raises(DataError, match="non-finite"):
            ensure_numeric_series(s, "test_column")
    
    def test_ensure_numeric_series_empty(self):
        """Test validation handles empty series."""
        s = pd.Series([], dtype=float)
        
        # Should not raise but may warn
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ensure_numeric_series(s, "test_column")
    
    def test_ensure_numeric_series_all_nan(self):
        """Test validation handles all-NaN series."""
        s = pd.Series([np.nan, np.nan, np.nan])
        
        # Should warn but not raise
        with warnings.catch_warnings(record=True) as w:
            ensure_numeric_series(s, "test_column")
            # May have warning about only null values
    
    def test_is_binary_series_strict(self):
        """Test strict binary detection (exactly 0 and 1)."""
        # Valid binary
        s = pd.Series([0, 1, 0, 1, 1, 0])
        assert is_binary_series(s, strict=True) is True
        
        # Not strictly binary (has 2)
        s = pd.Series([0, 1, 2])
        assert is_binary_series(s, strict=True) is False
        
        # Not strictly binary (different values)
        s = pd.Series([1, 2])
        assert is_binary_series(s, strict=True) is False
    
    def test_is_binary_series_non_strict(self):
        """Test non-strict binary detection (any two values)."""
        # Standard 0/1
        s = pd.Series([0, 1, 0, 1])
        assert is_binary_series(s, strict=False) is True
        
        # Any two values
        s = pd.Series([5, 10, 5, 10, 5])
        assert is_binary_series(s, strict=False) is True
        
        # Three values - not binary
        s = pd.Series([1, 2, 3])
        assert is_binary_series(s, strict=False) is False
    
    def test_is_binary_series_with_nan(self):
        """Test binary detection with NaN values."""
        # Binary with NaN
        s = pd.Series([0, 1, np.nan, 0, 1])
        assert is_binary_series(s, strict=True) is True
        
        # Non-binary with NaN
        s = pd.Series([0, 1, 2, np.nan])
        assert is_binary_series(s, strict=True) is False
    
    def test_is_binary_series_edge_cases(self):
        """Test binary detection edge cases."""
        # Single unique value
        s = pd.Series([1, 1, 1])
        assert is_binary_series(s, strict=False) is False
        
        # Empty series
        s = pd.Series([], dtype=float)
        assert is_binary_series(s, strict=False) is False
        
        # All NaN
        s = pd.Series([np.nan, np.nan])
        assert is_binary_series(s, strict=False) is False
    
    def test_validate_column_exists(self):
        """Test column existence validation."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        
        # Should not raise for existing column
        validate_column_exists(df, 'a')
        validate_column_exists(df, 'b')
        
        # Should raise for non-existent column
        with pytest.raises(DataError, match="Column.*not found"):
            validate_column_exists(df, 'c')


class TestPartitioning:
    """Test suite for data partitioning functions."""
    
    def test_partition_basic(self):
        """Test basic partitioning into clean/missing/excluded."""
        df = pd.DataFrame({
            'x': [1, 2, np.nan, 4, 5, -999],
            'y': [10, 20, 30, 40, 50, 60]
        })
        
        parts = partition_df(
            df, x='x', y='y',
            exclude_values=[-999]
        )
        
        # Check partition sizes
        assert len(parts.clean) == 4  # Rows with x in [1,2,4,5]
        assert len(parts.missing) == 1  # Row with NaN
        assert len(parts.excluded) == 1  # Row with -999
    
    def test_partition_no_missing(self):
        """Test partitioning when no missing values."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })
        
        parts = partition_df(df, x='x', y='y')
        
        assert len(parts.clean) == 5
        assert len(parts.missing) == 0
        assert len(parts.excluded) == 0
    
    def test_partition_multiple_exclude_values(self):
        """Test partitioning with multiple exclude values."""
        df = pd.DataFrame({
            'x': [1, 2, -999, 4, -888, 6],
            'y': [10, 20, 30, 40, 50, 60]
        })
        
        parts = partition_df(
            df, x='x', y='y',
            exclude_values=[-999, -888]
        )
        
        assert len(parts.clean) == 4
        assert len(parts.excluded) == 2
    
    def test_partition_all_missing(self):
        """Test partitioning when all values are missing."""
        df = pd.DataFrame({
            'x': [np.nan, np.nan, np.nan],
            'y': [1, 2, 3]
        })
        
        parts = partition_df(df, x='x', y='y')
        
        assert len(parts.clean) == 0
        assert len(parts.missing) == 3
    
    def test_partition_preserves_indices(self):
        """Test that partitioning preserves original indices."""
        df = pd.DataFrame({
            'x': [1, np.nan, 3],
            'y': [10, 20, 30]
        }, index=[100, 200, 300])
        
        parts = partition_df(df, x='x', y='y')
        
        # Check indices are preserved
        assert list(parts.clean.index) == [100, 300]
        assert list(parts.missing.index) == [200]
    
    def test_parts_summary(self):
        """Test Parts.summary() method."""
        df = pd.DataFrame({
            'x': [1, 2, np.nan, 4, -999],
            'y': [10, 20, 30, 40, 50]
        })
        
        parts = partition_df(
            df, x='x', y='y',
            exclude_values=[-999]
        )
        
        summary = parts.summary()
        
        assert 'clean' in summary
        assert 'missing' in summary
        assert 'excluded' in summary
        assert summary['clean'] == 3
        assert summary['missing'] == 1
        assert summary['excluded'] == 1
        assert summary['total'] == 5
    
    def test_parts_is_empty(self):
        """Test Parts.is_empty property."""
        # Empty parts
        empty_parts = Parts(
            clean=pd.DataFrame(),
            missing=pd.DataFrame(),
            excluded=pd.DataFrame()
        )
        assert empty_parts.is_empty is True
        
        # Non-empty parts
        df = pd.DataFrame({'x': [1], 'y': [2]})
        non_empty_parts = Parts(
            clean=df,
            missing=pd.DataFrame(),
            excluded=pd.DataFrame()
        )
        assert non_empty_parts.is_empty is False


class TestWoEIVCalculations:
    """Test suite for Weight of Evidence and Information Value calculations."""
    
    def test_woe_iv_basic(self):
        """Test basic WoE/IV calculation."""
        # Create bins with known event rates
        bins = pd.DataFrame({
            'nsamples': [100, 100, 100],
            'bads': [10, 20, 30],  # Increasing bad rate
            'goods': [90, 80, 70]
        })
        
        result = woe_iv(bins)
        
        # Check structure
        assert 'woe' in result.columns
        assert 'iv_grp' in result.columns
        assert 'iv_total' in result.attrs or hasattr(result, 'iv_total')
        
        # WoE should be negative for low bad rates, positive for high
        assert result['woe'].iloc[0] < 0  # Low bad rate
        assert result['woe'].iloc[2] > 0  # High bad rate
    
    def test_woe_iv_edge_cases(self):
        """Test WoE/IV with edge cases."""
        # Bin with no bads
        bins = pd.DataFrame({
            'nsamples': [100, 100],
            'bads': [0, 50],
            'goods': [100, 50]
        })
        
        result = woe_iv(bins)
        
        # Should handle zero bads gracefully (with smoothing)
        assert not result['woe'].isna().any()
        assert not result['iv_grp'].isna().any()
    
    def test_woe_iv_perfect_separation(self):
        """Test WoE/IV with perfect separation."""
        # Perfect separation
        bins = pd.DataFrame({
            'nsamples': [100, 100],
            'bads': [100, 0],
            'goods': [0, 100]
        })
        
        result = woe_iv(bins)
        
        # Should handle with smoothing
        assert not np.isinf(result['woe']).any()
        assert not np.isinf(result['iv_grp']).any()
    
    def test_woe_iv_single_bin(self):
        """Test WoE/IV with single bin."""
        bins = pd.DataFrame({
            'nsamples': [200],
            'bads': [50],
            'goods': [150]
        })
        
        result = woe_iv(bins)
        
        # Single bin should have WoE close to 0
        assert abs(result['woe'].iloc[0]) < 0.1
        assert result['iv_grp'].iloc[0] >= 0


class TestStatisticalFunctions:
    """Test suite for statistical helper functions."""
    
    def test_calculate_correlation(self):
        """Test correlation calculation."""
        # Perfect positive correlation
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([2, 4, 6, 8, 10])
        
        corr = calculate_correlation(x, y)
        assert abs(corr - 1.0) < 1e-10
        
        # Perfect negative correlation
        y = pd.Series([10, 8, 6, 4, 2])
        corr = calculate_correlation(x, y)
        assert abs(corr - (-1.0)) < 1e-10
        
        # No correlation
        y = pd.Series([3, 1, 4, 1, 5])
        corr = calculate_correlation(x, y)
        assert abs(corr) < 0.5
    
    def test_calculate_correlation_with_nan(self):
        """Test correlation with missing values."""
        x = pd.Series([1, 2, np.nan, 4, 5])
        y = pd.Series([2, 4, 6, 8, 10])
        
        # Should handle NaN appropriately
        corr = calculate_correlation(x, y)
        assert not np.isnan(corr)
    
    def test_safe_log(self):
        """Test safe logarithm function."""
        # Normal values
        assert abs(safe_log(2.718) - 1.0) < 0.01
        
        # Zero should return large negative value
        assert safe_log(0) < -10
        
        # Negative should return NaN or raise
        result = safe_log(-1)
        assert np.isnan(result) or result < -10
    
    def test_clip_values(self):
        """Test value clipping function."""
        values = pd.Series([-10, -1, 0, 1, 10])
        
        # Clip to [0, 5]
        clipped = clip_values(values, min_val=0, max_val=5)
        
        assert clipped.min() >= 0
        assert clipped.max() <= 5
        assert list(clipped) == [0, 0, 0, 1, 5]
    
    def test_format_number(self):
        """Test number formatting for display."""
        # Small numbers
        assert format_number(0.0001) in ["0.0001", "1e-04", "1.0e-4"]
        
        # Large numbers  
        assert format_number(1000000) in ["1000000", "1e+06", "1.0e6"]
        
        # Normal numbers
        assert format_number(3.14159)[:4] == "3.14"
        
        # Integers
        assert format_number(42) in ["42", "42.0"]


class TestIntegrationScenarios:
    """Integration tests for utility functions working together."""
    
    def test_full_pipeline_utilities(self):
        """Test utilities in a typical pipeline scenario."""
        # Create test data
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'feature': np.concatenate([
                np.random.normal(0, 1, 800),  # Normal values
                [np.nan] * 100,  # Missing
                [-999] * 100  # Special code
            ]),
            'target': np.random.binomial(1, 0.3, n)
        })
        
        # Step 1: Validate columns exist
        validate_column_exists(df, 'feature')
        validate_column_exists(df, 'target')
        
        # Step 2: Check if target is binary
        assert is_binary_series(df['target'], strict=True)
        
        # Step 3: Partition data
        parts = partition_df(
            df, x='feature', y='target',
            exclude_values=[-999]
        )
        
        assert len(parts.clean) == 800
        assert len(parts.missing) == 100
        assert len(parts.excluded) == 100
        
        # Step 4: Validate clean data
        ensure_numeric_series(parts.clean['feature'], 'feature')
        ensure_numeric_series(parts.clean['target'], 'target')
        
        # Step 5: Check correlation
        corr = calculate_correlation(
            parts.clean['feature'],
            parts.clean['target']
        )
        assert -1 <= corr <= 1
    
    def test_woe_iv_full_calculation(self):
        """Test complete WoE/IV calculation workflow."""
        # Simulate binning results
        bins = pd.DataFrame({
            'left': [-np.inf, -1, 0, 1],
            'right': [-1, 0, 1, np.inf],
            'nsamples': [250, 250, 250, 250],
            'bads': [75, 50, 40, 35],  # Decreasing bad rate
            'goods': [175, 200, 210, 215]
        })
        
        # Calculate WoE/IV
        result = woe_iv(bins)
        
        # Verify all columns present
        expected_cols = ['left', 'right', 'nsamples', 'bads', 
                        'goods', 'woe', 'iv_grp']
        for col in expected_cols:
            assert col in result.columns
        
        # Verify WoE monotonicity (should decrease with bad rate)
        woe_values = result['woe'].values
        assert all(woe_values[i] >= woe_values[i+1] 
                  for i in range(len(woe_values)-1))
        
        # Verify IV values are non-negative
        assert (result['iv_grp'] >= 0).all()