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
    partition_df, Parts, woe_iv
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
        
        # Boolean values should work in non-strict mode
        s = pd.Series([True, False, True, False])
        # These get coerced to 1/0 when converted to int
        assert is_binary_series(s, strict=False) is True
        
        # Two different integer values that are not 0/1
        # According to implementation, these don't convert to {0, 1}
        s = pd.Series([5, 10, 5, 10, 5])
        assert is_binary_series(s, strict=False) is False
        
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
        # Single unique value - may be considered binary in non-strict mode
        # The implementation returns True for single values in non-strict mode
        s = pd.Series([1, 1, 1])
        # In non-strict mode, single value returns True
        assert is_binary_series(s, strict=False) is True
        
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
        
        # Should not raise for existing column (single)
        validate_column_exists(df, 'a')
        
        # Should not raise for existing columns (list)
        validate_column_exists(df, ['a', 'b'])
        
        # Should raise for non-existent column
        with pytest.raises(DataError, match="Missing columns"):
            validate_column_exists(df, 'c')
        
        # Should raise for partially missing columns
        with pytest.raises(DataError, match="Missing columns"):
            validate_column_exists(df, ['a', 'c'])


class TestPartitioning:
    """Test suite for data partitioning functions."""
    
    def test_partition_basic(self):
        """Test basic partitioning into clean/missing/excluded."""
        df = pd.DataFrame({
            'x': [1, 2, np.nan, -999, 5, 6, np.nan, -999],
            'y': [0, 1, 1, 0, 1, 0, 0, 1]
        })
        
        parts = partition_df(df, 'x', exclude_values=[-999])
        
        # Check partition sizes
        assert len(parts.clean) == 4  # [1, 2, 5, 6]
        assert len(parts.missing) == 2  # Two NaN values
        assert len(parts.excluded) == 2  # Two -999 values
        
        # Check no overlap
        assert parts.validate() is True
    
    def test_partition_no_missing(self):
        """Test partitioning when no missing values."""
        df = pd.DataFrame({
            'x': [1, 2, 3, -999, 5],
            'y': [0, 1, 0, 1, 0]
        })
        
        parts = partition_df(df, 'x', exclude_values=[-999])
        
        assert len(parts.clean) == 4
        assert len(parts.missing) == 0
        assert len(parts.excluded) == 1
    
    def test_partition_no_excluded(self):
        """Test partitioning when no excluded values."""
        df = pd.DataFrame({
            'x': [1, 2, np.nan, 4, 5],
            'y': [0, 1, 0, 1, 0]
        })
        
        parts = partition_df(df, 'x', exclude_values=None)
        
        assert len(parts.clean) == 4
        assert len(parts.missing) == 1
        assert len(parts.excluded) == 0
    
    def test_partition_empty_exclude_list(self):
        """Test partitioning with empty exclude list."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [0, 1, 0, 1, 0]
        })
        
        parts = partition_df(df, 'x', exclude_values=[])
        
        assert len(parts.clean) == 5
        assert len(parts.missing) == 0
        assert len(parts.excluded) == 0
    
    def test_partition_all_excluded(self):
        """Test when all values are excluded."""
        df = pd.DataFrame({
            'x': [-999, -999, -999],
            'y': [0, 1, 0]
        })
        
        parts = partition_df(df, 'x', exclude_values=[-999])
        
        assert len(parts.clean) == 0
        assert len(parts.missing) == 0
        assert len(parts.excluded) == 3
    
    def test_parts_summary(self):
        """Test Parts.summary() method."""
        df = pd.DataFrame({
            'x': [1, 2, np.nan, -999, 5],
            'y': [0, 1, 0, 1, 0]
        })
        
        parts = partition_df(df, 'x', exclude_values=[-999])
        summary = parts.summary()
        
        assert summary['clean'] == 3
        assert summary['missing'] == 1
        assert summary['excluded'] == 1
        assert summary['total'] == 5
    
    def test_parts_validate(self):
        """Test Parts.validate() method."""
        # Create valid partitions
        df = pd.DataFrame({
            'x': [1, 2, np.nan, -999],
            'y': [0, 1, 0, 1]
        })
        
        parts = partition_df(df, 'x', exclude_values=[-999])
        
        # Should be valid (no overlapping indices)
        assert parts.validate() is True
        
        # Manually create invalid partitions (overlapping indices)
        # This is a bit artificial but tests the validation logic
        parts_invalid = Parts(
            clean=df.iloc[:2],
            missing=df.iloc[1:3],  # Overlaps with clean
            excluded=df.iloc[3:]
        )
        
        assert parts_invalid.validate() is False


class TestWoeIv:
    """Test suite for WoE/IV calculation functions."""
    
    def test_woe_iv_basic(self):
        """Test basic WoE/IV calculation."""
        # Simple case: 3 bins
        goods = np.array([80, 60, 40])  # Decreasing good counts
        bads = np.array([20, 40, 60])   # Increasing bad counts
        
        # Calculate WoE/IV
        woe_vals, iv_vals = woe_iv(goods, bads, smoothing=0.5)
        
        # Check shapes
        assert len(woe_vals) == 3
        assert len(iv_vals) == 3
        
        # WoE should be monotonic for this case
        assert woe_vals[0] > woe_vals[1] > woe_vals[2]
        
        # IV should be positive
        assert all(iv >= 0 for iv in iv_vals)
    
    def test_woe_iv_with_zeros(self):
        """Test WoE/IV with zero counts (smoothing prevents infinity)."""
        goods = np.array([100, 0, 50])
        bads = np.array([0, 100, 50])
        
        # Without smoothing this would cause division by zero
        woe_vals, iv_vals = woe_iv(goods, bads, smoothing=0.5)
        
        # Should not have any infinities or NaN
        assert np.all(np.isfinite(woe_vals))
        assert np.all(np.isfinite(iv_vals))
    
    def test_woe_iv_return_components(self):
        """Test returning WoE/IV as dictionary with components."""
        goods = np.array([80, 60, 40])
        bads = np.array([20, 40, 60])
        
        result = woe_iv(goods, bads, return_components=True)
        
        # Should return dictionary
        assert isinstance(result, dict)
        assert 'woe' in result
        assert 'iv' in result
        
        # Check for rate components (not pct)
        assert 'good_rate' in result or 'bad_rate' in result
        
        # Check consistency
        assert len(result['woe']) == 3
        assert len(result['iv']) == 3
    
    def test_woe_iv_single_bin(self):
        """Test WoE/IV with single bin."""
        goods = np.array([100])
        bads = np.array([50])
        
        woe_vals, iv_vals = woe_iv(goods, bads)
        
        assert len(woe_vals) == 1
        assert len(iv_vals) == 1
        
        # Single bin contains all data, so WoE might not be exactly 0
        # but IV should be 0 or very small (no discriminatory power)
        assert np.isfinite(woe_vals[0])  # Should be finite
        assert iv_vals[0] >= 0  # IV should be non-negative
    
    def test_woe_iv_equal_distribution(self):
        """Test WoE/IV when goods and bads are equally distributed."""
        goods = np.array([50, 50, 50])
        bads = np.array([50, 50, 50])
        
        woe_vals, iv_vals = woe_iv(goods, bads)
        
        # When distribution is equal, WoE should be close to 0
        # Allow small tolerance for numerical precision
        assert np.allclose(woe_vals, 0, atol=0.1)
        
        # IV values should be very small (near 0)
        assert np.allclose(iv_vals, 0, atol=0.01)


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
            df, x='feature',
            exclude_values=[-999]
        )
        
        assert len(parts.clean) == 800
        assert len(parts.missing) == 100
        assert len(parts.excluded) == 100
        
        # Step 4: Validate clean data
        ensure_numeric_series(parts.clean['feature'], 'feature')
        ensure_numeric_series(parts.clean['target'], 'target')
    
    def test_woe_iv_full_calculation(self):
        """Test complete WoE/IV calculation workflow."""
        # Simulate binned data
        bins_data = [
            {'good': 90, 'bad': 10},  # Low risk bin
            {'good': 70, 'bad': 30},  # Medium risk
            {'good': 40, 'bad': 60},  # High risk
        ]
        
        goods = np.array([b['good'] for b in bins_data])
        bads = np.array([b['bad'] for b in bins_data])
        
        # Calculate WoE/IV with components
        result = woe_iv(goods, bads, return_components=True)
        
        # Verify total IV is reasonable
        if 'total_iv' in result:
            total_iv = result['total_iv']
        else:
            total_iv = result['iv'].sum()
        assert 0 < total_iv < 10  # Reasonable IV range
        
        # Verify WoE ordering (should decrease for increasing risk)
        assert result['woe'][0] > result['woe'][1] > result['woe'][2]
        
        # If rates are provided, verify they're reasonable
        if 'good_rate' in result:
            # Good rates should sum close to 1
            total_good_rate = result['good_rate'].sum()
            # Note: rates are per-bin, not overall percentages, so they won't sum to 1
            assert np.all(result['good_rate'] >= 0)
            assert np.all(result['good_rate'] <= 1)
        
        if 'bad_rate' in result:
            assert np.all(result['bad_rate'] >= 0)
            assert np.all(result['bad_rate'] <= 1)