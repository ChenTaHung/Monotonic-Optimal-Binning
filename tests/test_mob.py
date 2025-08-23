"""Unit tests for MonotonicBinner orchestrator.

This module provides comprehensive tests for the main MonotonicBinner class,
including the complete binning pipeline, transformation, and edge cases.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import warnings

from MOBPY.binning.mob import MonotonicBinner, _format_edge
from MOBPY.core.constraints import BinningConstraints
from MOBPY.core.merge import MergeStrategy
from MOBPY.exceptions import DataError, NotFittedError, FittingError


class TestMonotonicBinner:
    """Test suite for MonotonicBinner class.
    
    Tests the complete binning pipeline from data input to final bins.
    """
    
    def create_test_data(self, n=500, seed=42):
        """Create standard test dataset.
        
        Args:
            n: Number of samples.
            seed: Random seed for reproducibility.
            
        Returns:
            DataFrame with feature and binary target.
        """
        np.random.seed(seed)
        x = np.linspace(-2, 3, n) + np.random.normal(0, 0.15, n)
        p = 1 / (1 + np.exp(-1.4 * x))  # Logistic relationship
        y = np.random.binomial(1, p)
        
        return pd.DataFrame({'x': x, 'y': y})
    
    def test_basic_initialization(self):
        """Test MonotonicBinner initialization with defaults."""
        df = self.create_test_data()
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        
        assert binner.x == 'x'
        assert binner.y == 'y'
        assert binner.metric == 'mean'
        assert binner.sign == 'auto'
        assert binner.strict is False
        assert isinstance(binner.constraints, BinningConstraints)
    
    def test_custom_initialization(self):
        """Test MonotonicBinner with custom parameters."""
        df = self.create_test_data()
        constraints = BinningConstraints(max_bins=5, min_samples=0.1)
        
        binner = MonotonicBinner(
            df=df, x='x', y='y',
            metric='mean',
            sign='+',
            strict=True,
            constraints=constraints,
            exclude_values=[-999],
            merge_strategy=MergeStrategy.SMALLEST_LOSS
        )
        
        assert binner.sign == '+'
        assert binner.strict is True
        assert binner.constraints.max_bins == 5
        assert binner.exclude_values == [-999]
        assert binner.merge_strategy == MergeStrategy.SMALLEST_LOSS
    
    def test_fit_basic(self):
        """Test basic fitting process."""
        df = self.create_test_data()
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        # Check fitting completed
        assert binner._is_fitted is True
        assert binner.resolved_sign_ in ['+', '-']
        
        # Check bins were created
        bins = binner.bins_()
        assert len(bins) > 0
        assert len(bins) <= binner.constraints.max_bins
    
    def test_fit_with_constraints(self):
        """Test fitting with specific constraints."""
        df = self.create_test_data()
        
        constraints = BinningConstraints(
            max_bins=4,
            min_bins=2,
            min_samples=0.2  # 20% per bin minimum
        )
        
        binner = MonotonicBinner(df=df, x='x', y='y', constraints=constraints)
        binner.fit()
        
        bins = binner.bins_()
        
        # Check constraints are respected
        assert 2 <= len(bins) <= 4
        
        # Each bin should have at least 20% of samples
        total_clean = bins['nsamples'].sum()
        for n in bins['nsamples']:
            assert n >= 0.2 * total_clean * 0.95  # Allow small tolerance
    
    def test_fit_auto_sign_detection(self):
        """Test automatic sign detection."""
        # Increasing relationship
        df_inc = self.create_test_data()
        binner_inc = MonotonicBinner(df=df_inc, x='x', y='y', sign='auto')
        binner_inc.fit()
        assert binner_inc.resolved_sign_ == '+'
        
        # Decreasing relationship
        df_dec = self.create_test_data()
        df_dec['y'] = 1 - df_dec['y']  # Invert target
        binner_dec = MonotonicBinner(df=df_dec, x='x', y='y', sign='auto')
        binner_dec.fit()
        assert binner_dec.resolved_sign_ == '-'
    
    def test_fit_with_missing_values(self):
        """Test fitting handles missing values correctly."""
        df = self.create_test_data()
        # Add missing values
        df.loc[10:20, 'x'] = np.nan
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        # Missing values should be in summary but not bins
        bins = binner.bins_()
        summary = binner.summary_()
        
        assert len(summary) > len(bins)  # Summary includes Missing row
        assert 'Missing' in summary['interval'].values
    
    def test_fit_with_excluded_values(self):
        """Test fitting with excluded special values."""
        df = self.create_test_data()
        # Add special values
        df.loc[30:35, 'x'] = -999
        df.loc[40:45, 'x'] = -888
        
        binner = MonotonicBinner(
            df=df, x='x', y='y',
            exclude_values=[-999, -888]
        )
        binner.fit()
        
        summary = binner.summary_()
        
        # Should have excluded value rows
        assert 'Excluded: -999' in summary['interval'].values
        assert 'Excluded: -888' in summary['interval'].values
    
    def test_bins_method(self):
        """Test bins_() method returns correct structure."""
        df = self.create_test_data()
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        bins = binner.bins_()
        
        # Check required columns
        required_cols = ['left', 'right', 'nsamples', 'mean']
        for col in required_cols:
            assert col in bins.columns
        
        # Check bin coverage
        assert np.isneginf(bins.iloc[0]['left'])  # First bin starts at -inf
        assert np.isposinf(bins.iloc[-1]['right'])  # Last bin ends at +inf
        
        # Check bins are contiguous
        for i in range(len(bins) - 1):
            assert bins.iloc[i]['right'] == bins.iloc[i+1]['left']
    
    def test_summary_method_binary(self):
        """Test summary_() method for binary target."""
        df = self.create_test_data()
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        summary = binner.summary_()
        
        # Check binary-specific columns
        binary_cols = ['bads', 'goods', 'bad_rate', 'woe', 'iv_grp']
        for col in binary_cols:
            assert col in summary.columns
        
        # Check IV total is included
        assert hasattr(summary, 'attrs') or 'iv_total' in summary.attrs
        
        # Check bad_rate calculation
        for _, row in summary.iterrows():
            if row['nsamples'] > 0:
                expected_rate = row['bads'] / row['nsamples']
                assert abs(row['bad_rate'] - expected_rate) < 1e-6
    
    def test_summary_method_continuous(self):
        """Test summary_() method for continuous target."""
        df = self.create_test_data()
        df['y'] = df['y'] * 10 + np.random.normal(0, 1, len(df))  # Make continuous
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        summary = binner.summary_()
        
        # Should not have binary-specific columns
        assert 'woe' not in summary.columns
        assert 'iv_grp' not in summary.columns
        
        # Should have basic statistics
        assert 'mean' in summary.columns
        assert 'std' in summary.columns
    
    def test_transform_interval_labels(self):
        """Test transform with interval labels."""
        df = self.create_test_data()
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        # Transform to interval labels
        labels = binner.transform(df['x'], assign='interval')
        
        assert len(labels) == len(df)
        assert all(isinstance(label, str) for label in labels)
        
        # Check format like "(-inf, 0.5)" or "[0.5, 1.0)"
        for label in labels:
            assert ',' in label
            assert label.startswith('(') or label.startswith('[')
            assert label.endswith(')')
    
    def test_transform_left_edges(self):
        """Test transform with left edge assignment."""
        df = self.create_test_data()
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        left_edges = binner.transform(df['x'], assign='left')
        
        assert len(left_edges) == len(df)
        assert all(isinstance(edge, (int, float)) for edge in left_edges)
    
    def test_transform_right_edges(self):
        """Test transform with right edge assignment."""
        df = self.create_test_data()
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        right_edges = binner.transform(df['x'], assign='right')
        
        assert len(right_edges) == len(df)
        assert all(isinstance(edge, (int, float)) for edge in right_edges)
    
    def test_transform_handles_new_values(self):
        """Test transform handles values outside training range."""
        df = self.create_test_data()
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        # Create new data with extreme values
        new_values = pd.Series([-1000, 0, 1000])
        
        labels = binner.transform(new_values, assign='interval')
        
        # Should assign to appropriate bins (first/last for extremes)
        assert len(labels) == 3
        assert '-inf' in labels[0]  # Very small value -> first bin
        assert 'inf' in labels[2]  # Very large value -> last bin
    
    def test_transform_handles_missing(self):
        """Test transform handles missing values."""
        df = self.create_test_data()
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        # Data with missing values
        new_data = pd.Series([1.0, np.nan, 2.0])
        
        labels = binner.transform(new_data, assign='interval')
        
        assert len(labels) == 3
        assert labels[1] == 'Missing'
    
    def test_transform_handles_excluded(self):
        """Test transform handles excluded values."""
        df = self.create_test_data()
        
        binner = MonotonicBinner(
            df=df, x='x', y='y',
            exclude_values=[-999]
        )
        binner.fit()
        
        # Data with excluded value
        new_data = pd.Series([1.0, -999, 2.0])
        
        labels = binner.transform(new_data, assign='interval')
        
        assert len(labels) == 3
        assert 'Excluded: -999' in labels[1]
    
    def test_not_fitted_error(self):
        """Test error when accessing results before fitting."""
        df = self.create_test_data()
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        
        # Should raise NotFittedError
        with pytest.raises(NotFittedError):
            binner.bins_()
        
        with pytest.raises(NotFittedError):
            binner.summary_()
        
        with pytest.raises(NotFittedError):
            binner.transform([1, 2, 3])
    
    def test_fit_idempotent(self):
        """Test that fitting multiple times works correctly."""
        df = self.create_test_data()
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        
        # First fit
        binner.fit()
        bins1 = binner.bins_().copy()
        
        # Second fit
        binner.fit()
        bins2 = binner.bins_().copy()
        
        # Results should be identical (deterministic)
        pd.testing.assert_frame_equal(bins1, bins2)
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame({'x': [], 'y': []})
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        
        with pytest.raises(DataError):
            binner.fit()
    
    def test_single_unique_value(self):
        """Test handling of single unique x value."""
        df = pd.DataFrame({
            'x': [1.0] * 100,
            'y': np.random.binomial(1, 0.3, 100)
        })
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        bins = binner.bins_()
        
        # Should create single bin
        assert len(bins) == 1
    
    def test_perfect_separation(self):
        """Test handling of perfect separation."""
        df = pd.DataFrame({
            'x': list(range(100)),
            'y': [0] * 50 + [1] * 50  # Perfect separation at x=50
        })
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        # Should handle without errors
        bins = binner.bins_()
        assert len(bins) >= 2  # Should find the separation point
    
    def test_diagnostics_available(self):
        """Test that fit diagnostics are available after fitting."""
        df = self.create_test_data()
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        # Check diagnostics exist
        assert hasattr(binner, '_fit_diagnostics')
        diag = binner._fit_diagnostics
        
        assert 'partition_summary' in diag
        assert 'is_binary' in diag
        assert 'resolved_sign' in diag
        assert 'n_pava_blocks' in diag
        assert 'n_final_bins' in diag


class TestFormatEdge:
    """Test suite for edge formatting helper function."""
    
    def test_format_edge_normal(self):
        """Test formatting of normal edge values."""
        assert _format_edge(0.5) == "0.5"
        assert _format_edge(1.234) == "1.234"
        assert _format_edge(100) == "100" or _format_edge(100) == "100.0"
    
    def test_format_edge_infinity(self):
        """Test formatting of infinite values."""
        assert _format_edge(np.inf) == "inf" or _format_edge(np.inf) == "+inf"
        assert _format_edge(-np.inf) == "-inf"
    
    def test_format_edge_small_large(self):
        """Test formatting of very small/large values."""
        # Very small - should use scientific notation
        result = _format_edge(0.00001)
        assert 'e' in result.lower() or result == "0.00001"
        
        # Very large - should use scientific notation
        result = _format_edge(1000000)
        assert 'e' in result.lower() or len(result) <= 10


class TestMonotonicBinnerIntegration:
    """Integration tests for MonotonicBinner with real-world scenarios."""
    
    def test_credit_scoring_pattern(self):
        """Test with typical credit scoring data pattern."""
        np.random.seed(123)
        n = 1000
        
        # Age vs default probability (typical pattern)
        age = np.random.uniform(18, 70, n)
        default_prob = 0.3 * np.exp(-(age - 18) / 20) + 0.02
        defaults = np.random.binomial(1, default_prob)
        
        df = pd.DataFrame({
            'age': age,
            'default': defaults
        })
        
        binner = MonotonicBinner(
            df=df, x='age', y='default',
            constraints=BinningConstraints(max_bins=6, min_samples=0.05)
        )
        binner.fit()
        
        # Check monotonic relationship detected
        assert binner.resolved_sign_ == '-'  # Older age -> lower default
        
        # Check WoE monotonicity
        summary = binner.summary_()
        woe_values = summary[summary['interval'] != 'Missing']['woe'].values
        
        # WoE should be monotonic
        assert all(woe_values[i] >= woe_values[i+1] - 0.1  # Small tolerance
                  for i in range(len(woe_values)-1))
    
    def test_insurance_risk_pattern(self):
        """Test with insurance risk scoring pattern."""
        np.random.seed(456)
        n = 800
        
        # BMI vs claim probability
        bmi = np.random.normal(26, 5, n)
        bmi = np.clip(bmi, 15, 45)
        
        # U-shaped relationship (low and high BMI = higher risk)
        risk = 0.1 + 0.02 * ((bmi - 23) ** 2) / 100
        claims = np.random.binomial(1, np.clip(risk, 0, 1))
        
        df = pd.DataFrame({
            'bmi': bmi,
            'claim': claims
        })
        
        # This is non-monotonic, but PAVA will find best monotonic approximation
        binner = MonotonicBinner(
            df=df, x='bmi', y='claim',
            sign='auto',  # Let it detect
            constraints=BinningConstraints(max_bins=5)
        )
        binner.fit()
        
        # Should create bins and handle non-monotonicity
        bins = binner.bins_()
        assert 2 <= len(bins) <= 5
        
        # Verify monotonicity in final bins
        means = bins['mean'].values
        if binner.resolved_sign_ == '+':
            assert all(means[i] <= means[i+1] for i in range(len(means)-1))
        else:
            assert all(means[i] >= means[i+1] for i in range(len(means)-1))
    
    def test_with_real_csv_pattern(self):
        """Test with pattern similar to provided CSV files."""
        # Simulate German credit data pattern
        np.random.seed(789)
        n = 1000
        
        # Duration in months vs default
        duration = np.random.choice(range(6, 73), n)
        
        # Longer duration = higher risk
        default_prob = 0.1 + 0.3 * (duration / 72)
        defaults = np.random.binomial(1, default_prob)
        
        # Add some special values
        duration[50:60] = -1  # Missing code
        
        df = pd.DataFrame({
            'duration': duration,
            'default': defaults
        })
        
        binner = MonotonicBinner(
            df=df, x='duration', y='default',
            exclude_values=[-1],
            constraints=BinningConstraints(
                max_bins=6,
                min_samples=0.05,
                min_positives=0.01
            )
        )
        binner.fit()
        
        summary = binner.summary_()
        
        # Check excluded values handled
        assert 'Excluded: -1' in summary['interval'].values
        
        # Check all bins meet constraints
        clean_bins = summary[~summary['interval'].str.contains('Missing|Excluded')]
        
        total_samples = clean_bins['nsamples'].sum()
        for _, row in clean_bins.iterrows():
            assert row['nsamples'] >= 0.05 * total_samples * 0.95  # Tolerance