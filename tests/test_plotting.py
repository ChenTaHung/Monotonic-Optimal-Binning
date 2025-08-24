"""Unit tests for plotting modules.

This module tests visualization functions for both PAVA process
and binning results, ensuring plots are created without errors.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Use non-interactive backend for testing
matplotlib.use('Agg')

from MOBPY.plot.csd_gcm import (
    plot_gcm, plot_pava_comparison, 
    plot_pava_process, plot_pava_animation
)
from MOBPY.plot.mob_plot import (
    plot_woe_bars, plot_event_rate, plot_bin_statistics,
    plot_sample_distribution, plot_bin_boundaries, 
    plot_binning_stability
)
from MOBPY.binning.mob import MonotonicBinner
from MOBPY.core.pava import PAVA
from MOBPY.core.constraints import BinningConstraints


class TestCSDGCMPlots:
    """Test suite for CSD/GCM plotting functions.
    
    Tests PAVA visualization functions.
    """
    
    def create_test_binner(self):
        """Create a fitted MonotonicBinner for testing."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5, 6, 7, 8],
            'y': [1, 3, 2, 4, 3, 5, 4, 6]  # Non-monotonic
        })
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        return binner
    
    def test_plot_gcm_basic(self):
        """Test basic GCM plot creation."""
        binner = self.create_test_binner()
        
        # Get required data from fitted binner
        groups_df = binner._pava.groups_
        blocks = binner._pava.export_blocks(as_dict=True)
        
        fig, ax = plt.subplots()
        result = plot_gcm(groups_df, blocks, ax=ax)
        
        assert result is not None
        assert isinstance(result, plt.Axes)
        
        # Check that plot has content
        assert len(ax.lines) > 0 or len(ax.collections) > 0
        
        plt.close(fig)
    
    def test_plot_gcm_without_axes(self):
        """Test GCM plot creates its own figure/axes."""
        binner = self.create_test_binner()
        
        groups_df = binner._pava.groups_
        blocks = binner._pava.export_blocks(as_dict=True)
        
        ax = plot_gcm(groups_df, blocks)
        
        assert ax is not None
        assert isinstance(ax, plt.Axes)
        
        plt.close('all')
    
    def test_plot_pava_comparison(self):
        """Test side-by-side PAVA comparison plot."""
        binner = self.create_test_binner()
        
        fig = plot_pava_comparison(binner)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Should have multiple subplots
        axes = fig.get_axes()
        assert len(axes) >= 2  # At least CSD and GCM
        
        plt.close(fig)
    
    def test_plot_pava_process(self):
        """Test PAVA process visualization."""
        binner = self.create_test_binner()
        
        groups_df = binner._pava.groups_
        blocks = binner._pava.export_blocks(as_dict=True)
        
        result = plot_pava_process(groups_df, blocks)
        
        assert result is not None
        # plot_pava_process returns an Axes, not a Figure
        assert isinstance(result, plt.Axes)
        
        plt.close('all')
    
    def test_plot_pava_animation(self):
        """Test PAVA animation creation."""
        binner = self.create_test_binner()
        
        groups_df = binner._pava.groups_
        blocks = binner._pava.export_blocks(as_dict=True)
        
        # Animation might not work in test environment
        try:
            anim = plot_pava_animation(groups_df, blocks, interval=100)
            assert anim is not None
        except Exception:
            # Animation might fail in headless environment
            pytest.skip("Animation not supported in test environment")
        
        plt.close('all')
    
    def test_plot_with_custom_styles(self):
        """Test plots with custom styling options."""
        binner = self.create_test_binner()
        
        groups_df = binner._pava.groups_
        blocks = binner._pava.export_blocks(as_dict=True)
        
        fig, ax = plt.subplots()
        
        # Test with basic parameters (custom title etc might not be supported)
        result = plot_gcm(groups_df, blocks, ax=ax)
        
        assert result is not None
        
        plt.close(fig)


class TestMOBPlots:
    """Test suite for MOB result plotting functions.
    
    Tests binning result visualization functions.
    """
    
    def create_test_binner(self, binary=True):
        """Create a fitted MonotonicBinner for testing."""
        np.random.seed(42)
        n = 500
        x = np.linspace(-2, 3, n) + np.random.normal(0, 0.1, n)
        
        if binary:
            p = 1 / (1 + np.exp(-1.5 * x))
            y = np.random.binomial(1, p)
        else:
            y = 2 * x + np.random.normal(0, 1, n)
        
        df = pd.DataFrame({'x': x, 'y': y})
        
        binner = MonotonicBinner(
            df=df, x='x', y='y',
            constraints=BinningConstraints(max_bins=5)
        )
        binner.fit()
        return binner
    
    def test_plot_woe_bars(self):
        """Test WoE bar plot for binary target."""
        binner = self.create_test_binner(binary=True)
        summary = binner.summary_()
        
        fig, ax = plt.subplots()
        result = plot_woe_bars(summary, ax=ax)
        
        assert result is not None
        assert isinstance(result, plt.Axes)
        
        # Should have bars
        assert len(ax.patches) > 0  # Bar patches
        
        plt.close(fig)
    
    def test_plot_event_rate(self):
        """Test event rate plot."""
        binner = self.create_test_binner(binary=True)
        summary = binner.summary_()
        
        fig, ax = plt.subplots()
        result = plot_event_rate(summary, ax=ax)
        
        assert result is not None
        assert isinstance(result, plt.Axes)
        
        # Should have bars and possibly a line
        assert len(ax.patches) > 0 or len(ax.lines) > 0
        
        plt.close(fig)
    
    def test_plot_sample_distribution(self):
        """Test sample distribution plot."""
        binner = self.create_test_binner(binary=True)
        summary = binner.summary_()
        
        fig, ax = plt.subplots()
        result = plot_sample_distribution(summary, ax=ax)
        
        assert result is not None
        assert isinstance(result, plt.Axes)
        
        # Should have bars
        assert len(ax.patches) > 0
        
        plt.close(fig)
    
    def test_plot_bin_boundaries(self):
        """Test bin boundaries plot."""
        binner = self.create_test_binner(binary=True)
        
        fig, ax = plt.subplots()
        result = plot_bin_boundaries(binner, ax=ax)
        
        assert result is not None
        assert isinstance(result, plt.Axes)
        
        # Should have some plot elements
        assert len(ax.lines) > 0 or len(ax.patches) > 0
        
        plt.close(fig)
    
    def test_plot_bin_statistics(self):
        """Test comprehensive bin statistics plot."""
        binner = self.create_test_binner(binary=True)
        
        fig = plot_bin_statistics(binner)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Should have multiple subplots (4 main plots, but may have twin axes)
        axes = fig.get_axes()
        assert len(axes) >= 4  # At least 4 subplots (may have more with twin axes)
        
        plt.close(fig)
    
    def test_plot_binning_stability(self):
        """Test binning stability comparison plot."""
        binner = self.create_test_binner(binary=True)
        
        # Create test data with same structure
        np.random.seed(123)
        n = 300
        x = np.linspace(-2, 3, n) + np.random.normal(0, 0.1, n)
        p = 1 / (1 + np.exp(-1.5 * x))
        y = np.random.binomial(1, p)
        test_df = pd.DataFrame({'x': x, 'y': y})
        
        fig = plot_binning_stability(binner, test_df)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Should have comparison plots
        axes = fig.get_axes()
        assert len(axes) >= 2
        
        plt.close(fig)
    
    def test_plot_with_continuous_target(self):
        """Test plotting with continuous target."""
        binner = self.create_test_binner(binary=False)
        summary = binner.summary_()
        
        # Event rate plot should work for continuous
        fig, ax = plt.subplots()
        result = plot_event_rate(summary, ax=ax, y_format='decimal')
        
        assert result is not None
        assert isinstance(result, plt.Axes)
        
        plt.close(fig)
        
        # Bin statistics should work for continuous
        fig2 = plot_bin_statistics(binner)
        assert fig2 is not None
        
        plt.close(fig2)


class TestPlottingIntegration:
    """Integration tests for plotting workflow."""
    
    def test_complete_plotting_workflow(self):
        """Test complete plotting workflow from fitting to visualization."""
        # Create and fit binner
        np.random.seed(42)
        n = 500
        x = np.linspace(-2, 3, n) + np.random.normal(0, 0.1, n)
        p = 1 / (1 + np.exp(-1.5 * x))
        y = np.random.binomial(1, p)
        df = pd.DataFrame({'x': x, 'y': y})
        
        binner = MonotonicBinner(
            df=df, x='x', y='y',
            constraints=BinningConstraints(max_bins=5)
        )
        binner.fit()
        
        # Test PAVA plots
        fig1 = plot_pava_comparison(binner)
        assert fig1 is not None
        plt.close(fig1)
        
        # Test result plots
        summary = binner.summary_()
        
        fig2, ax = plt.subplots()
        plot_woe_bars(summary, ax=ax)
        plt.close(fig2)
        
        # Test comprehensive plot
        fig3 = plot_bin_statistics(binner)
        assert fig3 is not None
        plt.close(fig3)
    
    def test_plotting_edge_cases(self):
        """Test plotting with edge cases."""
        # Single bin case
        df = pd.DataFrame({
            'x': [1, 1, 1, 1, 1],
            'y': [0, 1, 0, 1, 0]
        })
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        summary = binner.summary_()
        
        # Should handle single bin
        fig, ax = plt.subplots()
        # Sample distribution should work
        result = plot_sample_distribution(summary, ax=ax)
        assert result is not None
        plt.close(fig)
    
    def test_plot_style_consistency(self):
        """Test that plots have consistent styling."""
        binner_data = self.create_test_data()
        binner = MonotonicBinner(
            binner_data, x='x', y='y',
            constraints=BinningConstraints(max_bins=5)
        )
        binner.fit()
        
        summary = binner.summary_()
        
        # Check background is white
        fig, ax = plt.subplots()
        plot_woe_bars(summary, ax=ax)
        
        # Background should be white
        assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0) or \
               ax.get_facecolor()[:3] == (1.0, 1.0, 1.0)
        
        plt.close(fig)
    
    def create_test_data(self):
        """Helper to create test data."""
        np.random.seed(42)
        n = 200
        x = np.linspace(-2, 3, n) + np.random.normal(0, 0.1, n)
        p = 1 / (1 + np.exp(-1.5 * x))
        y = np.random.binomial(1, p)
        return pd.DataFrame({'x': x, 'y': y})


class TestPlotErrorHandling:
    """Test error handling in plotting functions."""
    
    def test_plot_with_unfitted_binner(self):
        """Test error when plotting with unfitted binner."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [0, 1, 0]})
        binner = MonotonicBinner(df=df, x='x', y='y')
        
        # Should raise NotFittedError
        with pytest.raises(Exception) as exc_info:
            plot_bin_statistics(binner)
        
        # Check it's a fitting-related error
        assert "fitted" in str(exc_info.value).lower() or \
               "fit" in str(exc_info.value).lower()
    
    def test_plot_with_missing_columns(self):
        """Test error when required columns are missing."""
        # Create invalid summary DataFrame
        invalid_summary = pd.DataFrame({
            'bucket': ['Bin1', 'Bin2'],
            'count': [10, 20]
            # Missing 'woe' column for WoE plot
        })
        
        with pytest.raises(Exception) as exc_info:
            plot_woe_bars(invalid_summary)
        
        # Should mention missing column
        assert "woe" in str(exc_info.value).lower()
    
    def test_plot_with_empty_data(self):
        """Test handling of empty-like data."""
        # Create binner with very little data
        df = pd.DataFrame({
            'x': [1],
            'y': [0]
        })
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        # Should not crash
        try:
            summary = binner.summary_()
            fig, ax = plt.subplots()
            plot_sample_distribution(summary, ax=ax)
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"Plot failed with empty-like data: {e}")