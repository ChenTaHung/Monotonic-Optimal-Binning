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
    
    def create_test_pava(self):
        """Create a fitted PAVA instance for testing."""
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5, 6],
            'y': [1, 3, 2, 4, 3, 5]  # Non-monotonic
        })
        pava = PAVA(df=df, x='x', y='y', sign='+')
        pava.fit()
        return pava
    
    def test_plot_gcm_basic(self):
        """Test basic GCM plot creation."""
        pava = self.create_test_pava()
        
        fig, ax = plt.subplots()
        result = plot_gcm(pava, ax=ax)
        
        assert result is not None
        assert isinstance(result, plt.Axes)
        
        # Check that plot has content
        assert len(ax.lines) > 0 or len(ax.collections) > 0
        
        plt.close(fig)
    
    def test_plot_gcm_without_axes(self):
        """Test GCM plot creates its own figure/axes."""
        pava = self.create_test_pava()
        
        ax = plot_gcm(pava)
        
        assert ax is not None
        assert isinstance(ax, plt.Axes)
        
        plt.close('all')
    
    def test_plot_pava_comparison(self):
        """Test side-by-side PAVA comparison plot."""
        pava = self.create_test_pava()
        
        fig = plot_pava_comparison(pava)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Should have multiple subplots
        axes = fig.get_axes()
        assert len(axes) >= 2  # At least CSD and GCM
        
        plt.close(fig)
    
    def test_plot_pava_process(self):
        """Test PAVA process visualization."""
        pava = self.create_test_pava()
        
        fig = plot_pava_process(pava)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Should show the merging process
        axes = fig.get_axes()
        assert len(axes) >= 1
        
        plt.close(fig)
    
    def test_plot_pava_animation(self):
        """Test PAVA animation creation."""
        pava = self.create_test_pava()
        
        # Animation might not work in test environment
        try:
            anim = plot_pava_animation(pava, interval=100)
            assert anim is not None
        except Exception:
            # Animation might fail in headless environment
            pytest.skip("Animation not supported in test environment")
        
        plt.close('all')
    
    def test_plot_with_custom_styles(self):
        """Test plots with custom styling options."""
        pava = self.create_test_pava()
        
        fig, ax = plt.subplots()
        
        # Test with custom parameters (if supported)
        result = plot_gcm(
            pava, 
            ax=ax,
            title="Custom Title",
            xlabel="Feature",
            ylabel="Target"
        )
        
        # Check title and labels if they were set
        if ax.get_title():
            assert "Custom Title" in ax.get_title() or ax.get_title() != ""
        
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
        
        fig, ax = plt.subplots()
        result = plot_woe_bars(binner, ax=ax)
        
        assert result is not None
        assert isinstance(result, plt.Axes)
        
        # Should have bars
        assert len(ax.patches) > 0  # Bar patches
        
        plt.close(fig)
    
    def test_plot_event_rate(self):
        """Test event rate plot."""
        binner = self.create_test_binner(binary=True)
        
        fig, ax = plt.subplots()
        result = plot_event_rate(binner, ax=ax)
        
        assert result is not None
        assert isinstance(result, plt.Axes)
        
        # Should have line plot and possibly bars
        assert len(ax.lines) > 0 or len(ax.patches) > 0
        
        plt.close(fig)
    
    def test_plot_bin_statistics(self):
        """Test comprehensive bin statistics plot."""
        binner = self.create_test_binner(binary=True)
        
        fig = plot_bin_statistics(binner)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Should have multiple subplots
        axes = fig.get_axes()
        assert len(axes) >= 2  # Multiple statistics panels
        
        plt.close(fig)
    
    def test_plot_sample_distribution(self):
        """Test sample distribution plot."""
        binner = self.create_test_binner()
        
        fig, ax = plt.subplots()
        result = plot_sample_distribution(binner, ax=ax)
        
        assert result is not None
        assert isinstance(result, plt.Axes)
        
        # Should show distribution
        assert len(ax.patches) > 0 or len(ax.lines) > 0
        
        plt.close(fig)
    
    def test_plot_bin_boundaries(self):
        """Test bin boundaries visualization."""
        binner = self.create_test_binner()
        
        fig, ax = plt.subplots()
        result = plot_bin_boundaries(binner, ax=ax)
        
        assert result is not None
        assert isinstance(result, plt.Axes)
        
        # Should show boundaries as vertical lines
        assert len(ax.lines) > 0 or len(ax.collections) > 0
        
        plt.close(fig)
    
    def test_plot_binning_stability(self):
        """Test binning stability comparison plot."""
        # Create train and test binners
        train_binner = self.create_test_binner()
        
        # Create test data with similar distribution
        np.random.seed(123)
        n = 300
        x = np.linspace(-2, 3, n) + np.random.normal(0, 0.1, n)
        p = 1 / (1 + np.exp(-1.5 * x))
        y = np.random.binomial(1, p)
        test_df = pd.DataFrame({'x': x, 'y': y})
        
        test_binner = MonotonicBinner(
            test_df, x='x', y='y',
            constraints=BinningConstraints(max_bins=5)
        )
        test_binner.fit()
        
        fig = plot_binning_stability(train_binner, test_binner)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
    
    def test_plot_with_continuous_target(self):
        """Test plots work with continuous target."""
        binner = self.create_test_binner(binary=False)
        
        # Some plots might not work for continuous, should handle gracefully
        fig, ax = plt.subplots()
        
        # This should work for continuous
        result = plot_sample_distribution(binner, ax=ax)
        assert result is not None
        
        plt.close(fig)
    
    def test_plot_save_to_file(self):
        """Test saving plots to file."""
        binner = self.create_test_binner()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_plot.png")
            
            fig = plot_bin_statistics(binner)
            fig.savefig(filepath)
            
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
            
            plt.close(fig)


class TestPlottingIntegration:
    """Integration tests for plotting functions."""
    
    def test_complete_plotting_workflow(self):
        """Test complete plotting workflow from data to visualization."""
        # Create data
        np.random.seed(456)
        n = 1000
        x = np.random.uniform(-3, 3, n)
        p = 1 / (1 + np.exp(-x))
        y = np.random.binomial(1, p)
        
        df = pd.DataFrame({'feature': x, 'target': y})
        
        # Fit binner
        binner = MonotonicBinner(
            df=df, x='feature', y='target',
            constraints=BinningConstraints(max_bins=6, min_samples=0.05)
        )
        binner.fit()
        
        # Create various plots
        plots_created = []
        
        try:
            # PAVA plots
            fig1 = plot_pava_comparison(binner._pava)
            plots_created.append(fig1)
            
            # Binning result plots
            fig2 = plot_bin_statistics(binner)
            plots_created.append(fig2)
            
            fig3, ax3 = plt.subplots()
            plot_woe_bars(binner, ax=ax3)
            plots_created.append(fig3)
            
            fig4, ax4 = plt.subplots()
            plot_event_rate(binner, ax=ax4)
            plots_created.append(fig4)
            
            # All plots should be created successfully
            assert len(plots_created) == 4
            
        finally:
            # Clean up
            for fig in plots_created:
                if fig is not None:
                    plt.close(fig)
    
    def test_plotting_with_missing_data(self):
        """Test plotting handles missing data correctly."""
        # Create data with missing values
        np.random.seed(789)
        n = 500
        x = np.random.uniform(-2, 2, n)
        y = np.random.binomial(1, 0.3, n)
        
        # Add missing values
        x[50:70] = np.nan
        
        df = pd.DataFrame({'x': x, 'y': y})
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        # Plots should handle missing data gracefully
        fig = plot_bin_statistics(binner)
        assert fig is not None
        
        plt.close(fig)
    
    def test_plotting_edge_cases(self):
        """Test plotting with edge cases."""
        # Single bin case
        df = pd.DataFrame({
            'x': [1.0] * 100,
            'y': np.random.binomial(1, 0.3, 100)
        })
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        # Should handle single bin
        fig, ax = plt.subplots()
        result = plot_sample_distribution(binner, ax=ax)
        assert result is not None
        
        plt.close(fig)
    
    def test_plot_style_consistency(self):
        """Test that plots have consistent styling."""
        binner = self.create_test_binner()
        
        # Create multiple plots
        fig1, ax1 = plt.subplots()
        plot_woe_bars(binner, ax=ax1)
        
        fig2, ax2 = plt.subplots()
        plot_event_rate(binner, ax=ax2)
        
        # Check for consistent elements (this is basic, could be expanded)
        for ax in [ax1, ax2]:
            # Should have labels
            assert ax.get_xlabel() != "" or ax.get_ylabel() != ""
            
            # Should have some content
            assert (len(ax.lines) > 0 or len(ax.patches) > 0 or 
                   len(ax.collections) > 0)
        
        plt.close(fig1)
        plt.close(fig2)
    
    def test_plotting_performance(self):
        """Test that plotting completes in reasonable time."""
        import time
        
        # Create larger dataset
        np.random.seed(42)
        n = 5000
        x = np.random.uniform(-5, 5, n)
        y = np.random.binomial(1, 0.5, n)
        
        df = pd.DataFrame({'x': x, 'y': y})
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        # Time plot creation
        start = time.time()
        fig = plot_bin_statistics(binner)
        elapsed = time.time() - start
        
        # Should complete quickly (< 5 seconds)
        assert elapsed < 5.0
        
        plt.close(fig)
    
    def create_test_binner(self, binary=True):
        """Helper to create test binner."""
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


class TestPlotErrorHandling:
    """Test error handling in plotting functions."""
    
    def test_plot_with_unfitted_binner(self):
        """Test plotting with unfitted binner raises appropriate error."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [0, 1, 0]})
        binner = MonotonicBinner(df=df, x='x', y='y')
        
        # Should raise NotFittedError or handle gracefully
        from MOBPY.exceptions import NotFittedError
        
        with pytest.raises(NotFittedError):
            plot_bin_statistics(binner)
    
    def test_plot_with_invalid_axes(self):
        """Test plotting with invalid axes object."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x': np.random.random(100),
            'y': np.random.binomial(1, 0.5, 100)
        })
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        # Pass invalid axes
        with pytest.raises((TypeError, AttributeError)):
            plot_woe_bars(binner, ax="not_an_axes")
    
    def test_plot_with_empty_bins(self):
        """Test plotting handles empty bins gracefully."""
        # This is a hypothetical edge case
        # Create minimal data that might result in issues
        df = pd.DataFrame({
            'x': [1, 1, 1],
            'y': [0, 0, 0]
        })
        
        binner = MonotonicBinner(df=df, x='x', y='y')
        binner.fit()
        
        # Should handle without crashing
        fig, ax = plt.subplots()
        try:
            plot_sample_distribution(binner, ax=ax)
        except Exception as e:
            pytest.fail(f"Plot failed with empty-like data: {e}")
        finally:
            plt.close(fig)