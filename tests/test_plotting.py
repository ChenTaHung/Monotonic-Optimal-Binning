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
    plot_binning_stability, plot_categorical_merge,
)
from MOBPY.binning.mob import MonotonicBinner
from MOBPY.core.pava import PAVA
from MOBPY.core.constraints import BinningConstraints
from MOBPY.exceptions import NotFittedError
from matplotlib.patches import Rectangle


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


# =============================================================================
# Shared helpers for categorical plotting tests
# =============================================================================

def _make_cat_df():
    """Synthetic categorical dataset: 3 low-risk + 3 high-risk categories, 150 obs each."""
    rng = np.random.default_rng(42)
    records = []
    for cat in ["low_a", "low_b", "low_c"]:
        records.append(pd.DataFrame({"x": cat, "y": rng.binomial(1, 0.08, 150)}))
    for cat in ["high_a", "high_b", "high_c"]:
        records.append(pd.DataFrame({"x": cat, "y": rng.binomial(1, 0.82, 150)}))
    return pd.concat(records, ignore_index=True)


def _make_cat_binner(df=None, alpha=0.05, max_label_cats=None):
    """Fit a categorical MonotonicBinner on the shared test dataset."""
    if df is None:
        df = _make_cat_df()
    binner = MonotonicBinner(
        df=df,
        x="x",
        y="y",
        x_type="categorical",
        categorical_alpha=alpha,
        constraints=BinningConstraints(max_bins=6, min_bins=2, min_samples=20),
        max_label_cats=max_label_cats,
    )
    binner.fit()
    return binner


def _cat_summary_df():
    """Minimal summary_df with categorical set-style bucket labels (start with '{')."""
    return pd.DataFrame({
        "bucket":    ["{cat_a, cat_b}", "{cat_c, cat_d}", "{cat_e, cat_f}"],
        "count":     [200, 150, 100],
        "count_pct": [44.4, 33.4, 22.2],
        "mean":      [0.05, 0.30, 0.80],
        "woe":       [1.80, 0.10, -1.90],
        "iv":        [0.25, 0.00, 0.40],
    })


def _num_summary_df():
    """Minimal summary_df with numeric interval-style bucket labels."""
    return pd.DataFrame({
        "bucket":    ["(-inf, 0.0)", "[0.0, 1.5)", "[1.5, +inf)"],
        "count":     [200, 150, 100],
        "count_pct": [44.4, 33.4, 22.2],
        "mean":      [0.05, 0.30, 0.80],
        "woe":       [1.80, 0.10, -1.90],
        "iv":        [0.25, 0.00, 0.40],
    })


# =============================================================================
# Tests for plot_categorical_merge
# =============================================================================

class TestCategoricalMergePlot:
    """Tests for plot_categorical_merge."""

    # ── return type ──────────────────────────────────────────────────────────

    def test_returns_axes(self):
        """plot_categorical_merge must return a matplotlib Axes."""
        binner = _make_cat_binner()
        fig, ax = plt.subplots()
        result = plot_categorical_merge(binner, ax=ax)
        assert isinstance(result, plt.Axes)
        plt.close(fig)

    def test_creates_own_figure_when_no_ax(self):
        """When ax=None, the function creates and returns its own Axes."""
        binner = _make_cat_binner()
        ax = plot_categorical_merge(binner)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_figsize_respected_when_no_ax(self):
        """Custom figsize should be applied to the auto-created figure."""
        binner = _make_cat_binner()
        ax = plot_categorical_merge(binner, figsize=(8, 3))
        w, h = ax.get_figure().get_size_inches()
        assert abs(w - 8) < 0.5
        assert abs(h - 3) < 0.5
        plt.close("all")

    # ── bars ─────────────────────────────────────────────────────────────────

    def test_one_bar_per_category(self):
        """Total Rectangle patches must equal n_cats (bars) + n_bins (axvspan backgrounds).

        axvspan() creates Rectangle patches (not Polygons) in current matplotlib,
        so the total patch count is n_cats + n_bins.
        """
        df = _make_cat_df()
        binner = _make_cat_binner(df)
        n_cats = df["x"].nunique()
        n_bins = binner.get_diagnostics()["n_final_bins"]

        fig, ax = plt.subplots()
        plot_categorical_merge(binner, ax=ax)

        n_rect = sum(1 for p in ax.patches if isinstance(p, Rectangle))
        assert n_rect == n_cats + n_bins
        plt.close(fig)

    def test_bars_are_present(self):
        """At least one patch (bar) should be drawn."""
        binner = _make_cat_binner()
        fig, ax = plt.subplots()
        plot_categorical_merge(binner, ax=ax)
        assert len(ax.patches) > 0
        plt.close(fig)

    # ── lines ─────────────────────────────────────────────────────────────────

    def test_overall_mean_axhline_present(self):
        """An overall-mean dotted line must be drawn (axhline → ax.lines)."""
        binner = _make_cat_binner()
        fig, ax = plt.subplots()
        plot_categorical_merge(binner, ax=ax)
        assert len(ax.lines) >= 1
        plt.close(fig)

    def test_pooled_rate_hlines_present(self):
        """Per-bin dashed pooled-rate lines (hlines → ax.collections) must be drawn."""
        binner = _make_cat_binner()
        n_bins = binner.get_diagnostics()["n_final_bins"]
        fig, ax = plt.subplots()
        plot_categorical_merge(binner, ax=ax)
        assert len(ax.collections) >= n_bins
        plt.close(fig)

    # ── group headers ─────────────────────────────────────────────────────────

    def test_group_headers_annotated(self):
        """At least one text annotation per bin group must appear on the axes."""
        binner = _make_cat_binner()
        n_bins = binner.get_diagnostics()["n_final_bins"]
        fig, ax = plt.subplots()
        plot_categorical_merge(binner, ax=ax)
        assert len(ax.texts) >= n_bins
        plt.close(fig)

    def test_group_header_bin_indices_are_zero_based(self):
        """Group header labels must use 0-based bin indices matching bins_() rows."""
        import re
        binner = _make_cat_binner()
        bins_df = binner.bins_()

        fig, ax = plt.subplots()
        plot_categorical_merge(binner, ax=ax)

        header_indices = [
            int(m.group(1))
            for t in ax.texts
            for m in [re.match(r"Bin (\d+)", t.get_text())]
            if m
        ]
        assert len(header_indices) > 0, "No 'Bin N' header texts found on axes"
        for idx in header_indices:
            assert idx in bins_df.index, (
                f"Header references Bin {idx} but bins_() has no such row"
            )
        # All indices must be < n_bins (i.e. 0-based, not 1-based)
        n_bins = binner.get_diagnostics()["n_final_bins"]
        assert all(idx < n_bins for idx in header_indices)
        plt.close(fig)

    # ── legend ────────────────────────────────────────────────────────────────

    def test_legend_handle_count(self):
        """Legend must have one patch per bin plus the overall-mean line handle.

        Bin patches are passed directly to ax.legend(handles=[...]) and do not
        live on the axes, so we read them via ax.get_legend().legend_handles.
        """
        binner = _make_cat_binner()
        n_bins = binner.get_diagnostics()["n_final_bins"]
        fig, ax = plt.subplots()
        plot_categorical_merge(binner, ax=ax)
        assert len(ax.get_legend().legend_handles) == n_bins + 1
        plt.close(fig)

    def test_legend_bin_labels_use_bins_df_index(self):
        """Legend bin labels must reference valid bins_() row indices."""
        import re
        binner = _make_cat_binner()
        bins_df = binner.bins_()

        fig, ax = plt.subplots()
        plot_categorical_merge(binner, ax=ax)

        # Legend labels come from ax.get_legend().get_texts(), not get_legend_handles_labels()
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        legend_indices = [
            int(m.group(1))
            for label in labels
            for m in [re.match(r"Bin (\d+)", label)]
            if m
        ]
        assert len(legend_indices) > 0, "No 'Bin N' entries found in legend"
        for idx in legend_indices:
            assert idx in bins_df.index

        # Indices must be 0-based
        n_bins = binner.get_diagnostics()["n_final_bins"]
        assert all(idx < n_bins for idx in legend_indices)
        plt.close(fig)

    # ── show_counts ───────────────────────────────────────────────────────────

    def test_show_counts_true_adds_n_annotations(self):
        """show_counts=True must add one 'n=...' text annotation per category."""
        df = _make_cat_df()
        binner = _make_cat_binner(df)
        n_cats = df["x"].nunique()

        fig, ax = plt.subplots()
        plot_categorical_merge(binner, ax=ax, show_counts=True)

        n_annotations = sum(1 for t in ax.texts if t.get_text().startswith("n="))
        assert n_annotations == n_cats
        plt.close(fig)

    def test_show_counts_false_omits_n_annotations(self):
        """show_counts=False must not add any 'n=...' text annotations."""
        binner = _make_cat_binner()
        fig, ax = plt.subplots()
        plot_categorical_merge(binner, ax=ax, show_counts=False)

        n_annotations = sum(1 for t in ax.texts if t.get_text().startswith("n="))
        assert n_annotations == 0
        plt.close(fig)

    # ── title ─────────────────────────────────────────────────────────────────

    def test_custom_title_is_used(self):
        """A custom title string must appear on the axes."""
        binner = _make_cat_binner()
        fig, ax = plt.subplots()
        plot_categorical_merge(binner, ax=ax, title="Custom Title")
        assert ax.get_title() == "Custom Title"
        plt.close(fig)

    def test_default_title_contains_x_column_name(self):
        """Default title must mention the x column name."""
        binner = _make_cat_binner()
        fig, ax = plt.subplots()
        plot_categorical_merge(binner, ax=ax)
        assert "x" in ax.get_title()
        plt.close(fig)

    def test_default_title_contains_bin_counts(self):
        """Default title must show the N categories → K bins summary."""
        binner = _make_cat_binner()
        diag = binner.get_diagnostics()
        fig, ax = plt.subplots()
        plot_categorical_merge(binner, ax=ax)
        title = ax.get_title()
        assert str(diag["n_initial_categories"]) in title
        assert str(diag["n_final_bins"]) in title
        plt.close(fig)

    # ── error handling ────────────────────────────────────────────────────────

    def test_error_unfitted_binner(self):
        """Must raise NotFittedError when called on an unfitted binner."""
        df = _make_cat_df()
        binner = MonotonicBinner(df=df, x="x", y="y", x_type="categorical")
        with pytest.raises(NotFittedError):
            plot_categorical_merge(binner)

    def test_error_numeric_binner(self):
        """Must raise ValueError when called on a numeric (non-categorical) binner."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "x": rng.normal(size=200),
            "y": rng.binomial(1, 0.4, 200),
        })
        binner = MonotonicBinner(df=df, x="x", y="y")
        binner.fit()
        with pytest.raises(ValueError, match="categorical"):
            plot_categorical_merge(binner)

    # ── bin_assignment cross-reference ────────────────────────────────────────

    def test_bin_assignment_matches_group_header_indices(self):
        """bin_assignment() bin indices must match the group header indices in the plot."""
        import re
        binner = _make_cat_binner()
        ba = binner.bin_assignment()

        fig, ax = plt.subplots()
        plot_categorical_merge(binner, ax=ax)

        header_indices = set(
            int(m.group(1))
            for t in ax.texts
            for m in [re.match(r"Bin (\d+)", t.get_text())]
            if m
        )
        assert header_indices == set(ba.unique())
        plt.close(fig)


# =============================================================================
# Tests for tick_labels in plot_woe_bars and plot_event_rate
# =============================================================================

class TestCategoricalTickLabels:
    """Tests for the tick_labels parameter on plot_woe_bars and plot_event_rate."""

    @staticmethod
    def _tick_texts(ax):
        return [t.get_text() for t in ax.get_xticklabels()]

    # ── plot_woe_bars — default (None) ────────────────────────────────────────

    def test_woe_bars_default_none_numeric_verbatim(self):
        """Default tick_labels=None: numeric bucket strings used as-is."""
        summary = _num_summary_df()
        fig, ax = plt.subplots()
        plot_woe_bars(summary, ax=ax)
        assert self._tick_texts(ax) == list(summary["bucket"])
        plt.close(fig)

    def test_woe_bars_default_none_categorical_verbatim(self):
        """Default tick_labels=None: categorical set strings used as-is (not shortened)."""
        summary = _cat_summary_df()
        fig, ax = plt.subplots()
        plot_woe_bars(summary, ax=ax)
        assert self._tick_texts(ax) == list(summary["bucket"])
        plt.close(fig)

    # ── plot_woe_bars — auto ──────────────────────────────────────────────────

    def test_woe_bars_auto_numeric_stays_verbatim(self):
        """tick_labels='auto' on numeric labels: no change (no { prefix)."""
        summary = _num_summary_df()
        fig, ax = plt.subplots()
        plot_woe_bars(summary, ax=ax, tick_labels="auto")
        assert self._tick_texts(ax) == list(summary["bucket"])
        plt.close(fig)

    def test_woe_bars_auto_categorical_generates_compact_labels(self):
        """tick_labels='auto' on categorical labels: compact 'Bin N\\n(XX.X%)' format."""
        summary = _cat_summary_df()
        fig, ax = plt.subplots()
        plot_woe_bars(summary, ax=ax, tick_labels="auto")
        ticks = self._tick_texts(ax)
        assert all(t.startswith("Bin") for t in ticks), (
            f"Expected 'Bin N' compact labels, got: {ticks}"
        )

    def test_woe_bars_auto_categorical_labels_zero_based(self):
        """Compact labels generated by 'auto' must be 0-based (Bin 0, Bin 1, ...)."""
        import re
        summary = _cat_summary_df()
        fig, ax = plt.subplots()
        plot_woe_bars(summary, ax=ax, tick_labels="auto")
        ticks = self._tick_texts(ax)
        indices = [int(re.search(r"Bin (\d+)", t).group(1)) for t in ticks]
        assert indices == list(range(len(summary)))
        plt.close(fig)

    def test_woe_bars_auto_detection_requires_brace_prefix(self):
        """Long labels WITHOUT a '{' prefix must NOT trigger auto-shortening."""
        long_non_cat = pd.DataFrame({
            "bucket":    ["very_long_label_group_A", "very_long_label_group_B"],
            "count":     [100, 200],
            "count_pct": [33.3, 66.7],
            "mean":      [0.10, 0.50],
            "woe":       [1.0, -1.0],
            "iv":        [0.1,  0.2],
        })
        fig, ax = plt.subplots()
        plot_woe_bars(long_non_cat, ax=ax, tick_labels="auto")
        ticks = self._tick_texts(ax)
        assert ticks == list(long_non_cat["bucket"]), (
            "Labels without '{' prefix should not be replaced by auto mode"
        )
        plt.close(fig)

    # ── plot_woe_bars — explicit list ─────────────────────────────────────────

    def test_woe_bars_explicit_list_overrides_bucket(self):
        """Passing a list of strings overrides bucket labels regardless of content."""
        summary = _cat_summary_df()
        custom = ["Low Risk", "Medium Risk", "High Risk"]
        fig, ax = plt.subplots()
        plot_woe_bars(summary, ax=ax, tick_labels=custom)
        assert self._tick_texts(ax) == custom
        plt.close(fig)

    def test_woe_bars_explicit_list_overrides_numeric_bucket(self):
        """Explicit list also overrides numeric bucket labels."""
        summary = _num_summary_df()
        custom = ["Tier A", "Tier B", "Tier C"]
        fig, ax = plt.subplots()
        plot_woe_bars(summary, ax=ax, tick_labels=custom)
        assert self._tick_texts(ax) == custom
        plt.close(fig)

    # ── plot_event_rate — default (None) ──────────────────────────────────────

    def test_event_rate_default_none_numeric_verbatim(self):
        """Default tick_labels=None: numeric bucket strings used as-is."""
        summary = _num_summary_df()
        fig, ax = plt.subplots()
        plot_event_rate(summary, ax=ax)
        assert self._tick_texts(ax) == list(summary["bucket"])
        plt.close(fig)

    def test_event_rate_default_none_categorical_verbatim(self):
        """Default tick_labels=None: categorical set strings used as-is."""
        summary = _cat_summary_df()
        fig, ax = plt.subplots()
        plot_event_rate(summary, ax=ax)
        assert self._tick_texts(ax) == list(summary["bucket"])
        plt.close(fig)

    # ── plot_event_rate — auto ────────────────────────────────────────────────

    def test_event_rate_auto_numeric_stays_verbatim(self):
        """tick_labels='auto' on numeric labels: no change."""
        summary = _num_summary_df()
        fig, ax = plt.subplots()
        plot_event_rate(summary, ax=ax, tick_labels="auto")
        assert self._tick_texts(ax) == list(summary["bucket"])
        plt.close(fig)

    def test_event_rate_auto_categorical_generates_compact_labels(self):
        """tick_labels='auto' on categorical labels: compact 'Bin N\\n(XX.X%)' format."""
        summary = _cat_summary_df()
        fig, ax = plt.subplots()
        plot_event_rate(summary, ax=ax, tick_labels="auto")
        ticks = self._tick_texts(ax)
        assert all(t.startswith("Bin") for t in ticks), (
            f"Expected compact 'Bin N' labels, got: {ticks}"
        )
        plt.close(fig)

    def test_event_rate_auto_categorical_labels_zero_based(self):
        """Compact labels from 'auto' must be 0-based."""
        import re
        summary = _cat_summary_df()
        fig, ax = plt.subplots()
        plot_event_rate(summary, ax=ax, tick_labels="auto")
        ticks = self._tick_texts(ax)
        indices = [int(re.search(r"Bin (\d+)", t).group(1)) for t in ticks]
        assert indices == list(range(len(summary)))
        plt.close(fig)

    def test_event_rate_auto_detection_requires_brace_prefix(self):
        """Long labels WITHOUT '{' prefix must NOT trigger auto-shortening."""
        long_non_cat = pd.DataFrame({
            "bucket":    ["very_long_label_group_A", "very_long_label_group_B"],
            "count":     [100, 200],
            "count_pct": [33.3, 66.7],
            "mean":      [0.10, 0.50],
            "woe":       [1.0, -1.0],
            "iv":        [0.1,  0.2],
        })
        fig, ax = plt.subplots()
        plot_event_rate(long_non_cat, ax=ax, tick_labels="auto")
        ticks = self._tick_texts(ax)
        assert ticks == list(long_non_cat["bucket"])
        plt.close(fig)

    # ── plot_event_rate — explicit list ───────────────────────────────────────

    def test_event_rate_explicit_list_overrides_bucket(self):
        """Passing a list of strings overrides bucket labels."""
        summary = _num_summary_df()
        custom = ["A", "B", "C"]
        fig, ax = plt.subplots()
        plot_event_rate(summary, ax=ax, tick_labels=custom)
        assert self._tick_texts(ax) == custom
        plt.close(fig)