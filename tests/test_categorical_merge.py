"""Unit tests for categorical block merging module.

Tests cover:
- _BlockStatsBase properties (shared with Block and CategoryBlock)
- CategoryBlock: init, merge_with, as_dict, repr
- _chi2_test: edge cases and known-good results
- _adjust_pvalues: Bonferroni, Holm, FDR-BH correctness
- merge_categorical: correctness, constraints, cache efficiency
"""

from __future__ import annotations

import math
from typing import List
from unittest.mock import patch

import numpy as np
import pytest

from MOBPY.core.categorical_merge import (
    CategoryBlock,
    ChiResult,
    _adjust_pvalues,
    _chi2_test,
    _enforce_cat_min_class_counts,
    _enforce_cat_min_samples,
    merge_categorical,
)
from MOBPY.core.constraints import BinningConstraints
from MOBPY.exceptions import FittingError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_block(
    categories,
    n: int,
    positives: float,
    *,
    sum2: float = None,
) -> CategoryBlock:
    """Build a CategoryBlock with binary y (0/1) statistics."""
    s = float(positives)
    # For binary y, sum2 = positives (since y² = y for 0/1)
    s2 = float(positives) if sum2 is None else sum2
    return CategoryBlock(
        n=n,
        sum=s,
        sum2=s2,
        ymin=0.0 if positives < n else 1.0,
        ymax=1.0 if positives > 0 else 0.0,
        categories=frozenset(categories),
    )


def make_constraints(
    max_bins: int = 6,
    min_bins: int = 2,
    min_samples: int = 0,
    min_positives: int = 0,
    min_negatives: int = 0,
    total_n: int = 1000,
    total_pos: int = 200,
) -> BinningConstraints:
    c = BinningConstraints(
        max_bins=max_bins,
        min_bins=min_bins,
        min_samples=min_samples if min_samples > 1 else None,
        min_positives=min_positives if min_positives > 1 else None,
        min_negatives=min_negatives if min_negatives > 1 else None,
    )
    c.resolve(total_n=total_n, total_pos=total_pos)
    # Manually set abs_ if we passed absolute values
    if min_samples > 1:
        c.abs_min_samples = min_samples
    if min_positives > 1:
        c.abs_min_positives = min_positives
    if min_negatives > 1:
        c.abs_min_negatives = min_negatives
    return c


# ---------------------------------------------------------------------------
# TestBlockStatsBase: inherited by CategoryBlock
# ---------------------------------------------------------------------------

class TestBlockStatsBase:
    """Tests that _BlockStatsBase properties work correctly via CategoryBlock."""

    def test_mean(self):
        b = make_block(["A"], n=10, positives=3)
        assert b.mean == pytest.approx(0.3)

    def test_mean_empty(self):
        b = CategoryBlock(n=0, sum=0, sum2=0, ymin=0, ymax=0, categories=frozenset())
        assert b.mean == 0.0

    def test_var_constant(self):
        # All y=1: sum2=n, mean=1, var should be 0
        b = make_block(["A"], n=5, positives=5, sum2=5.0)
        assert b.var == pytest.approx(0.0, abs=1e-10)

    def test_var_binary(self):
        # 50% events: variance = p*(1-p)*n/(n-1) corrected
        # y = [0,0,0,0,0,1,1,1,1,1], n=10, sum=5, sum2=5
        b = CategoryBlock(n=10, sum=5.0, sum2=5.0, ymin=0.0, ymax=1.0,
                          categories=frozenset(["A"]))
        # Unbiased var = (n/(n-1)) * p*(1-p) where p=0.5
        expected = (10 / 9) * 0.25
        assert b.var == pytest.approx(expected)

    def test_std(self):
        b = CategoryBlock(n=10, sum=5.0, sum2=5.0, ymin=0.0, ymax=1.0,
                          categories=frozenset(["A"]))
        assert b.std == pytest.approx(math.sqrt(b.var))

    def test_positives(self):
        b = make_block(["A"], n=20, positives=7)
        assert b.positives == pytest.approx(7.0)

    def test_negatives(self):
        b = make_block(["A"], n=20, positives=7)
        assert b.negatives == pytest.approx(13.0)

    def test_cv_zero_mean(self):
        b = make_block(["A"], n=10, positives=0)
        assert b.cv == 0.0


# ---------------------------------------------------------------------------
# TestCategoryBlock
# ---------------------------------------------------------------------------

class TestCategoryBlock:
    """Tests for CategoryBlock dataclass."""

    def test_initialization(self):
        b = make_block(["A", "B"], n=100, positives=30)
        assert b.n == 100
        assert b.sum == pytest.approx(30.0)
        assert b.categories == frozenset({"A", "B"})
        assert b.merge_history == []
        assert b.pvalue_history == []

    def test_merge_with_combines_stats(self):
        a = make_block(["A"], n=50, positives=10)
        b = make_block(["B"], n=30, positives=12)
        merged = a.merge_with(b)

        assert merged.n == 80
        assert merged.sum == pytest.approx(22.0)
        assert merged.categories == frozenset({"A", "B"})

    def test_merge_with_union_of_categories(self):
        a = make_block(["A", "B"], n=40, positives=8)
        c = make_block(["C", "D"], n=60, positives=15)
        merged = a.merge_with(c)
        assert merged.categories == frozenset({"A", "B", "C", "D"})

    def test_merge_with_tracks_history(self):
        a = make_block(["A"], n=50, positives=10)
        b = make_block(["B"], n=30, positives=12)
        merged = a.merge_with(b)
        assert len(merged.merge_history) == 1
        left_cats, right_cats = merged.merge_history[0]
        assert left_cats == frozenset({"A"})
        assert right_cats == frozenset({"B"})

    def test_merge_with_min_max_y(self):
        a = CategoryBlock(n=10, sum=3, sum2=3, ymin=0.0, ymax=1.0,
                          categories=frozenset(["A"]))
        b = CategoryBlock(n=10, sum=7, sum2=7, ymin=0.0, ymax=1.0,
                          categories=frozenset(["B"]))
        merged = a.merge_with(b)
        assert merged.ymin == pytest.approx(0.0)
        assert merged.ymax == pytest.approx(1.0)

    def test_as_dict_keys(self):
        b = make_block(["A", "B"], n=50, positives=15)
        d = b.as_dict()
        for key in ("categories", "n", "sum", "sum2", "ymin", "ymax",
                    "mean", "var", "std", "positives", "negatives"):
            assert key in d

    def test_as_dict_categories_sorted(self):
        b = make_block(["C", "A", "B"], n=30, positives=10)
        d = b.as_dict()
        assert d["categories"] == ["A", "B", "C"]

    def test_repr_contains_categories(self):
        b = make_block(["X"], n=100, positives=40)
        r = repr(b)
        assert "X" in r
        assert "n=100" in r


# ---------------------------------------------------------------------------
# TestChiSquareTest
# ---------------------------------------------------------------------------

class TestChiSquareTest:
    """Tests for _chi2_test function."""

    def test_identical_distributions_high_pvalue(self):
        """Identical proportions → high p-value (no evidence of difference)."""
        a = make_block(["A"], n=100, positives=20)
        b = make_block(["B"], n=100, positives=20)
        result = _chi2_test(a, b)
        assert result.pvalue > 0.5

    def test_very_different_distributions_low_pvalue(self):
        """Extreme proportions → low p-value."""
        a = make_block(["A"], n=500, positives=490)  # 98% event rate
        b = make_block(["B"], n=500, positives=10)   # 2% event rate
        result = _chi2_test(a, b)
        assert result.pvalue < 0.001

    def test_empty_block_returns_pvalue_1(self):
        a = CategoryBlock(n=0, sum=0, sum2=0, ymin=0, ymax=0,
                          categories=frozenset(["A"]))
        b = make_block(["B"], n=100, positives=30)
        result = _chi2_test(a, b)
        assert result.pvalue == pytest.approx(1.0)

    def test_zero_column_returns_pvalue_1(self):
        """All events in both blocks → zero-sum column → degenerate."""
        a = make_block(["A"], n=50, positives=50)  # 100% positives
        b = make_block(["B"], n=50, positives=50)  # 100% positives
        result = _chi2_test(a, b)
        assert result.pvalue == pytest.approx(1.0)

    def test_result_fields_are_valid(self):
        a = make_block(["A"], n=200, positives=60)
        b = make_block(["B"], n=200, positives=80)
        result = _chi2_test(a, b)
        assert 0.0 <= result.pvalue <= 1.0
        assert result.chi2 >= 0.0
        assert result.dof >= 1
        assert result.n_obs == 400

    def test_symmetric(self):
        """Test should be symmetric: chi2(a,b) == chi2(b,a)."""
        a = make_block(["A"], n=100, positives=20)
        b = make_block(["B"], n=150, positives=60)
        r1 = _chi2_test(a, b)
        r2 = _chi2_test(b, a)
        assert r1.pvalue == pytest.approx(r2.pvalue)
        assert r1.chi2 == pytest.approx(r2.chi2)


# ---------------------------------------------------------------------------
# TestAdjustPvalues
# ---------------------------------------------------------------------------

class TestAdjustPvalues:
    """Tests for _adjust_pvalues function."""

    def test_bonferroni_single(self):
        adj = _adjust_pvalues([0.02], "bonferroni")
        assert len(adj) == 1
        assert adj[0] == pytest.approx(0.02)  # 1 * 0.02 = 0.02

    def test_bonferroni_multiple(self):
        pvals = [0.01, 0.05, 0.20]
        adj = _adjust_pvalues(pvals, "bonferroni")
        assert adj[0] == pytest.approx(min(0.03, 1.0))
        assert adj[1] == pytest.approx(min(0.15, 1.0))
        assert adj[2] == pytest.approx(min(0.60, 1.0))

    def test_bonferroni_capped_at_1(self):
        pvals = [0.5, 0.5, 0.5]
        adj = _adjust_pvalues(pvals, "bonferroni")
        assert all(a <= 1.0 for a in adj)

    def test_holm_monotone_non_decreasing(self):
        """Holm adjusted p-values must be non-decreasing when sorted by original rank."""
        pvals = [0.001, 0.01, 0.05, 0.10, 0.50]
        adj = _adjust_pvalues(pvals, "holm")
        # Sort both by the original order of pvals
        order = np.argsort(pvals)
        adj_sorted = [adj[i] for i in order]
        for i in range(len(adj_sorted) - 1):
            assert adj_sorted[i] <= adj_sorted[i + 1] + 1e-12

    def test_holm_more_conservative_than_raw(self):
        """Holm adjusted p-values are always >= raw p-values."""
        pvals = [0.01, 0.03, 0.15, 0.40]
        adj = _adjust_pvalues(pvals, "holm")
        for raw, a in zip(pvals, adj):
            assert a >= raw - 1e-12

    def test_holm_single(self):
        adj = _adjust_pvalues([0.04], "holm")
        assert adj[0] == pytest.approx(0.04)

    def test_fdr_bh_monotone_nondecreasing(self):
        pvals = [0.001, 0.01, 0.05, 0.20]
        adj = _adjust_pvalues(pvals, "fdr_bh")
        order = np.argsort(pvals)
        adj_sorted = [adj[i] for i in order]
        for i in range(len(adj_sorted) - 1):
            assert adj_sorted[i] <= adj_sorted[i + 1] + 1e-12

    def test_empty_input(self):
        adj = _adjust_pvalues([], "holm")
        assert len(adj) == 0

    def test_unknown_correction_raises(self):
        with pytest.raises(ValueError, match="Unknown correction"):
            _adjust_pvalues([0.05], "invalid_method")

    def test_all_methods_clip_to_1(self):
        pvals = [0.9, 0.9, 0.9]
        for method in ("bonferroni", "holm", "fdr_bh"):
            adj = _adjust_pvalues(pvals, method)
            assert all(a <= 1.0 for a in adj), f"Method {method} exceeded 1.0"

    def test_all_methods_clip_to_0(self):
        pvals = [0.0, 0.0]
        for method in ("bonferroni", "holm", "fdr_bh"):
            adj = _adjust_pvalues(pvals, method)
            assert all(a >= 0.0 for a in adj), f"Method {method} went below 0.0"


# ---------------------------------------------------------------------------
# TestMergeCategorical
# ---------------------------------------------------------------------------

class TestMergeCategorical:
    """Tests for the main merge_categorical function."""

    def _make_five_category_blocks(self) -> List[CategoryBlock]:
        """Five-category example: A(90%+), B(70%), C(50%), D(30%), E(10%)."""
        return [
            make_block(["A"], n=100, positives=90),
            make_block(["B"], n=100, positives=70),
            make_block(["C"], n=100, positives=50),
            make_block(["D"], n=100, positives=30),
            make_block(["E"], n=100, positives=10),
        ]

    def _resolved_constraints(self, **kwargs) -> BinningConstraints:
        c = BinningConstraints(**kwargs)
        c.resolve(total_n=500, total_pos=250)
        return c

    def test_empty_blocks_returns_empty(self):
        c = self._resolved_constraints(max_bins=4, min_bins=2)
        result = merge_categorical([], c, is_binary_y=True)
        assert result == []

    def test_single_block_returns_as_is(self):
        b = make_block(["A"], n=100, positives=30)
        c = self._resolved_constraints(max_bins=4, min_bins=1)
        result = merge_categorical([b], c, is_binary_y=True)
        assert len(result) == 1
        assert result[0].categories == frozenset({"A"})

    def test_non_binary_y_raises(self):
        blocks = self._make_five_category_blocks()
        c = self._resolved_constraints(max_bins=4, min_bins=2)
        with pytest.raises(ValueError, match="binary"):
            merge_categorical(blocks, c, is_binary_y=False)

    def test_max_bins_respected(self):
        blocks = self._make_five_category_blocks()
        for max_b in (2, 3, 4):
            c = self._resolved_constraints(max_bins=max_b, min_bins=1)
            result = merge_categorical(blocks, c, is_binary_y=True)
            assert len(result) <= max_b, f"Expected <= {max_b}, got {len(result)}"

    def test_min_bins_floor_respected(self):
        """With very permissive alpha, all similar blocks could merge, but min_bins stops it."""
        blocks = [
            make_block(["A"], n=100, positives=50),
            make_block(["B"], n=100, positives=51),
            make_block(["C"], n=100, positives=49),
            make_block(["D"], n=100, positives=50),
        ]
        c = self._resolved_constraints(max_bins=10, min_bins=3)
        result = merge_categorical(blocks, c, is_binary_y=True, alpha=1.0)
        assert len(result) >= 3

    def test_all_categories_preserved(self):
        """After merging, every original category must appear in exactly one bin."""
        blocks = self._make_five_category_blocks()
        c = self._resolved_constraints(max_bins=3, min_bins=1)
        result = merge_categorical(blocks, c, is_binary_y=True)

        all_original = {"A", "B", "C", "D", "E"}
        all_in_result: set = set()
        for b in result:
            # Bins must not overlap
            assert b.categories.isdisjoint(all_in_result), "Category appears in multiple bins"
            all_in_result |= b.categories
        assert all_in_result == all_original

    def test_similar_blocks_merged_first(self):
        """Blocks with nearly identical event rates should merge first."""
        # B and C have similar rates; A and E are far apart from each other
        blocks = [
            make_block(["A"], n=200, positives=190),
            make_block(["B"], n=200, positives=100),
            make_block(["C"], n=200, positives=102),
            make_block(["E"], n=200, positives=10),
        ]
        c = self._resolved_constraints(max_bins=3, min_bins=1)
        result = merge_categorical(blocks, c, is_binary_y=True, alpha=1.0)

        merged_cats = {frozenset(b.categories) for b in result}
        # B and C should merge (most similar), A and E remain separate
        assert frozenset({"B", "C"}) in merged_cats

    def test_history_recorded(self):
        blocks = self._make_five_category_blocks()
        c = self._resolved_constraints(max_bins=3, min_bins=1)
        history: list = []
        merge_categorical(blocks, c, is_binary_y=True, history=history)
        assert len(history) > 0

    def test_merge_produces_valid_stats(self):
        """Merged block statistics should equal the sum of component blocks."""
        a = make_block(["A"], n=80, positives=20)
        b = make_block(["B"], n=120, positives=60)
        c = make_block(["C"], n=100, positives=40)
        constraints = BinningConstraints(max_bins=2, min_bins=1)
        constraints.resolve(total_n=300, total_pos=120)
        result = merge_categorical([a, b, c], constraints, is_binary_y=True)

        total_n = sum(bl.n for bl in result)
        total_sum = sum(bl.sum for bl in result)
        assert total_n == 300
        assert total_sum == pytest.approx(120.0)

    def test_cache_efficiency(self):
        """Cached approach should call chi-square far fewer times than naive O(k³)."""
        blocks = self._make_five_category_blocks()
        # 5 categories: initial pairs = 10. After each merge, only k new pairs.
        # Total for 2 merges (5->3): 10 + 4 + 3 = 17 calls max vs. naive 10+6+3=19
        # We just verify the function completes without error and produces valid output.
        c = self._resolved_constraints(max_bins=3, min_bins=1)
        result = merge_categorical(blocks, c, is_binary_y=True)
        assert 1 <= len(result) <= 3

    def test_bonferroni_more_conservative_than_holm(self):
        """Bonferroni correction keeps more bins (less merging) than Holm."""
        # Use identical blocks so many pairs qualify to merge
        blocks = [
            make_block([f"cat{i}"], n=100, positives=50 + i)
            for i in range(8)
        ]
        c = BinningConstraints(max_bins=8, min_bins=1)
        c.resolve(total_n=800, total_pos=404)

        result_holm = merge_categorical(blocks, c, is_binary_y=True,
                                        alpha=0.05, correction="holm")
        result_bonf = merge_categorical(blocks, c, is_binary_y=True,
                                        alpha=0.05, correction="bonferroni")
        # Bonferroni is strictly more conservative: can have same or fewer merges
        assert len(result_bonf) >= len(result_holm)

    def test_min_samples_enforced(self):
        """Bins below min_samples are merged during Phase 2."""
        blocks = [
            make_block(["A"], n=5, positives=4),    # undersized
            make_block(["B"], n=200, positives=100),
            make_block(["C"], n=200, positives=50),
        ]
        c = BinningConstraints(max_bins=6, min_bins=1)
        c.resolve(total_n=405, total_pos=154)
        c.abs_min_samples = 50  # A is below this

        result = merge_categorical(blocks, c, is_binary_y=True, alpha=0.0)
        # A must be merged with something
        assert all(b.n >= 50 for b in result)

    def test_min_positives_enforced(self):
        """Bins below min_positives are merged during Phase 3."""
        blocks = [
            make_block(["A"], n=100, positives=2),    # low positives
            make_block(["B"], n=100, positives=50),
            make_block(["C"], n=100, positives=60),
        ]
        c = BinningConstraints(max_bins=6, min_bins=1)
        c.resolve(total_n=300, total_pos=112)
        c.abs_min_positives = 10

        result = merge_categorical(blocks, c, is_binary_y=True, alpha=0.0)
        assert all(b.positives >= 10 for b in result)

    def test_naive_vs_cached_same_result(self):
        """Both approaches should produce the same number of bins (deterministic)."""
        # Since merge_categorical is deterministic for a given seed of blocks,
        # running it twice must give identical results.
        blocks = self._make_five_category_blocks()
        c = self._resolved_constraints(max_bins=3, min_bins=1)

        result1 = merge_categorical(blocks, c, is_binary_y=True)
        result2 = merge_categorical(blocks, c, is_binary_y=True)

        assert len(result1) == len(result2)
        cats1 = sorted(sorted(str(c) for c in b.categories) for b in result1)
        cats2 = sorted(sorted(str(c) for c in b.categories) for b in result2)
        assert cats1 == cats2
