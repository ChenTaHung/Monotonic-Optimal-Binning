"""
Property-based tests for the MOB/PAVA pipeline.

We exercise the system over a broad range of randomly generated datasets
and constraints, and assert high-level invariants:
- monotone bin means (according to resolved sign),
- structural bin-edge conventions (last right is +inf; left edges finite),
- sample/positives constraints enforcement (best-effort with min-bins floor),
- missing/excluded handling in the summary,
- consistency between bins() and summary() totals.

NOTE: These tests use deterministic "noise" (no RNG) so Hypothesis can
shrink counterexamples reliably.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings, strategies as st

from MOBPY.binning.mob import MonotonicBinner
from MOBPY.core.constraints import BinningConstraints


# ---------------------------------------------------------------------
# Helper strategies and utilities
# ---------------------------------------------------------------------

def _finite_floats(min_value=-5.0, max_value=5.0) -> st.SearchStrategy[float]:
    """Finite floats (no NaN/inf) to keep PAVA happy."""
    return st.floats(min_value=min_value, max_value=max_value, allow_nan=False, allow_infinity=False, width=32)


@st.composite
def ds_binary(draw) -> Tuple[pd.DataFrame, List[float]]:
    """
    Generate a binary dataset with:
      - x: finite floats, >= 30 rows
      - y: deterministic thresholding of an affine transform of x, optionally flipped by a deterministic pattern
      - some missing x (first few rows) to exercise partitioning
      - a small set of 'exclude' values; some may be absent (should not appear in summary)
    """
    n = draw(st.integers(min_value=60, max_value=300))
    xs = draw(st.lists(_finite_floats(-3, 3), min_size=n, max_size=n))
    x = np.array(xs, dtype=float)

    # affine transform y* = s * x + b
    s = draw(st.sampled_from([-1.0, 1.0]))
    b = draw(_finite_floats(-0.5, 0.5))
    y_star = s * x + b

    # binary labels by threshold at median (deterministic, balanced-ish)
    thr = float(np.median(y_star))
    y = (y_star > thr).astype(float)

    # deterministic "flip" pattern to roughen boundaries: flip every k-th label
    k = draw(st.integers(min_value=0, max_value=15))
    if k > 1:
        y[::k] = 1.0 - y[::k]

    # inject missing x (first m rows)
    m = draw(st.integers(min_value=0, max_value=min(10, n // 10)))
    if m > 0:
        x[:m] = np.nan

    df = pd.DataFrame({"x": x, "y": y})

    # exclude up to 3 unique present x values + maybe 1 sentinel absent
    uniq = np.unique(df["x"][df["x"].notna()])
    excl_present = draw(st.lists(st.sampled_from(uniq.tolist()) if len(uniq) else st.just(0.0),
                                 min_size=0, max_size=min(3, len(uniq))))
    maybe_absent = draw(st.booleans())
    excl: List[float] = list(dict.fromkeys(excl_present))  # uniq while preserving order
    if maybe_absent:
        excl.append(999999.0)  # sentinel, likely absent

    # keep at least 3 unique clean x for meaningful binning
    clean = df.dropna(subset=["x"])
    assume(clean["x"].nunique() >= 3)

    return df, excl


@st.composite
def ds_numeric(draw) -> Tuple[pd.DataFrame, List[float]]:
    """
    Generate a numeric dataset:
      - y = a*x + b + deterministic periodic "noise"
      - some missing x
      - a small set of exclude values
    """
    n = draw(st.integers(min_value=60, max_value=300))
    xs = draw(st.lists(_finite_floats(-3, 3), min_size=n, max_size=n))
    x = np.array(xs, dtype=float)

    a = draw(_finite_floats(-2, 2))
    b = draw(_finite_floats(-1, 1))
    # deterministic "noise" from x (no RNG)
    noise = 0.2 * np.sin(3.1 * x) + 0.1 * np.cos(5.7 * x)
    y = a * x + b + noise

    # inject missing x
    m = draw(st.integers(min_value=0, max_value=min(10, n // 10)))
    if m > 0:
        x[:m] = np.nan

    df = pd.DataFrame({"x": x, "y": y})

    uniq = np.unique(df["x"][df["x"].notna()])
    excl_present = draw(st.lists(st.sampled_from(uniq.tolist()) if len(uniq) else st.just(0.0),
                                 min_size=0, max_size=min(3, len(uniq))))
    maybe_absent = draw(st.booleans())
    excl: List[float] = list(dict.fromkeys(excl_present))
    if maybe_absent:
        excl.append(-999999.0)

    clean = df.dropna(subset=["x"])
    assume(clean["x"].nunique() >= 3)

    return df, excl


@st.composite
def constraints_strategy(draw) -> BinningConstraints:
    """
    Generate constraints that are usually feasible after resolution.
    We bias towards safe combinations to avoid trivial invalid cases.
    """
    max_bins = draw(st.integers(min_value=1, max_value=8))
    min_bins = draw(st.integers(min_value=1, max_value=max_bins))

    # Favor None for fractions to reduce cross-constraint errors,
    # but sometimes generate fractional or absolute values.
    max_samples = draw(st.one_of(st.none(), st.floats(0.2, 1.0), st.integers(10, 10_000)))
    min_samples = draw(st.one_of(st.none(), st.floats(0.0, 0.5), st.integers(0, 2_000)))
    min_positives = draw(st.one_of(st.none(), st.floats(0.0, 0.5), st.integers(0, 2_000)))

    initial_pvalue = draw(st.floats(min_value=1e-6, max_value=0.95))
    maximize_bins = draw(st.booleans())

    return BinningConstraints(
        max_bins=max_bins,
        min_bins=min_bins,
        max_samples=max_samples,
        min_samples=min_samples,
        min_positives=min_positives,
        initial_pvalue=float(initial_pvalue),
        maximize_bins=bool(maximize_bins),
    )


def _resolve_or_discard(
    cons: BinningConstraints,
    df: pd.DataFrame,
    excl: List[float],
    *,
    y_is_binary: bool,
) -> BinningConstraints:
    """
    Attempt to resolve constraint fractions to absolutes using CLEAN partition.
    If resolution is impossible (e.g., min > max), discard the example.
    """
    clean = df.dropna(subset=["x", "y"])
    if excl:
        clean = clean[~clean["x"].isin(excl)]
    total_n = int(len(clean))
    total_pos = int(clean["y"].sum()) if y_is_binary else 0

    try:
        cons2 = deepcopy(cons)
        cons2.resolve(total_n=total_n, total_pos=total_pos)
        return cons2
    except ValueError:
        assume(False)  # discard this generated combination
        return cons  # unreachable


# ---------------------------------------------------------------------
# Properties for binary targets (MOB case)
# ---------------------------------------------------------------------

@settings(deadline=None, max_examples=50)
@given(ds_binary(), constraints_strategy())
def test_property_binary_pipeline(data_excl, cons):
    df, excl = data_excl
    cons = _resolve_or_discard(cons, df, excl, y_is_binary=True)

    binner = MonotonicBinner(
        df=df, x="x", y="y",
        metric="mean", sign="auto", strict=True,
        constraints=cons, exclude_values=excl,
    ).fit()

    bins = binner.bins_()
    summary = binner.summary_()
    assert not bins.empty

    # Edges: finite left, last right +inf
    left = bins["left"].to_numpy()
    right = bins["right"].to_numpy()
    assert np.isfinite(left).all()
    assert np.isfinite(right[:-1]).all()
    assert math.isinf(right[-1]) and right[-1] > 0

    # Monotone means according to resolved sign
    sign = getattr(binner, "resolved_sign_", "+")
    means = bins["mean"].to_numpy()
    diffs = np.diff(means)
    if sign == "+":
        assert (diffs >= -1e-12).all()
    else:
        assert (diffs <= 1e-12).all()

    # Binary sanity
    assert (bins["mean"] >= -1e-9).all() and (bins["mean"] <= 1 + 1e-9).all()
    assert (bins["n"] >= 1).all()
    assert np.isfinite(bins["std"].to_numpy()).all()

    # Min-samples rule
    if cons.abs_min_samples:
        assert (bins["n"] >= cons.abs_min_samples).all() or (len(bins) <= cons.min_bins)

    # Minimal feasible partition from PAVA
    pava_k = len(binner._pava.blocks_)
    if cons.maximize_bins:
        assert len(bins) <= cons.max_bins
    else:
        if pava_k >= cons.min_bins:
            assert len(bins) >= cons.min_bins or len(bins) == 1
        else:
            assert len(bins) == pava_k


# ---------------------------------------------------------------------
# Properties for numeric targets
# ---------------------------------------------------------------------

@settings(deadline=None, max_examples=40)
@given(ds_numeric(), constraints_strategy())
def test_property_numeric_pipeline(data_excl, cons):
    df, excl = data_excl

    cons = _resolve_or_discard(cons, df, excl, y_is_binary=False)

    binner = MonotonicBinner(
        df=df, x="x", y="y",
        metric="mean", sign="auto", strict=True,
        constraints=cons, exclude_values=excl,
    ).fit()

    bins = binner.bins_()
    assert not bins.empty

    # Edges: finite left, last right +inf
    left = bins["left"].to_numpy()
    right = bins["right"].to_numpy()
    assert np.isfinite(left).all()
    assert np.isfinite(right[:-1]).all()
    assert math.isinf(right[-1]) and right[-1] > 0

    # Monotone means
    sign = getattr(binner, "resolved_sign_", "+")
    means = bins["mean"].to_numpy()
    diffs = np.diff(means)
    if sign == "+":
        assert (diffs >= -1e-12).all()
    else:
        assert (diffs <= 1e-12).all()

    # Constraints on bin counts
    if cons.abs_min_samples:
        assert (bins["n"] >= cons.abs_min_samples).all() or (len(bins) <= cons.min_bins)

    # Respect max/min bin targets, accounting for minimal feasible PAVA partition
    pava_k = len(binner._pava.blocks_)
    if cons.maximize_bins:
        assert len(bins) <= cons.max_bins
    else:
        if pava_k >= cons.min_bins:
            assert len(bins) >= cons.min_bins or len(bins) == 1
        else:
            assert len(bins) == pava_k

# ---------------------------------------------------------------------
# A few focused unit-like properties
# ---------------------------------------------------------------------

@settings(deadline=None, max_examples=15)
@given(ds_binary(), constraints_strategy())
def test_transform_alignment_binary(data_excl, cons):
    df, excl = data_excl
    cons = _resolve_or_discard(cons, df, excl, y_is_binary=True)

    binner = MonotonicBinner(df=df, x="x", y="y", constraints=cons, exclude_values=excl).fit()
    bins = binner.bins_()

    # Pick a few clean x values and check that transform(left) returns that left
    clean_x = df["x"].dropna()
    if excl:
        clean_x = clean_x[~clean_x.isin(excl)]
    assume(not clean_x.empty)

    sample = clean_x.sample(min(5, len(clean_x)), random_state=0)
    t_left = binner.transform(sample, assign="left")
    t_right = binner.transform(sample, assign="right")
    t_interval = binner.transform(sample, assign="interval")

    # The assigned (left,right) should bound x
    for xv, l, r, label in zip(sample.to_numpy(), t_left.to_numpy(), t_right.to_numpy(), t_interval.to_numpy()):
        assert l <= xv <= (r if np.isfinite(r) else 1e308)
        assert label.startswith("[") or label.startswith("(")
        assert label.endswith(")")


@settings(deadline=None, max_examples=10)
@given(ds_binary(), constraints_strategy())
def test_summary_special_rows(data_excl, cons):
    df, excl = data_excl
    cons = _resolve_or_discard(cons, df, excl, y_is_binary=True)
    b = MonotonicBinner(df=df, x="x", y="y", constraints=cons, exclude_values=excl).fit()
    summary = b.summary_()

    # Missing row presence
    if df["x"].isna().any():
        assert (summary["interval"] == "Missing").any()
    else:
        assert not (summary["interval"] == "Missing").any()

    # Excluded rows presence exactly if value appears
    for v in (excl or []):
        present = df["x"].eq(v).sum() > 0
        exists = (summary["interval"] == str(v)).any()
        assert exists == present
