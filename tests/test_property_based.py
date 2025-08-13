# tests/test_property_based.py
from __future__ import annotations

import math
import warnings
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from MOBPY.binning.mob import MonotonicBinner
from MOBPY.core.constraints import BinningConstraints


# -- keep test output clean if numpy changes internal warnings in corrcoef --
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning,
)


# --------------------------- Hypothesis strategies --------------------------- #

@st.composite
def constraints_strategy(draw):
    """Generate a variety of realistic constraint configurations."""
    maximize_bins = draw(st.booleans())

    max_bins = draw(st.integers(min_value=2, max_value=6))
    min_bins = draw(st.integers(min_value=1, max_value=max_bins))

    # Fractions or None for sample constraints
    max_samples = draw(st.one_of(st.none(), st.floats(min_value=0.05, max_value=0.8)))
    min_samples = draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=0.4)))

    # For binary targets; numeric tests will set this to None effectively
    min_positives = draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=0.5)))

    initial_pvalue = draw(st.floats(min_value=1e-6, max_value=0.8))

    return BinningConstraints(
        max_bins=max_bins,
        min_bins=min_bins,
        max_samples=max_samples,
        min_samples=min_samples,
        min_positives=min_positives,
        initial_pvalue=initial_pvalue,
        maximize_bins=maximize_bins,
    )


@st.composite
def ds_binary(draw) -> Tuple[pd.DataFrame, List[float]]:
    """Random binary dataset with clustered x (1..6 unique values)."""
    n = draw(st.integers(min_value=60, max_value=300))
    k = draw(st.integers(min_value=1, max_value=6))

    # Assign each row to a cluster label in [0, k-1]
    labels = draw(st.lists(st.integers(min_value=0, max_value=k - 1), min_size=n, max_size=n))
    x = np.array(labels, dtype=float)

    # Per-cluster Bernoulli probabilities
    ps = [draw(st.floats(min_value=0.02, max_value=0.98)) for _ in range(k)]
    # Deterministic U(0,1) per item to avoid extra RNG
    uniforms = draw(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=n, max_size=n))
    y = np.array([1.0 if u < ps[int(lbl)] else 0.0 for lbl, u in zip(labels, uniforms)], dtype=float)

    df = pd.DataFrame({"x": x, "y": y})

    # Optional exclude list: subset of present x values (0-2 values)
    uniq = sorted(set(x.tolist()))
    excl = draw(st.lists(st.sampled_from(uniq), unique=True, min_size=0, max_size=min(2, len(uniq))))
    return df, excl


@st.composite
def ds_numeric(draw) -> Tuple[pd.DataFrame, List[float]]:
    """Random numeric dataset with clustered x (1..6 unique values)."""
    n = draw(st.integers(min_value=60, max_value=400))
    k = draw(st.integers(min_value=1, max_value=6))
    labels = draw(st.lists(st.integers(min_value=0, max_value=k - 1), min_size=n, max_size=n))
    x = np.array(labels, dtype=float)

    # Per-cluster mean levels (may be non-monotone)
    mus = [draw(st.floats(min_value=-3.0, max_value=3.0)) for _ in range(k)]
    # Per-row noise in [-0.5, 0.5]
    eps = draw(st.lists(st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False), min_size=n, max_size=n))
    y = np.array([mus[int(lbl)] + e for lbl, e in zip(labels, eps)], dtype=float)

    df = pd.DataFrame({"x": x, "y": y})

    uniq = sorted(set(x.tolist()))
    excl = draw(st.lists(st.sampled_from(uniq), unique=True, min_size=0, max_size=min(2, len(uniq))))
    return df, excl


# ------------------------------- helper funcs ------------------------------- #

def _resolve_or_discard(cons: BinningConstraints, df: pd.DataFrame, excl: List[float], *, y_is_binary: bool) -> BinningConstraints:
    """Resolve constraints on the clean subset; discard (assume False) if impossible."""
    clean = df.dropna(subset=["x", "y"])
    if excl:
        clean = clean[~clean["x"].isin(excl)]
    assume(not clean.empty)

    total_n = int(clean.shape[0])
    total_pos = int(clean["y"].sum()) if y_is_binary else 0
    try:
        cons.resolve(total_n=total_n, total_pos=total_pos)
    except ValueError:
        # e.g., min_samples > max_samples → discard this draw
        assume(False)
    return cons


def _assert_monotone_means(bins: pd.DataFrame, sign: str):
    means = bins["mean"].to_numpy()
    diffs = np.diff(means)
    if sign == "+":
        assert (diffs >= -1e-12).all()
    else:
        assert (diffs <= 1e-12).all()


def _assert_bin_edges(df_bins: pd.DataFrame):
    left = df_bins["left"].to_numpy()
    right = df_bins["right"].to_numpy()
    # first left is -inf, last right is +inf
    assert math.isinf(left[0]) and left[0] < 0
    assert np.isfinite(left[1:]).all()

    assert np.isfinite(right[:-1]).all()
    assert math.isinf(right[-1]) and right[-1] > 0


# ---------------------------------- tests ----------------------------------- #

@settings(deadline=None, max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(ds_binary(), constraints_strategy())
def test_property_binary_pipeline(data_excl, cons):
    df, excl = data_excl

    # Resolve constraints early to avoid invalid combinations during fit
    cons = _resolve_or_discard(cons, df, excl, y_is_binary=True)

    binner = MonotonicBinner(
        df=df, x="x", y="y",
        metric="mean", sign="auto", strict=True,
        constraints=cons, exclude_values=excl,
    ).fit()

    bins = binner.bins_()
    assert not bins.empty

    # Structural edges
    _assert_bin_edges(bins)

    # monotone means according to resolved sign
    sign = getattr(binner, "resolved_sign_", "+")
    _assert_monotone_means(bins, sign=sign)

    # means in [0,1] for binary; std is finite; counts positive
    assert (bins["mean"] >= -1e-9).all() and (bins["mean"] <= 1 + 1e-9).all()
    assert (bins["n"] >= 1).all()
    assert np.isfinite(bins["std"].to_numpy()).all()

    # If abs_min_samples was requested, either all bins meet it OR we could not
    # further merge because we reached (or went below) min_bins.
    if cons.abs_min_samples:
        assert (bins["n"] >= cons.abs_min_samples).all() or (len(bins) <= cons.min_bins)

    # Compare achievable bins to PAVA output; post-merge cannot create bins
    pava_blocks = getattr(binner, "_pava").blocks_
    n0 = len(pava_blocks)

    if cons.maximize_bins:
        assert len(bins) <= cons.max_bins
    else:
        target = min(cons.min_bins, n0)
        assert len(bins) >= target or len(bins) == 1


@settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.too_slow])
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
    _assert_bin_edges(bins)

    # Monotone means
    sign = getattr(binner, "resolved_sign_", "+")
    _assert_monotone_means(bins, sign=sign)

    # Constraints on bin counts:
    if cons.abs_min_samples:
        assert (bins["n"] >= cons.abs_min_samples).all() or (len(bins) <= cons.min_bins)

    # Cannot exceed PAVA's piecewise-constant output in non-maximize mode
    pava_blocks = getattr(binner, "_pava").blocks_
    n0 = len(pava_blocks)

    if cons.maximize_bins:
        assert len(bins) <= cons.max_bins
    else:
        target = min(cons.min_bins, n0)
        assert len(bins) >= target or len(bins) == 1
