import math
import numpy as np
import pandas as pd

from MOBPY.binning.mob import MonotonicBinner
from MOBPY.core.constraints import BinningConstraints


def _assert_monotone_means(df_bins: pd.DataFrame, sign: str):
    means = df_bins["mean"].to_numpy(dtype=float)
    diffs = np.diff(means)
    if sign == "+":
        assert (diffs >= -1e-12).all()
    else:
        assert (diffs <= 1e-12).all()


def test_binner_binary_end_to_end():
    rng = np.random.default_rng(123)
    n = 400
    x = np.linspace(-2, 2, n) + rng.normal(0, 0.05, n)
    # probability increases with x (logistic)
    p = 1 / (1 + np.exp(-2 * x))
    y = rng.binomial(1, p, size=n)
    # add some missing/specials
    x[:5] = np.nan
    x[5:10] = 999.0

    df = pd.DataFrame({"x": x, "y": y})
    cons = BinningConstraints(max_bins=5, min_bins=2, min_samples=0.05, initial_pvalue=0.2)
    binner = MonotonicBinner(df=df, x="x", y="y", constraints=cons, exclude_values=[999.0]).fit()

    bins = binner.bins_()
    summary = binner.summary_()

    assert not bins.empty
    assert np.isfinite(bins["left"]).all()
    # last right must be +inf for clean transform behavior
    assert math.isinf(bins["right"].iloc[-1])

    # resolved sign monotonicity
    sign = binner.resolved_sign_
    _assert_monotone_means(bins, sign)

    # binary sanity
    assert (bins["mean"] >= -1e-9).all() and (bins["mean"] <= 1 + 1e-9).all()
    # summary contains special rows
    assert (summary["interval"] == "Missing").any()
    assert (summary["interval"] == "999.0").any()

    # transform sanity (interval strings and edges)
    s = pd.Series([-10, 0, 10, np.nan, 999.0])
    out_interval = binner.transform(s, assign="interval")
    out_left = binner.transform(s, assign="left")
    out_right = binner.transform(s, assign="right")
    assert out_interval.iloc[3] == "Missing"
    assert out_interval.iloc[4] == "999.0"
    assert np.isnan(out_left.iloc[4]) and np.isnan(out_right.iloc[4])
    # numeric assignments for normal values
    assert isinstance(out_left.iloc[0], float)
    assert isinstance(out_right.iloc[0], float)


def test_binner_numeric_end_to_end():
    rng = np.random.default_rng(321)
    n = 300
    x = np.linspace(0, 3, n) + rng.normal(0, 0.02, n)
    # numeric y decreasing with x
    y = 5 - 1.5 * x + rng.normal(0, 0.2, n)
    df = pd.DataFrame({"x": x, "y": y})

    cons = BinningConstraints(max_bins=4, min_bins=2, initial_pvalue=0.25, min_samples=0.05)
    b = MonotonicBinner(df=df, x="x", y="y", constraints=cons).fit()

    bins = b.bins_()
    assert not bins.empty
    assert np.isfinite(bins["left"]).all()
    assert math.isinf(bins["right"].iloc[-1])

    sign = b.resolved_sign_
    _assert_monotone_means(bins, sign)

    # summary numeric columns present for numeric y
    summary = b.summary_()
    for col in ["sum", "mean", "std", "min", "max"]:
        assert col in summary.columns
