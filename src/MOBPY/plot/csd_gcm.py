# src/MOBPY/plot/csd_gcm.py
"""
CSD/GCM visualization for a fitted PAVA instance.

This plot shows, side-by-side on one set of axes:

- CSD (Cumulative Summary Diagram) on the chosen metric over ordered `x`.
- GCM (Greatest Convex Minorant, i.e., the isotonic/PAVA fit) as a
  piecewise-constant red line sampled at bin right edges.

It works directly with a fitted `MOBPY.core.pava.PAVA` object.

Definitions used here
---------------------
CSD (this implementation):
    For each unique, sorted `x`, we compute cumulative versions of supported
    metrics up to and including that `x`:
      - cum_count  = cumulative sum of counts
      - cum_sum    = cumulative sum of y
      - cum_mean   = cum_sum / cum_count
      - cum_var    = unbiased sample variance over all points seen so far
      - cum_std    = sqrt(cum_var)
      - cum_min    = min so far
      - cum_max    = max so far
      - cum_ptp    = cum_max - cum_min

GCM:
    The PAVA output bins (left-closed, right-open). We take the *bin-level*
    value of the same metric and draw it as a piecewise-constant line,
    sampled at each bin's right edge.

Usage
-----
>>> from MOBPY.core.pava import PAVA
>>> from MOBPY.plot.csd_gcm import plot_csd_gcm
>>>
>>> p = PAVA(df, x="Durationinmonth", y="default", metric="mean").fit()
>>> plot_csd_gcm(p)  # renders CSD (blue) and GCM (red)

Notes
-----
- For metrics like "std"/"var"/"ptp", the cumulative version is well-defined
  (running std/var/min/max span) but is *not* a convex diagram in the strict
  theoretical sense. The plot still gives an intuitive comparison between
  the raw cumulative behavior and the fitted monotone (GCM) step function.
- For binary y and metric="mean", this corresponds to the classic isotonic
  regression visualization: CSD derived from (count, sum), GCM slopes equal
  to isotonic bin means.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MOBPY.core.pava import PAVA


def _cum_stats_from_groups(groups: pd.DataFrame) -> pd.DataFrame:
    """Build cumulative stats from PAVA.groups_.

    Args:
        groups: DataFrame with columns
            ['x_value','count','sum','sum2','min','max'] sorted by x_value.

    Returns:
        DataFrame with columns
            ['x_value','cum_count','cum_sum','cum_mean','cum_var','cum_std',
             'cum_min','cum_max','cum_ptp']
    """
    g = groups.sort_values("x_value", kind="mergesort").reset_index(drop=True).copy()

    # Cumulative count and sum are straightforward
    cum_count = g["count"].cumsum().to_numpy(dtype=float)
    cum_sum = g["sum"].cumsum().to_numpy(dtype=float)

    # Running mean
    with np.errstate(invalid="ignore", divide="ignore"):
        cum_mean = np.where(cum_count > 0, cum_sum / cum_count, np.nan)

    # Running sum of squares → running unbiased sample variance
    # sum2(cumulative) = sum of individual sum2 (additive)
    cum_sum2 = g["sum2"].cumsum().to_numpy(dtype=float)
    # sample variance: (Σy^2 - (Σy)^2 / n) / (n - 1)
    cum_var = np.zeros_like(cum_sum2, dtype=float)
    mask_var = cum_count > 1
    num = cum_sum2[mask_var] - (cum_sum[mask_var] * cum_sum[mask_var]) / cum_count[mask_var]
    cum_var[mask_var] = np.maximum(num / (cum_count[mask_var] - 1.0), 0.0)
    cum_std = np.sqrt(cum_var)

    # Running min/max/ptp
    cum_min = g["min"].cummin().to_numpy(dtype=float)
    cum_max = g["max"].cummax().to_numpy(dtype=float)
    cum_ptp = cum_max - cum_min

    out = pd.DataFrame(
        {
            "x_value": g["x_value"].to_numpy(dtype=float),
            "cum_count": cum_count,
            "cum_sum": cum_sum,
            "cum_mean": cum_mean,
            "cum_var": cum_var,
            "cum_std": cum_std,
            "cum_min": cum_min,
            "cum_max": cum_max,
            "cum_ptp": cum_ptp,
        }
    )
    return out


def _select_csd_series(cum_df: pd.DataFrame, metric: str) -> pd.Series:
    """Pick the appropriate cumulative series for the requested metric."""
    m = metric.lower()
    if m == "count":
        return cum_df["cum_count"]
    if m == "sum":
        return cum_df["cum_sum"]
    if m == "mean":
        return cum_df["cum_mean"]
    if m == "std":
        return cum_df["cum_std"]
    if m == "var":
        return cum_df["cum_var"]
    if m == "min":
        return cum_df["cum_min"]
    if m == "max":
        return cum_df["cum_max"]
    if m == "ptp":
        return cum_df["cum_ptp"]
    raise ValueError(f"Unsupported metric for CSD/GCM plot: {metric!r}")


def _bin_metric_from_bins_df(bins_df: pd.DataFrame, metric: str) -> np.ndarray:
    """Extract the bin-level metric from the PAVA bins DataFrame."""
    m = metric.lower()
    if m == "count":
        return bins_df["n"].to_numpy(dtype=float)
    if m == "sum":
        return bins_df["sum"].to_numpy(dtype=float)
    if m == "mean":
        return bins_df["mean"].to_numpy(dtype=float)
    if m == "std":
        return bins_df["std"].to_numpy(dtype=float)
    if m == "var":
        # bins_df provides std; var = std^2
        std = bins_df["std"].to_numpy(dtype=float)
        return std * std
    if m == "min":
        return bins_df["min"].to_numpy(dtype=float)
    if m == "max":
        return bins_df["max"].to_numpy(dtype=float)
    if m == "ptp":
        return (bins_df["max"] - bins_df["min"]).to_numpy(dtype=float)
    raise ValueError(f"Unsupported metric for CSD/GCM plot: {metric!r}")


def plot_csd_gcm(
    pava: PAVA,
    *,
    metric: Optional[str] = None,
    title: Optional[str] = None,
    figsave_path: Optional[str] = None,
    dpi: int = 300,
) -> None:
    """Plot the CSD (blue) and GCM (red) for a fitted PAVA instance.

    Args:
        pava: A fitted `MOBPY.core.pava.PAVA` instance (`fit()` must have been called).
        metric: Metric to display. If None, uses `pava.metric`.
                One of {"count","mean","sum","std","var","min","max","ptp"}.
        title: Optional plot title; a default is generated if None.
        figsave_path: Optional path to save the figure.
        dpi: Save DPI if `figsave_path` is provided.

    Raises:
        RuntimeError: If `pava.fit()` has not been called.
        ValueError: If metric unsupported or required internals are missing.
    """
    if pava.groups_ is None:
        raise RuntimeError("PAVA must be fitted before plotting. Call pava.fit().")

    metric = (metric or pava.metric).lower()

    # 1) Build cumulative diagram (CSD) from grouped stats
    cum_df = _cum_stats_from_groups(pava.groups_)
    y_csd = _select_csd_series(cum_df, metric)

    # 2) Build GCM curve from the PAVA bins (piecewise constant by bin)
    bins_df = pava.bins_()
    right_edges = bins_df["right"].to_numpy(dtype=float)
    gcm_vals = _bin_metric_from_bins_df(bins_df, metric)

    # We will render GCM by sampling at bin right edges; it will look like
    # a step function (piecewise constant segments) sampled at ends.
    # To aid readability, we align the first red point with the first CSD x.
    x_csd = cum_df["x_value"].to_numpy(dtype=float)

    # 3) Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    # CSD: blue line with dot markers at each x_value
    ax.plot(x_csd, y_csd.to_numpy(dtype=float), "bo-", label="CSD")

    # GCM: red line at bin right edges with dot markers
    ax.plot(right_edges, gcm_vals, "ro-", label="GCM (PAVA fit)")

    # Scatter emphasis
    ax.scatter(x_csd, y_csd.to_numpy(dtype=float), color="blue")
    ax.scatter(right_edges, gcm_vals, color="red")

    # Optional interval labels near GCM points
    interval_labels = [f"[{l}, {r})" for l, r in zip(bins_df["left"], bins_df["right"])]
    for xe, ye, lab in zip(right_edges, gcm_vals, interval_labels):
        ax.annotate(
            lab,
            xy=(xe, ye),
            xytext=(2, -10),
            textcoords="offset points",
            ha="left",
            va="top",
            weight="bold",
            color="red",
            fontsize=9,
        )

    # Axes labels & title
    ax.set_xlabel(pava.x)
    ax.set_ylabel(metric)
    if title is None:
        title = f"PAVA CSD/GCM · x={pava.x} · y={pava.y} · metric='{metric}'"
    ax.set_title(title)

    ax.legend(loc="best")
    fig.tight_layout()

    if figsave_path:
        fig.patch.set_facecolor("white")
        plt.savefig(figsave_path, dpi=dpi)
    plt.show()
