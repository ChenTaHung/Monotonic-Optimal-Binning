# src/MOBPY/plot/mob_plot.py
"""
Plot helpers for monotone bins (MOB/PAVA outputs).

This module provides a small, dependency-light plotting utility that accepts the
DataFrame returned by `MonotonicBinner.summary_()` (recommended) or, with minor
limitations, `MonotonicBinner.bins_()`.

Primary chart
-------------
- `plot_bins_summary(df, ...)`:
    - Bars: WoE (if binary) or observations share (fallback)
    - Line: bad rate (binary) or mean (numeric)
    - X-axis: interval labels in left-closed, right-open form "[a, b)"

Robustness & compatibility
--------------------------
- Accepts either full summary (includes WoE/IV) or just clean bins.
- If WoE is missing (numeric targets), bars fallback to `dist_obs`.
- Validates required columns and raises helpful ValueError messages.

Example
-------
>>> from MOBPY.plot.mob_plot import MOBPlot
>>> df = binner.summary_()  # from MonotonicBinner.fit().summary_()
>>> MOBPlot.plot_bins_summary(df, title="Duration in Month")
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MOBPlot:
    """Static plotting helpers for MOB/PAVA outputs."""

    @staticmethod
    def _ensure_interval_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the frame has a string interval label column.

        Strategy:
            - If '[intervalStart' and 'intervalEnd)' exist, build 'interval' from them.
            - Else if 'left'/'right' exist, build from numeric edges.
            - Else, require an existing 'interval' string column.

        Returns:
            A shallow copy of `df` with an 'interval' column.

        Raises:
            ValueError: if interval information is not present.
        """
        out = df.copy()
        if "[intervalStart" in out.columns and "intervalEnd)" in out.columns:
            # Already string-like in summary_
            lefts = out["[intervalStart"].astype(str).to_numpy()
            rights = out["intervalEnd)"].astype(str).to_numpy()
            labels = np.array([f"[{l}, {r})" for l, r in zip(lefts, rights)], dtype=object)
            # legacy: first interval uses "(" if it's -inf; keep consistent with input
            labels[0] = labels[0].replace("[", "(") if "-inf" in labels[0] else labels[0]
            out["interval"] = labels
            return out

        if "left" in out.columns and "right" in out.columns:
            lefts = out["left"].to_numpy()
            rights = out["right"].to_numpy()
            labels = np.array([f"[{l}, {r})" for l, r in zip(lefts, rights)], dtype=object)
            # if first left is -inf, render with "("
            if len(labels) and np.isneginf(lefts[0]):
                labels[0] = labels[0].replace("[", "(")
            out["interval"] = labels
            return out

        if "interval" in out.columns:
            return out

        raise ValueError(
            "Cannot construct interval labels. Expected columns "
            "`['[intervalStart','intervalEnd)']` or `['left','right']` or an "
            "existing 'interval' column."
        )

    @staticmethod
    def _is_binary_summary(df: pd.DataFrame) -> bool:
        """Heuristic: summary contains MOB columns if WoE is present."""
        return "woe" in df.columns or "bads" in df.columns or "rate" in df.columns

    @staticmethod
    def plot_bins_summary(
        df: pd.DataFrame,
        *,
        title: Optional[str] = None,
        figsave_path: Optional[str] = None,
        dpi: int = 300,
        bar_alpha: float = 0.5,
        bar_width: float = 0.5,
    ) -> None:
        """Plot a compact summary of bins.

        Bars show WoE (if available) or distribution of observations as a fallback.
        Line shows bad rate (binary) or mean (numeric).

        Args:
            df: DataFrame from `MonotonicBinner.summary_()` (preferred) or `bins_()`.
            title: Optional chart title; if None, a sensible default is used.
            figsave_path: Optional path to save the figure.
            dpi: Save DPI if `figsave_path` is provided.
            bar_alpha: Alpha (opacity) for bars.
            bar_width: Width of bars.

        Raises:
            ValueError: If the input does not contain the necessary columns.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("`df` must be a pandas DataFrame.")

        # Work on a copy and ensure interval labels exist
        data = MOBPlot._ensure_interval_columns(df)

        # Deduce whether we have binary-style output (WoE, bad rate)
        is_binary = MOBPlot._is_binary_summary(data)

        # Choose series to plot
        # Bars prefer WoE when available; otherwise fallback to distribution of obs.
        if is_binary and "woe" in data.columns:
            bar_series = data["woe"].astype(float)
            bar_label = "WoE"
        else:
            # fallback: dist of observations if present; else compute from 'n'
            if "dist_obs" in data.columns:
                bar_series = data["dist_obs"].astype(float)
            elif "n" in data.columns:
                n = data["n"].astype(float)
                bar_series = n / n.sum() if n.sum() > 0 else n * np.nan
            else:
                raise ValueError("Cannot determine bar series: need 'woe' or 'dist_obs' or 'n'.")
            bar_label = "Obs. share"

        # Line series: bad rate (binary) or mean (numeric)
        if is_binary and "rate" in data.columns:
            line_series = data["rate"].astype(float)
            line_label = "Bad rate"
        elif "mean" in data.columns:
            line_series = data["mean"].astype(float)
            line_label = "Mean"
        else:
            raise ValueError("Cannot determine line series: need 'rate' (binary) or 'mean'.")

        # X labels
        labels = data["interval"].astype(str).to_list()
        x = np.arange(len(labels))

        # Create plot
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        bars = ax1.bar(x, bar_series.to_numpy(), width=bar_width, alpha=bar_alpha)
        ax1.set_xticks(x, labels, rotation=0)
        ax1.axhline(0, linewidth=1)
        ax1.set_ylabel(bar_label)

        # Annotate bar with obs share if available
        if "dist_obs" in data.columns:
            dist = data["dist_obs"].astype(float).to_numpy()
            for i, bar in enumerate(bars):
                height = bar.get_height()
                if np.isnan(height):
                    continue
                # Position text above/below depending on sign
                y_off = 10 if height >= 0 else -8
                ax1.annotate(
                    f"{dist[i]:.1%}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, y_off),
                    textcoords="offset points",
                    ha="center",
                    va="top" if height >= 0 else "bottom",
                    weight="bold",
                )

        # Secondary axis for the line
        ax2 = ax1.twinx()
        ax2.plot(x, line_series.to_numpy(), linewidth=2, marker="o")
        ax2.set_ylabel(line_label)

        # Title
        if title is None:
            # If IV is available, put it in the title
            iv_val = None
            if "iv_grp" in data.columns:
                try:
                    iv_val = float(np.nansum(data["iv_grp"].to_numpy(dtype=float)))
                except Exception:
                    iv_val = None
            title = "Bins Summary"
            if iv_val is not None and np.isfinite(iv_val):
                title += f"  ·  IV={iv_val:.4f}"
        ax1.set_title(title)

        # Legend hint
        ax1.legend([bar_label, line_label], loc="lower center")

        fig.tight_layout()

        if figsave_path:
            fig.patch.set_facecolor("white")
            plt.savefig(figsave_path, dpi=dpi)
        plt.show()
