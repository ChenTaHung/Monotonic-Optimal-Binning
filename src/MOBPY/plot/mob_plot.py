from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for tests/CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MOBPlot:
    """Static plotting utilities for MOB-style summaries."""

    @staticmethod
    def plot_bins_summary(
        summary: pd.DataFrame,
        *,
        figsize=(12, 7),
        dpi: int = 120,
        bar_alpha: float = 0.55,
        bar_width: float = 0.6,
        annotate: bool = True,
        title: str | None = None,
        savepath: str | None = None,
    ) -> None:
        """Plot WoE bars + bad-rate line from a MOB-style summary.

        Args:
            summary: DataFrame returned by `MonotonicBinner.summary_()` when `y` is binary.
                     Must contain:
                       ['interval','nsamples','bads','goods','bad_rate','woe','iv_grp']
            figsize: Figure size in inches.
            dpi: Matplotlib DPI.
            bar_alpha: Alpha for WoE bars.
            bar_width: Width of bars.
            annotate: If True, annotate WoE bars with sample proportions and dots with bad-rate.
            title: Optional title string; defaults to total IV.
            savepath: Optional path to save the figure.

        Raises:
            ValueError: If required columns missing or no numeric bins to plot.
        """
        req = {"interval", "nsamples", "bads", "goods", "bad_rate", "woe", "iv_grp"}
        missing = req - set(summary.columns)
        if missing:
            raise ValueError(f"summary missing required columns: {sorted(missing)}")

        df = summary.copy().reset_index(drop=True)

        # Numeric bins: those with finite left/right (Missing/Excluded have NaN edges)
        numeric = df[df["interval"].notna() & df["left"].notna() & df["right"].notna()].copy()
        if numeric.empty:
            raise ValueError("No numeric bins found to plot.")

        x = np.arange(len(numeric))
        intervals = numeric["interval"].astype(str).tolist()
        woe = numeric["woe"].to_numpy(dtype=float)
        bad_rate = numeric["bad_rate"].to_numpy(dtype=float)

        fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
        bars = ax1.bar(x, woe, width=bar_width, alpha=bar_alpha)
        ax1.axhline(0.0, linewidth=1)
        ax1.set_xticks(x, intervals, rotation=0)
        ax1.set_ylabel("WoE")

        # annotate WoE bars with obs distribution
        if annotate and "nsamples" in numeric.columns:
            dist = numeric["nsamples"].to_numpy(dtype=float)
            dist = dist / (dist.sum() if dist.sum() > 0 else 1.0)
            for xi, bar, d in zip(x, bars, dist):
                h = bar.get_height()
                va = "bottom" if h >= 0 else "top"
                yoff = 0.01 if h >= 0 else -0.01
                ax1.text(xi, h + yoff, f"{d:.1%}", ha="center", va=va, fontsize=9)

        ax2 = ax1.twinx()
        ax2.plot(x, bad_rate, marker="o")
        ax2.set_ylabel("Bad Rate")

        iv_total = float(numeric["iv_grp"].sum())
        ax1.set_title(title or f"Bins Summary (IV = {iv_total:.4f})")

        fig.tight_layout()
        if savepath:
            fig.patch.set_facecolor("white")
            fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
