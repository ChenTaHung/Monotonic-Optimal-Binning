from __future__ import annotations

import matplotlib
# Use a non-interactive backend for headless/test environments
matplotlib.use("Agg")  # safe no-op if already set
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MOBPlot:
    """Static plotting utilities for MOB-like summaries."""

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
        """Plot WoE bars + Bad Rate line from a MOB-style summary.

        Styling:
        - Bad Rate line = orange; points = red; point labels = red.
        - Data percentage text near WoE=0 baseline = black.
        - WoE value at bar tip = blue (above if positive, below if negative).
        - Legend placed at top-center; explanatory note sits between legend and chart.
        - Y-limits expanded to keep annotations inside the frame.

        Args:
            summary: DataFrame from `MonotonicBinner.summary_()` (binary y), with:
                     ['interval','nsamples','bads','goods','bad_rate','woe','iv_grp']
            figsize: Figure size in inches.
            dpi: Matplotlib DPI.
            bar_alpha: Alpha for WoE bars.
            bar_width: Width of bars.
            annotate: Whether to annotate values on the plot.
            title: Optional title string; defaults to IV total.
            savepath: Optional path to save the figure.
        """
        req = {"interval", "nsamples", "bads", "goods", "bad_rate", "woe", "iv_grp"}
        missing = req - set(summary.columns)
        if missing:
            raise ValueError(f"summary missing required columns: {sorted(missing)}")

        df = summary.copy().reset_index(drop=True)

        # Keep only numeric interval rows (skip Missing/Excluded)
        numeric_mask = df["interval"].astype(str).str.contains(r"[\[\(].*,.*\)").fillna(False)
        bins = df[numeric_mask].copy()
        if bins.empty:
            raise ValueError("No numeric bins found to plot.")

        x = np.arange(len(bins))
        intervals = bins["interval"].astype(str).tolist()

        woe = bins["woe"].to_numpy(dtype=float)
        bad_rate = bins["bad_rate"].to_numpy(dtype=float)

        # Observations share per bin (for data % labels)
        ns = bins["nsamples"].to_numpy(dtype=float)
        ns_share = ns / float(ns.sum()) if ns.sum() > 0 else np.zeros_like(ns)

        fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)

        # Reserve extra room at the top for legend and explanatory note,
        # and a bit at bottom for x tick labels.
        fig.subplots_adjust(top=0.78, bottom=0.18)

        # ---------------- WoE bars (primary y-axis) ---------------- #
        bars = ax1.bar(x, woe, width=bar_width, alpha=bar_alpha, label="WoE")
        ax1.axhline(0.0, linewidth=1.0, color="black", alpha=0.7)

        ax1.set_xticks(x, intervals, rotation=0)
        ax1.set_ylabel("WoE")

        # ---------------- Bad rate line (secondary y-axis) ---------------- #
        ax2 = ax1.twinx()
        # Line = orange, scatter = red
        line = ax2.plot(
            x, bad_rate, marker=None, linestyle="-", linewidth=2.0, color="#ff7f0e", label="Bad rate (line)"
        )
        pts = ax2.scatter(x, bad_rate, color="red", label="Bad rate (points)", zorder=3)
        ax2.set_ylabel("Bad Rate")

        # ---------------- Compute y-limits for WoE axis (to avoid clipping) ---------------- #
        if len(woe) > 0:
            w_abs_max = max(1e-6, float(np.max(np.abs(woe))))
            base_span = max(w_abs_max * 1.6, 0.6)  # ensure a minimum visual span
            top = max(woe.max(), 0.0) + 0.25 * w_abs_max + 0.15
            bot = min(woe.min(), 0.0) - 0.25 * w_abs_max - 0.15
            if top - bot < base_span:
                mid = 0.0
                top = mid + base_span / 2.0
                bot = mid - base_span / 2.0
            ax1.set_ylim(bot, top)

        # ---------------- Annotations ---------------- #
        if annotate:
            # 1) Annotate each Bad Rate point value (red text) above the point
            for xi, br in zip(x, bad_rate):
                ax2.annotate(
                    f"{br:.2%}",
                    xy=(xi, br),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="red",
                    weight="bold",
                )

            # 2) Data % near the WoE=0 baseline (black)
            y0 = 0.0
            ylim = ax1.get_ylim()
            y_span = (ylim[1] - ylim[0]) if (ylim[1] > ylim[0]) else 1.0
            base_offset = 0.03 * y_span

            for xi, share, w in zip(x, ns_share, woe):
                y_text = y0 + (base_offset if w >= 0 else -base_offset)
                ax1.annotate(
                    f"{share:.1%}",
                    xy=(xi, y_text),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    weight="bold",
                )

            # 3) WoE at the tip of the bar (blue)
            for xi, w in zip(x, woe):
                if w >= 0:
                    xytext = (0, 8)
                    va = "bottom"
                else:
                    xytext = (0, -8)
                    va = "top"
                ax1.annotate(
                    f"{w:.3f}",
                    xy=(xi, w),
                    xytext=xytext,
                    textcoords="offset points",
                    ha="center",
                    va=va,
                    fontsize=9,
                    color="#1f77b4",
                    weight="bold",
                )

        # ---------------- Titles / Legend / Note ---------------- #
        iv_total = float(bins["iv_grp"].sum())
        if title is None:
            title = f"Bins Summary (IV = {iv_total:.4f})"
        ax1.set_title(title)
        ax1.grid(True, axis="y", alpha=0.2)

        # Combined legend (bars + line + points), positioned at the top-center
        handles = [bars, line[0], pts]
        labels = ["WoE", "Bad rate (line)", "Bad rate (points)"]
        ax1.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.25),  # higher to create room for note
            ncol=3,
            frameon=False,
        )

        # Explanatory note positioned *between* legend and chart.
        # We place it in axes coordinates slightly above the top of the axes (y > 1).
        note = "Text colors – Blue: WoE at bar tip • Black: Data % near WoE=0 • Red: Bad Rate at points"
        ax1.text(
            0.5,
            1.12,  # between legend (~1.25) and axes top (1.0)
            note,
            transform=ax1.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="gray",
            clip_on=False,
        )

        # Final layout and optional save:
        # Use rect to keep reserved top space for legend + note and bottom for ticks.
        fig.tight_layout(rect=[0.04, 0.10, 1.00, 0.82])
        if savepath:
            fig.patch.set_facecolor("white")
            fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
