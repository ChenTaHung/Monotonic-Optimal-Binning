from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MOBPY.core.pava import PAVA


def plot_csd_gcm(
    *,
    groups_df: pd.DataFrame,
    blocks: list[dict] | list[tuple],
    x_name: str = "x",
    y_name: str = "y",
    figsize=(12, 7),
    dpi: int = 120,
    title: str | None = None,
    savepath: str | None = None,
) -> None:
    """Visualize group means (CSD-like points) and fitted PAVA step function.

    Plot elements:
      * Scatter of group means vs x (one point per unique x).
      * Step function of the PAVA-fitted piecewise constant means across [left, right).

    Args:
        groups_df: PAVA.groups_ with columns {'x','count','sum',...}.
        blocks: PAVA.export_blocks(as_dict=True) (dicts) or tuples
                of (left, right, n, sum, sum2, ymin, ymax).
        x_name: Label for x-axis.
        y_name: Label for y-axis.
        figsize: Figure size.
        dpi: Figure DPI.
        title: Optional plot title.
        savepath: Optional path to save the figure to disk.

    Raises:
        ValueError: If groups_df missing or lacks required columns.
    """
    if groups_df is None or groups_df.empty:
        raise ValueError("groups_df is empty; run PAVA.fit() first.")
    if not {"x", "count", "sum"}.issubset(groups_df.columns):
        raise ValueError("groups_df must contain columns: {'x','count','sum'}.")

    df = groups_df.copy().sort_values("x")
    means = df["sum"] / df["count"]

    # Build step series from blocks
    step_x: list[float] = []
    step_y: list[float] = []
    for blk in blocks:
        if isinstance(blk, dict):
            left, right, n, s = blk["left"], blk["right"], blk["n"], blk["sum"]
        else:
            left, right, n, s, *_ = blk
        m = 0.0 if n == 0 else s / n
        if not step_x:
            step_x.extend([left, right])
            step_y.extend([m, m])
        else:
            step_x.extend([left, right])
            step_y.extend([m, m])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Scatter group means
    ax.scatter(df["x"].to_numpy(), means.to_numpy(), label="Group means")

    # Step function (mask +inf for plotting endcap)
    sx = np.array(step_x, dtype=float)
    sy = np.array(step_y, dtype=float)
    if np.isposinf(sx).any():
        finite_x = df["x"].to_numpy()
        xmax = float(finite_x.max())
        # push the +inf step slightly to the right for visibility
        sx = np.where(np.isposinf(sx), xmax + 0.05 * max(1.0, abs(xmax)), sx)
    ax.step(sx, sy, where="post", label="PAVA step")

    ax.set_xlabel(x_name)
    ax.set_ylabel(f"mean({y_name})")
    ax.set_title(title or "CSD/GCM Visualization")
    ax.legend(loc="best")
    fig.tight_layout()

    if savepath:
        fig.patch.set_facecolor("white")
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_csd_gcm_from_binner(binner, **kwargs) -> None:
    """Convenience wrapper to plot from a fitted MonotonicBinner."""
    if getattr(binner, "_pava", None) is None:
        raise RuntimeError("binner must be fitted; call fit() first.")
    pava: PAVA = binner._pava
    plot_csd_gcm(
        groups_df=pava.groups_,
        blocks=pava.export_blocks(as_dict=True),
        x_name=binner.x,
        y_name=binner.y,
        **kwargs,
    )
