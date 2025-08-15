from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Color palette (explicit per your request)
SKYBLUE = "#87CEEB"
LIGHT_ORANGE = "#FDBA74"
STRONG_RED = "#EF4444"
GRID_COLOR = "#D1D5DB"


def _blocks_to_step_xy(blocks: Sequence[dict | tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Build (x, y) pairs to draw a right-continuous step from a list of blocks.

    Notes:
        - Blocks are assumed half-open [left, right) with (-inf, +inf) at extremes.
        - For plotting, we clamp infinities to just outside the min/max finite x-range.
    """
    # Normalize access
    def _get(b, k, idx):
        return b[k] if isinstance(b, dict) else b[idx]

    lefts = np.array([_get(b, "left", 0) for b in blocks], dtype=float)
    rights = np.array([_get(b, "right", 1) for b in blocks], dtype=float)
    ns = np.array([_get(b, "n", 2) for b in blocks], dtype=float)
    sums = np.array([_get(b, "sum", 3) for b in blocks], dtype=float)
    means = np.divide(sums, ns, out=np.zeros_like(sums), where=ns > 0)

    # Build right-continuous step: for each block append (left, m) -> (right, m)
    xs, ys = [], []
    for l, r, m in zip(lefts, rights, means):
        xs.extend([l, r])
        ys.extend([m, m])
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)

    # Clamp infinities for display purposes only
    finite_bounds = np.array([v for v in np.concatenate([lefts, rights]) if np.isfinite(v)], float)
    if finite_bounds.size:
        x_min, x_max = finite_bounds.min(), finite_bounds.max()
        span = max(1.0, x_max - x_min)
        xs = np.where(np.isneginf(xs), x_min - 0.05 * span, xs)
        xs = np.where(np.isposinf(xs), x_max + 0.05 * span, xs)
    return xs, ys


def plot_csd_pava_step(
    *,
    groups_df: pd.DataFrame,
    pava_blocks: List[dict] | List[tuple],
    merged_blocks: List[dict] | List[tuple] | None = None,
    x_name: str = "x",
    y_name: str = "y",
    figsize=(12, 7),
    dpi: int = 120,
    title: str | None = None,
    savepath: str | None = None,
) -> None:
    """Plot group means (points/line), PAVA step, and optional *merged* final step.

    Visual encodings (per your spec):
      - Group means (by unique x):           skyblue points + line
      - PAVA monotone step (pre-merge):      light orange step
      - Final merged step (post-merge):      strong red step overlay

    Args:
        groups_df: PAVA.groups_ (must contain 'x', 'sum', 'count').
        pava_blocks: Blocks returned directly by PAVA (before merge-adjacent).
        merged_blocks: Final blocks after merge-adjacent (optional).
        x_name, y_name: Axis labels.
        figsize, dpi, title, savepath: Matplotlib options.
    """
    req = {"x", "sum", "count"}
    if not req.issubset(groups_df.columns):
        raise ValueError(f"groups_df must include {sorted(req)}.")
    df = groups_df.sort_values("x").copy()
    means = df["sum"].to_numpy() / df["count"].to_numpy()
    xs = df["x"].to_numpy()

    # figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.6)

    # Group means (skyblue)
    ax.plot(xs, means, marker="o", color=SKYBLUE, label="Group mean", linewidth=1.5)

    # PAVA step (light orange)
    px, py = _blocks_to_step_xy(pava_blocks)
    ax.step(px, py, where="post", color=LIGHT_ORANGE, linewidth=2.5, label="PAVA step")

    # Final merged step (red), if provided
    if merged_blocks is not None and len(merged_blocks) > 0:
        mx, my = _blocks_to_step_xy(merged_blocks)
        ax.step(mx, my, where="post", color=STRONG_RED, linewidth=3.2, label="Merged (final)")

    ax.set_xlabel(x_name)
    ax.set_ylabel(f"mean({y_name})")
    if title is None:
        title = "CSD vs. PAVA step (and merged)"
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()

    if savepath:
        fig.patch.set_facecolor("white")
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def plot_gcm_on_csd(
    *,
    groups_df: pd.DataFrame,
    pava_blocks: List[dict] | List[tuple],
    x_name: str = "x",
    y_name: str = "y",
    figsize=(12, 7),
    dpi: int = 120,
    title: str | None = None,
    savepath: str | None = None,
) -> None:
    """Plot **GCM on CSD** exactly as requested.

    - Blue line (skyblue): cumulative mean over the sorted unique x groups.
      We rely on `PAVA.groups_` columns `cum_mean`, computed during `PAVA.fit()`.
    - Red line: PAVA result rendered as a right-continuous step function.
      Red labels annotate the `[left, right)` interval at each step's **right** endpoint.

    Args:
        groups_df: PAVA.groups_. Must include 'x' and 'cum_mean' columns.
        pava_blocks: PAVA blocks (list of dicts/tuples).
        x_name, y_name: Axis labels.
        figsize, dpi, title, savepath: Matplotlib options.
    """
    req = {"x", "cum_mean"}
    if not req.issubset(groups_df.columns):
        raise ValueError("groups_df must include 'x' and 'cum_mean' (added by PAVA.fit()).")

    df = groups_df.sort_values("x").copy()
    xs = df["x"].to_numpy(dtype=float)
    cum_mean = df["cum_mean"].to_numpy(dtype=float)

    # Step arrays for PAVA blocks
    sx, sy = _blocks_to_step_xy(pava_blocks)

    # figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6, alpha=0.6)

    # CSD cumulative mean (skyblue)
    ax.plot(xs, cum_mean, "o-", color=SKYBLUE, label="CSD (cumulative mean)")

    # PAVA (GCM) as step in red
    ax.step(sx, sy, where="post", color=STRONG_RED, linewidth=2.8, label="PAVA (GCM)")

    # Interval labels at right endpoints
    # Normalize accessors for left/right/mean
    def _get(b, k, idx):
        return b[k] if isinstance(b, dict) else b[idx]

    lefts = np.array([_get(b, "left", 0) for b in pava_blocks], dtype=float)
    rights = np.array([_get(b, "right", 1) for b in pava_blocks], dtype=float)
    ns = np.array([_get(b, "n", 2) for b in pava_blocks], dtype=float)
    sums = np.array([_get(b, "sum", 3) for b in pava_blocks], dtype=float)
    means = np.divide(sums, ns, out=np.zeros_like(sums), where=ns > 0)

    # For display: clamp +/-inf slightly outside the finite data range
    finite = np.array([v for v in np.concatenate([lefts, rights]) if np.isfinite(v)], float)
    if finite.size:
        x_min, x_max = finite.min(), finite.max()
        span = max(1.0, x_max - x_min)
        r_plot = np.where(np.isposinf(rights), x_max + 0.05 * span, rights)
        r_plot = np.where(np.isneginf(r_plot), x_min - 0.05 * span, r_plot)
    else:
        r_plot = rights

    def _fmt(v: float) -> str:
        if np.isneginf(v):
            return "-inf"
        if np.isposinf(v):
            return "inf"
        s = f"{v:.12g}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s

    for r, l, m in zip(r_plot, lefts, means):
        ax.annotate(
            f"[{_fmt(l)},{_fmt(r)})",
            xy=(r, m),
            xytext=(2, -10),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=9,
            color=STRONG_RED,
            weight="bold",
        )

    ax.set_xlabel(x_name)
    ax.set_ylabel(f"mean({y_name})")
    if title is None:
        title = f'Pool Adjacent Violators : <{x_name}, {y_name}> on "cumMean"'
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()

    if savepath:
        fig.patch.set_facecolor("white")
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
