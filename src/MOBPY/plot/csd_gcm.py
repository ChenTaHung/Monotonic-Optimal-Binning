from __future__ import annotations

import math
from typing import Iterable, List, Literal, Tuple

import matplotlib
# Non-interactive backend for CI / headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd


# =============================================================================
# Utilities shared by plots
# =============================================================================
def _clamp_inf_for_plot(x_step: np.ndarray, finite_x: np.ndarray) -> np.ndarray:
    """Clamp -inf / +inf to finite range for display only.

    Args:
        x_step: Array possibly containing ±∞ (bin edges from the model).
        finite_x: Finite x values used to derive a display range.

    Returns:
        A copy of `x_step` where ±∞ are mapped just outside the data span.
    """
    xmin = float(np.min(finite_x))
    xmax = float(np.max(finite_x))
    span = max(1.0, abs(xmax - xmin))
    out = np.array(x_step, dtype=float, copy=True)
    out = np.where(np.isneginf(out), xmin - 0.05 * span, out)
    out = np.where(np.isposinf(out), xmax + 0.05 * span, out)
    return out


def _ensure_groups_df(groups_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a sorted copy of groups_df."""
    if groups_df is None or len(groups_df) == 0:
        raise ValueError("groups_df is empty; run PAVA.fit() first.")
    req = {"x", "count", "sum"}
    if not req.issubset(groups_df.columns):
        raise ValueError(f"groups_df must contain columns: {sorted(req)}.")
    return groups_df.sort_values("x").reset_index(drop=True).copy()


# =============================================================================
# 1) CSD points + PAVA MEAN step function (on x–y plane)
# =============================================================================
def plot_csd_gcm(
    *,
    groups_df: pd.DataFrame,
    blocks: list[dict] | list[tuple],
    x_name: str = "x",
    y_name: str = "y",
    figsize: tuple[int, int] = (12, 7),
    dpi: int = 120,
    title: str | None = None,
    savepath: str | None = None,
) -> None:
    """Plot grouped means (CSD-like points) and the fitted PAVA step function.

    Note:
        PAVA now produces first/last blocks with edges (-inf, ...] and [..., +inf).
        We clamp ±∞ only for rendering; the model still stores infinite edges.

    Args:
        groups_df: PAVA.groups_ with columns ['x','count','sum', ...], sorted by x.
        blocks: Output of PAVA.export_blocks(as_dict=True) or tuples
                (left, right, n, sum, sum2, ymin, ymax).
        x_name: Label for x-axis.
        y_name: Label for y-axis.
        figsize: Matplotlib figure size (width, height).
        dpi: Figure DPI.
        title: Optional title (default: "CSD + PAVA step").
        savepath: Optional file path to save the figure (PNG, PDF, ...).
    """
    df = _ensure_groups_df(groups_df)
    means = df["sum"] / df["count"].clip(lower=1)
    x_finite = df["x"].to_numpy(dtype=float)

    # Build the "post" step from blocks (dicts or tuples).
    step_x: list[float] = []
    step_y: list[float] = []
    for blk in blocks:
        if isinstance(blk, dict):
            left, right, n, s = blk["left"], blk["right"], blk["n"], blk["sum"]
        else:  # tuple
            left, right, n, s, *_ = blk
        m = 0.0 if n == 0 else float(s) / float(n)
        # Append as post-step (horizontal segment)
        if not step_x:
            step_x.extend([left, right])
            step_y.extend([m, m])
        else:
            step_x.extend([left, right])
            step_y.extend([m, m])

    sx = _clamp_inf_for_plot(np.asarray(step_x, dtype=float), x_finite)
    sy = np.asarray(step_y, dtype=float)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # CSD-like points (x vs mean(y|x))
    ax.scatter(x_finite, means.to_numpy(dtype=float), label="Group means")

    # PAVA step (piecewise-constant fit)
    ax.step(sx, sy, where="post", label="PAVA fit (mean)")

    ax.set_xlabel(x_name)
    ax.set_ylabel(f"mean({y_name})")
    ax.set_title(title or "CSD + PAVA step")
    ax.legend(loc="best")
    ax.grid(alpha=0.3, linestyle=":")

    fig.tight_layout()
    if savepath:
        fig.patch.set_facecolor("white")
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 2) Greatest Convex Minorant (GCM) on the Cumulative Sum Diagram (U,V)
# =============================================================================
def _csd_from_groups(groups_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute the cumulative-sum diagram points (U,V).

    U_k = cumulative counts (starting at 0); V_k = cumulative sums of y.

    Args:
        groups_df: DataFrame with columns ['count','sum'] ordered by x.

    Returns:
        (U, V) arrays of length K+1 including the origin (0,0).
    """
    arr = groups_df[["count", "sum"]].to_numpy(dtype=float)
    counts = arr[:, 0]
    sums = arr[:, 1]
    U = np.concatenate([[0.0], np.cumsum(counts)])
    V = np.concatenate([[0.0], np.cumsum(sums)])
    return U, V


def _gcm_polyline_from_blocks(blocks: list[dict] | list[tuple]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the GCM polyline directly from PAVA blocks.

    Each block contributes a straight segment in (U,V) with slope = mean = sum/n.

    Args:
        blocks: PAVA blocks (dict or tuple form).

    Returns:
        Ug, Vg: polyline vertices in (U,V).
        slopes: per-segment slopes (block means).
    """
    n_list: list[float] = []
    s_list: list[float] = []
    for blk in blocks:
        if isinstance(blk, dict):
            n_list.append(float(blk["n"]))
            s_list.append(float(blk["sum"]))
        else:
            _, _, n, s, *_ = blk
            n_list.append(float(n))
            s_list.append(float(s))

    n_arr = np.asarray(n_list, dtype=float)
    s_arr = np.asarray(s_list, dtype=float)
    Ug = np.concatenate([[0.0], np.cumsum(n_arr)])
    Vg = np.concatenate([[0.0], np.cumsum(s_arr)])
    slopes = np.divide(s_arr, n_arr, out=np.zeros_like(s_arr), where=n_arr > 0)
    return Ug, Vg, slopes


def plot_gcm(
    *,
    groups_df: pd.DataFrame,
    blocks: list[dict] | list[tuple],
    figsize: tuple[int, int] = (10, 7),
    dpi: int = 120,
    title: str | None = None,
    show_points: bool = True,
    annotate_slopes: bool = True,
    savepath: str | None = None,
) -> None:
    """Plot the CSD (U,V) and its Greatest Convex Minorant (GCM) from PAVA blocks.

    The GCM is drawn as a piecewise-linear lower convex hull in the (U,V) plane;
    segment slopes equal the PAVA block means.

    Args:
        groups_df: PAVA.groups_ with columns ['count','sum'] sorted by x.
        blocks: PAVA.export_blocks(as_dict=True) or tuples.
        figsize: Figure size.
        dpi: Figure DPI.
        title: Optional plot title.
        show_points: If True, draw raw CSD points.
        annotate_slopes: If True, annotate each segment with its slope (=mean).
        savepath: Optional path to save the figure.
    """
    df = _ensure_groups_df(groups_df)
    U, V = _csd_from_groups(df)
    Ug, Vg, slopes = _gcm_polyline_from_blocks(blocks)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Raw CSD polyline + points
    ax.plot(U, V, linestyle="--", linewidth=1.2, alpha=0.8, label="CSD (cumulative)")
    if show_points:
        ax.scatter(U, V, s=24, zorder=3, label="CSD points")

    # GCM polyline
    ax.plot(Ug, Vg, linewidth=2.2, label="GCM (from PAVA)", zorder=4)

    if annotate_slopes and len(Ug) >= 2:
        for i in range(len(slopes)):
            u0, u1 = Ug[i], Ug[i + 1]
            v0, v1 = Vg[i], Vg[i + 1]
            um = 0.5 * (u0 + u1)
            vm = 0.5 * (v0 + v1)
            ax.text(um, vm, f"m={slopes[i]:.3g}", fontsize=9, ha="center", va="bottom")

    ax.set_xlabel("Cumulative count (U)")
    ax.set_ylabel("Cumulative sum of y (V)")
    ax.set_title(title or "Greatest Convex Minorant (GCM) on CSD")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    fig.tight_layout()
    if savepath:
        fig.patch.set_facecolor("white")
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_gcm_from_binner(binner, **kwargs) -> None:
    """Convenience wrapper to draw GCM directly from a fitted MonotonicBinner.

    Args:
        binner: Fitted MonotonicBinner (uses `._pava.groups_` and `._pava.blocks_`).
        **kwargs: Forwarded to :func:`plot_gcm`.

    Raises:
        RuntimeError: If the binner is not yet fitted.
    """
    pava = getattr(binner, "_pava", None)
    if pava is None or pava.groups_ is None:
        raise RuntimeError("binner must be fitted; call fit() first.")
    plot_gcm(groups_df=pava.groups_, blocks=pava.export_blocks(as_dict=True), **kwargs)


# =============================================================================
# 3) PAVA MERGE ANIMATION on the CSD/GCM (GIF)
# =============================================================================
class _TraceBlock:
    """Minimal block structure for reenacting PAVA merges (means-only)."""

    __slots__ = ("n", "s")

    def __init__(self, n: float, s: float) -> None:
        self.n = float(n)
        self.s = float(s)

    @property
    def mean(self) -> float:
        return 0.0 if self.n <= 0 else self.s / self.n

    def merge_with(self, other: "_TraceBlock") -> "_TraceBlock":
        return _TraceBlock(self.n + other.n, self.s + other.s)


def _resolve_sign_from_groups(groups_df: pd.DataFrame, sign: Literal["+", "-", "auto"]) -> Literal["+", "-"]:
    """Match PAVA's sign resolution: '+' if corr(x, mean_y) >= 0 else '-'."""
    if sign in {"+", "-"}:
        return sign  # explicit
    # auto mode
    x = groups_df["x"].to_numpy(dtype=float)
    means = (groups_df["sum"] / groups_df["count"].clip(lower=1)).to_numpy(dtype=float)
    corr = np.corrcoef(x, means)[0, 1] if len(x) > 1 else 1.0
    return "+" if (not np.isfinite(corr) or corr >= 0) else "-"


def _pava_snapshots_from_groups(
    groups_df: pd.DataFrame,
    *,
    sign: Literal["+", "-", "auto"] = "auto",
    strict: bool = True,
    tol: float = 1e-12,
) -> list[list[_TraceBlock]]:
    """Reenact the stack-based PAVA to capture snapshots after each merge.

    This mirrors the algorithm used in `MOBPY.core.pava.PAVA.fit()` (means-only).

    Args:
        groups_df: PAVA.groups_ (sorted by x) with columns ['count','sum'].
        sign: '+', '-', or 'auto' (match PAVA logic).
        strict: If True, merge plateaus (equal means) to enforce strict monotone.
        tol: Numerical tolerance for plateau detection.

    Returns:
        A list of snapshots; each snapshot is the current list of blocks
        (each block stores only (n, s)). The final snapshot is the final PAVA solution.
    """
    df = _ensure_groups_df(groups_df)
    rsign: Literal["+", "-"] = _resolve_sign_from_groups(df, sign)

    # Initial atomic blocks: one per unique x with its (count, sum)
    init: list[_TraceBlock] = [
        _TraceBlock(n=float(c), s=float(s)) for c, s in df[["count", "sum"]].to_numpy()
    ]

    snapshots: list[list[_TraceBlock]] = [ [ _TraceBlock(b.n, b.s) for b in init ] ]  # include initial state
    stack: list[_TraceBlock] = []

    def violates(b1: _TraceBlock, b2: _TraceBlock) -> bool:
        if rsign == "+":
            return b2.mean < b1.mean - tol
        else:
            return b2.mean > b1.mean + tol

    for b in init:
        stack.append(b)
        # While the top two blocks violate monotonicity, merge them.
        while len(stack) >= 2:
            b2 = stack[-1]
            b1 = stack[-2]
            if violates(b1, b2):
                merged = b1.merge_with(b2)
                stack.pop(); stack.pop()
                stack.append(merged)
                snapshots.append([_TraceBlock(x.n, x.s) for x in stack])
            else:
                # Optional plateau merging for strict monotonicity
                if strict and abs(b2.mean - b1.mean) <= tol:
                    merged = b1.merge_with(b2)
                    stack.pop(); stack.pop()
                    stack.append(merged)
                    snapshots.append([_TraceBlock(x.n, x.s) for x in stack])
                else:
                    break

    # Ensure final solution is recorded
    if not snapshots or snapshots[-1] != stack:
        snapshots.append([_TraceBlock(x.n, x.s) for x in stack])
    return snapshots


def _polyline_from_trace_blocks(blocks: list[_TraceBlock]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a list of trace blocks into a (U,V) polyline and slopes."""
    n_arr = np.array([b.n for b in blocks], dtype=float)
    s_arr = np.array([b.s for b in blocks], dtype=float)
    Ug = np.concatenate([[0.0], np.cumsum(n_arr)])
    Vg = np.concatenate([[0.0], np.cumsum(s_arr)])
    slopes = np.divide(s_arr, n_arr, out=np.zeros_like(s_arr), where=n_arr > 0)
    return Ug, Vg, slopes


def animate_pava_gif(
    *,
    groups_df: pd.DataFrame,
    sign: Literal["+", "-", "auto"] = "auto",
    strict: bool = True,
    figsize: tuple[int, int] = (8, 6),
    dpi: int = 120,
    fps: float = 1.25,
    annotate_slopes: bool = True,
    savepath: str = "pava_gcm.gif",
) -> None:
    """Create a GIF animating the PAVA pooling on the CSD/GCM.

    The animation shows:
      1) The CSD (U,V) cumulative polyline and points.
      2) The evolving GCM polyline as adjacent violators are merged step-by-step.

    Args:
        groups_df: PAVA.groups_ with at least ['x','count','sum'] columns.
        sign: '+', '-', or 'auto' (match PAVA).
        strict: If True, merge equal-mean plateaus.
        figsize: Matplotlib figure size.
        dpi: Figure DPI.
        fps: Frames per second in the output GIF.
        annotate_slopes: If True, annotate segment slopes each frame.
        savepath: Output GIF filename (e.g. 'pava_gcm.gif').

    Raises:
        ValueError: If `groups_df` is invalid.

    Note:
        Uses the Pillow writer under the hood. Ensure pillow is installed.
    """
    df = _ensure_groups_df(groups_df)
    U, V = _csd_from_groups(df)
    snapshots = _pava_snapshots_from_groups(df, sign=sign, strict=strict)

    # Precompute polylines for each snapshot to keep the draw function small
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [
        _polyline_from_trace_blocks(s) for s in snapshots
    ]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Draw static CSD background
    (csd_line,) = ax.plot(U, V, linestyle="--", linewidth=1.2, alpha=0.8, label="CSD (cumulative)")
    csd_pts = ax.scatter(U, V, s=24, zorder=3, label="CSD points")

    # Dynamic GCM line and (optional) text annotations
    (gcm_line,) = ax.plot([], [], linewidth=2.2, label="GCM (PAVA evolving)", zorder=4)
    slope_texts: list[matplotlib.text.Text] = []

    def init():
        gcm_line.set_data([], [])
        # Clear any old texts
        for t in slope_texts:
            t.remove()
        slope_texts.clear()
        return (gcm_line,)

    def update(frame_idx: int):
        Ug, Vg, slopes = frames[frame_idx]
        gcm_line.set_data(Ug, Vg)

        # Update slope annotations
        for t in slope_texts:
            t.remove()
        slope_texts.clear()
        if annotate_slopes and len(Ug) >= 2:
            for i in range(len(slopes)):
                u0, u1 = Ug[i], Ug[i + 1]
                v0, v1 = Vg[i], Vg[i + 1]
                um = 0.5 * (u0 + u1)
                vm = 0.5 * (v0 + v1)
                slope_texts.append(ax.text(um, vm, f"m={slopes[i]:.3g}", fontsize=9, ha="center", va="bottom"))

        ax.set_xlim(0, max(1.0, U[-1]) * 1.05)
        v_max = max(1.0, V[-1]) if np.isfinite(V[-1]) else max(1.0, np.nanmax(V))
        ax.set_ylim(0, v_max * 1.05)
        return (gcm_line, *slope_texts)

    ax.set_xlabel("Cumulative count (U)")
    ax.set_ylabel("Cumulative sum of y (V)")
    ax.set_title("PAVA merges on CSD → evolving GCM")
    ax.legend(loc="best")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)

    fig.tight_layout()

    # Build animation and save as GIF
    # Pillow is the writer backend — make sure `pillow` is available at runtime.
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(frames),
        interval=1000.0 / max(fps, 0.1),
        blit=False,
        repeat=False,
    )
    anim.save(savepath, writer="pillow", dpi=dpi)
    plt.close(fig)


def animate_pava_from_binner(binner, **kwargs) -> None:
    """Animate PAVA merges directly from a fitted MonotonicBinner.

    This helper is equivalent to:

        animate_pava_gif(groups_df=binner._pava.groups_, ...)

    Args:
        binner: Fitted MonotonicBinner (needs `._pava.groups_`).
        **kwargs: Forwarded to :func:`animate_pava_gif`.

    Raises:
        RuntimeError: If the binner is not fitted yet.
    """
    pava = getattr(binner, "_pava", None)
    if pava is None or pava.groups_ is None:
        raise RuntimeError("binner must be fitted; call fit() first.")
    # Prefer using the resolved sign from the fitted PAVA if not overridden
    if "sign" not in kwargs and getattr(pava, "resolved_sign_", None) in {"+", "-"}:
        kwargs = dict(kwargs)
        kwargs["sign"] = pava.resolved_sign_
    animate_pava_gif(groups_df=pava.groups_, **kwargs)


__all__ = [
    "plot_csd_gcm",
    "plot_gcm",
    "plot_gcm_from_binner",
    "animate_pava_gif",
    "animate_pava_from_binner",
]
