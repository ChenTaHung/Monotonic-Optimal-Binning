# demo_mobpy_german.py
import os
from pathlib import Path
import math

import numpy as np
import pandas as pd

from MOBPY.binning.mob import MonotonicBinner
from MOBPY.core.constraints import BinningConstraints
from MOBPY.plot.csd_gcm import (
    plot_csd_pava_step,
    plot_gcm_from_binner,
    animate_pava_from_binner,
    plot_gcm_on_means
)

# -----------------------------------------------------------------------------
# 0) Paths
# -----------------------------------------------------------------------------
REPO = Path("/Users/chentahung/Desktop/git/mob-py").resolve()
DATA1 = REPO / "data" / "german_data_credit_cat.csv"  # if present
IMG_DIR = REPO / "doc" / "images"
GIF_DIR = IMG_DIR / "gif"
IMG_DIR.mkdir(parents=True, exist_ok=True)
GIF_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1) Load data (German credit if available, else synthetic)
# -----------------------------------------------------------------------------
if DATA1.exists():
    df = pd.read_csv(DATA1)
    # German dataset note: target is (default - 1) -> {0,1}
    df["default"] = (df["default"] - 1).clip(lower=0, upper=1).astype(int)
    x_col = "credit_amount" if "credit_amount" in df.columns else df.select_dtypes("number").columns[0]
    y_col = "default"
else:
    # Synthetic binary demo
    rng = np.random.default_rng(7)
    n = 800
    x = np.linspace(-2.0, 3.0, n) + rng.normal(0, 0.25, n)
    p = 1.0 / (1.0 + np.exp(-1.3 * x))
    y = rng.binomial(1, p)
    df = pd.DataFrame({"x": x, "y": y})
    x_col, y_col = "x", "y"

print(f"Using x={x_col!r}, y={y_col!r}; rows={len(df)}")

# -----------------------------------------------------------------------------
# 2) Set constraints & fit binner
# -----------------------------------------------------------------------------
cons = BinningConstraints(
    max_bins=6,
    min_bins=2,
    min_samples=0.05,   # 5% of clean rows per bin (auto-resolved)
    initial_pvalue=0.4, # merge threshold (simple two-sample test heuristic)
    maximize_bins=True, # classic MOB: don't exceed max_bins
)

binner = MonotonicBinner(
    df=df,
    x=x_col,
    y=y_col,
    metric="mean",          # (future work: median; currently mean-only)
    sign="auto",            # infer monotone direction
    strict=True,            # merge plateaus during PAVA
    constraints=cons,
    exclude_values=None,    # e.g., special codes to exclude as their own rows
).fit()

# -----------------------------------------------------------------------------
# 3) Inspect results
# -----------------------------------------------------------------------------
bins = binner.bins_()
summary = binner.summary_()
print("\n=== Clean bins ===")
print(bins)
print("\n=== Full summary ===")
print(summary)

# Check coverage convention: first left is -inf, last right is +inf
first_left = bins["left"].iloc[0]
last_right = bins["right"].iloc[-1]
assert math.isinf(first_left) and first_left < 0
assert math.isinf(last_right) and last_right > 0

# -----------------------------------------------------------------------------
# 4) Plots
# -----------------------------------------------------------------------------
# 4a) CSD-like group means + PAVA step (x–y plane)
plot_csd_pava_step(
    groups_df=binner._pava.groups_,
    blocks=binner._pava.export_blocks(as_dict=True),
    x_name=x_col,
    y_name=y_col,
    savepath=str(IMG_DIR / "csd_pava_step.png"),
)

# 4b) GCM (Greatest Convex Minorant) on the Cumulative Sum Diagram
plot_gcm_from_binner(
    binner,
    savepath=str(IMG_DIR / "gcm_on_csd.png"),
    annotate_intervals=True
)

# 4c) MOB summary plot (WoE bars + bad-rate line) — only for binary y
if getattr(binner, "_is_binary_y", False):
    from MOBPY.plot.mob_plot import MOBPlot
    MOBPlot.plot_bins_summary(
        binner.summary_(),
        savepath=str(IMG_DIR / "mob_summary.png"),
    )
    print("Saved: mob_summary.png")
else:
    print("Non-binary target: skipping WoE/IV summary plot.")

print("Saved: csd_pava_step.png, gcm_on_csd.png")

# -----------------------------------------------------------------------------
# 5) Animation: PAVA merges → evolving GCM (GIF)
# -----------------------------------------------------------------------------
animate_pava_from_binner(
    binner,
    savepath=str(GIF_DIR / "pava_gcm.gif"),
    fps=1.25,
    annotate_slopes=True,
)
print("Saved GIF:", GIF_DIR / "pava_gcm.gif")

plot_gcm_on_means(
    groups_df=binner._pava.groups_,
    blocks=binner._pava.export_blocks(as_dict=True),
    x_name=binner.x,
    y_name=binner.y,
    savepath=str(GIF_DIR / "gcm_means.png")
)
print("Saved: gcm_means.png")