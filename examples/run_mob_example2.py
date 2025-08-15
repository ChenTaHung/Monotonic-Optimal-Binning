#%% ===== Demo: MOBPY with insurance3r2.csv (age vs. insuranceclaim) =====
import os
from pathlib import Path
import numpy as np
import pandas as pd

# If you’re running this as a script from anywhere:
# repo_root = Path(__file__).resolve().parents[0]  # script folder
repo_root = Path("/Users/chentahung/Desktop/git/mob-py")
src_dir = repo_root / "src"
data_dir = repo_root / "data"
out_dir = repo_root / "doc" / "images"
out_dir.mkdir(parents=True, exist_ok=True)

import sys
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Non-interactive backend is already set in plotting modules; we just savefig.
from MOBPY.binning.mob import MonotonicBinner
from MOBPY.core.constraints import BinningConstraints
from MOBPY.plot.mob_plot import MOBPlot
from MOBPY.plot.csd_gcm import plot_csd_pava_step, plot_gcm_on_csd

#%% ---------- 1) Load data ----------
csv_path = data_dir / "insurance3r2.csv"
df = pd.read_csv(csv_path)

# Ensure expected columns
assert {"age", "insuranceclaim"}.issubset(df.columns), \
    f"Columns not found in {csv_path.name}"

# We’ll bin age against binary target insuranceclaim
x_col = "age"
y_col = "insuranceclaim"

# Optional clean-up (clip target to {0,1} if needed)
if df[y_col].notna().any():
    # make sure it's 0/1 int
    df[y_col] = (df[y_col] > 0).astype(int)

#%% ---------- 2) Fit MOBPY ----------
cons = BinningConstraints(
    max_bins=6,
    min_bins=2,
    min_samples=0.05,      # at least 5% of clean rows per bin if feasible
    initial_pvalue=0.5,    # merge threshold
    maximize_bins=True,    # keep #bins ≤ max_bins (classic MOB)
)

binner = MonotonicBinner(
    df=df,
    x=x_col,
    y=y_col,
    metric="mean",
    sign="auto",           # infer monotone direction
    strict=True,
    constraints=cons,
    exclude_values=None,
).fit()

bins = binner.bins_()
summary = binner.summary_()
print("Bins:")
print(bins)
print("\nSummary:")
print(summary)

#%% ---------- 3) Save MOB-style summary plot ----------
bins_png = out_dir / "bins_summary.png"
MOBPlot.plot_bins_summary(summary, savepath=str(bins_png), dpi=180)
print(f"Saved: {bins_png}")

#%% ---------- 4) Build 'merged_blocks' from final bins for step plotting ----------
# Our step-plot expects blocks with [left, right, n, sum] to compute means.
# bins already has left/right/n/sum, so we can wrap each row into a dict.
def bins_df_to_blocks(rows: pd.DataFrame):
    out = []
    for _, r in rows.iterrows():
        out.append(
            {
                "left": float(r["left"]),
                "right": float(r["right"]),
                "n": int(r["n"]),
                "sum": float(r["sum"]),
                # sum2/ymin/ymax are not needed for plotting—but we can pass dummies
                "sum2": float(r["std"]**2 * max(1, int(r["n"]) - 1) + (r["sum"]**2)/max(1, int(r["n"]))),
                "ymin": float(r["min"]),
                "ymax": float(r["max"]),
            }
        )
    return out

pava_blocks = binner._pava.export_blocks(as_dict=True)        # BEFORE merge
merged_blocks = bins_df_to_blocks(bins)                       # AFTER merge

#%% ---------- 5) Plot: CSD (group means) + PAVA step + merged step ----------
csd_pava_png = out_dir / "csd_pava_step.png"
plot_csd_pava_step(
    groups_df=binner._pava.groups_,        # needs columns x,sum,count
    pava_blocks=pava_blocks,
    merged_blocks=merged_blocks,
    x_name=x_col,
    y_name=y_col,
    title=f"CSD & PAVA & Merged: {x_col} vs {y_col}",
    savepath=str(csd_pava_png),
    dpi=180,
)
print(f"Saved: {csd_pava_png}")

#%% ---------- 6) Plot: GCM on CSD (cumulative mean vs. PAVA step) ----------
gcm_png = out_dir / "gcm_on_csd.png"
plot_gcm_on_csd(
    groups_df=binner._pava.groups_,        # needs columns x,cum_mean (added in PAVA.fit)
    pava_blocks=pava_blocks,
    x_name=x_col,
    y_name=y_col,
    title=f"GCM on CSD: {x_col} vs {y_col}",
    savepath=str(gcm_png),
    dpi=180,
)
print(f"Saved: {gcm_png}")

print("Demo complete.")

# %%
