#!/usr/bin/env python
# examples/run_mob_example.py
"""
Quick demo of the new MOB/PAVA pipeline on the German Credit dataset.

- Reads the CSV
- Converts `default` from {1,2} -> {0,1} by subtracting 1
- Runs PAVA + constraints-aware merging (MOB special case)
- Prints clean bins and the MOB-style summary (with WoE/IV)
- (Optional) Plots the bins summary and the CSD/GCM visualization

Run:
    python examples/run_mob_example.py \
        --csv /data/german_data_credit_cat.csv \
        --x Durationinmonth \
        --y default \
        --plot

Notes:
- The defaults mirror your prior settings: max_bins=6, min_bins=4,
  max_samples=0.4, min_samples=0.05, min_positives=0.05, initial_pvalue=0.4.
- For exclusions/missing handling, pass --exclude 999 or multiple --exclude values.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import pandas as pd

from MOBPY.core.constraints import BinningConstraints
from MOBPY.binning.mob import MonotonicBinner
from MOBPY.plot.mob_plot import MOBPlot
from MOBPY.plot.csd_gcm import plot_csd_gcm
from MOBPY.core.pava import PAVA


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run monotone optimal binning (MOB) demo.")
    p.add_argument("--csv", type=str, default="/data/german_data_credit_cat.csv",
                   help="Path to german_data_credit_cat.csv")
    p.add_argument("--x", type=str, default="Durationinmonth", help="Feature/variable to bin.")
    p.add_argument("--y", type=str, default="default", help="Response column (binary for MOB).")
    p.add_argument("--exclude", action="append", default=None,
                   help="Values in x to treat as special bins. Use multiple flags for multiple values.")
    p.add_argument("--max-bins", type=int, default=6, help="Maximum number of bins.")
    p.add_argument("--min-bins", type=int, default=4, help="Minimum number of bins.")
    p.add_argument("--max-samples", type=float, default=0.4,
                   help="Max samples per bin (absolute int or fraction in (0,1]).")
    p.add_argument("--min-samples", type=float, default=0.05,
                   help="Min samples per bin (absolute int or fraction in (0,1]).")
    p.add_argument("--min-positives", type=float, default=0.05,
                   help="Min positives per bin for binary targets (abs int or fraction in (0,1]).")
    p.add_argument("--pvalue", type=float, default=0.4, help="Initial p-value threshold.")
    p.add_argument("--no-maximize", action="store_true",
                   help="If set, run the alternate regime (keep >= min_bins).")
    p.add_argument("--plot", action="store_true", help="Render plots.")
    p.add_argument("--save-prefix", type=str, default=None,
                   help="If set, save bins/summary CSVs using this prefix.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1) Load data
    df = pd.read_csv(args.csv)

    # 2) Convert response to binary {0,1} as requested (original was {1,2})
    if args.y in df.columns:
        df[args.y] = df[args.y] - 1
    else:
        raise KeyError(f"Response column {args.y!r} not found in CSV.")

    # 3) Configure constraints
    cons = BinningConstraints(
        max_bins=args.max_bins,
        min_bins=args.min_bins,
        max_samples=args.max_samples,
        min_samples=args.min_samples,
        min_positives=args.min_positives,
        initial_pvalue=args.pvalue,
        maximize_bins=not args.no_maximize,
    )

    # 4) Run the orchestrator (MOB is metric='mean' + binary y)
    binner = MonotonicBinner(
        df=df,
        x=args.x,
        y=args.y,
        metric="mean",       # MOB special case (binary targets)
        sign="auto",         # infer monotone direction from data
        strict=True,
        constraints=cons,
        exclude_values=args.exclude,   # e.g., pass --exclude 999
    ).fit()

    # 5) Show results
    print("\n=== Clean bins (left/right are numeric bins only) ===")
    bins = binner.bins_()
    print(bins)

    print("\n=== Final summary (includes WoE/IV for binary targets; "
          "extra rows for Missing/Excluded if present) ===")
    summary = binner.summary_()
    print(summary)

    # 6) (Optional) Save artifacts
    if args.save_prefix:
        bins.to_csv(f"{args.save_prefix}_bins.csv", index=False)
        summary.to_csv(f"{args.save_prefix}_summary.csv", index=False)
        print(f"\nSaved:\n  - {args.save_prefix}_bins.csv\n  - {args.save_prefix}_summary.csv")

    # 7) (Optional) Plots
    if args.plot:
        # Bars = WoE, Line = Bad rate (for binary y)
        MOBPlot.plot_bins_summary(summary, title=f"Bins Summary · {args.x}")

        # CSD/GCM visualization requires a fitted PAVA object.
        # We re-run PAVA on the clean subset for this plot (read-only).
        df_clean = df[[args.x, args.y]].dropna(subset=[args.x, args.y])
        if args.exclude:
            df_clean = df_clean[~df_clean[args.x].isin(args.exclude)]
        pava = PAVA(df_clean, x=args.x, y=args.y, metric="mean", sign="auto").fit()
        plot_csd_gcm(pava, metric="mean", title=f"CSD/GCM · {args.x} vs {args.y}")

    # 8) Example: transform raw values to interval labels (for scoring)
    print("\n=== Transform example (first 10 values) ===")
    labels = binner.transform(df[args.x].head(10), assign="interval")
    print(pd.DataFrame({args.x: df[args.x].head(10), "bin": labels}))


if __name__ == "__main__":
    main()
