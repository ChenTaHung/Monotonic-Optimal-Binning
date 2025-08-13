#!/usr/bin/env python
from __future__ import annotations

import argparse
import pathlib
from typing import List, Optional

import pandas as pd

from MOBPY.core.constraints import BinningConstraints
from MOBPY.binning.mob import MonotonicBinner
from MOBPY.plot.mob_plot import MOBPlot
from MOBPY.plot.csd_gcm import plot_csd_gcm_from_binner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run monotone optimal binning (MOB) demo.")
    p.add_argument("--csv", type=str, default="data/german_data_credit_cat.csv",
                   help="Path to german_data_credit_cat.csv (default: data/german_data_credit_cat.csv)")
    p.add_argument("--x", type=str, default="Durationinmonth", help="Feature/variable to bin.")
    p.add_argument("--y", type=str, default="default", help="Response column (binary for MOB).")
    p.add_argument("--exclude", action="append", default=None,
                   help="Values in x to treat as special bins. Use multiple flags for multiple values.")
    p.add_argument("--max-bins", type=int, default=6, help="Maximum number of bins.")
    p.add_argument("--min-bins", type=int, default=4, help="Minimum number of bins (only used if maximize is false).")
    p.add_argument("--max-samples", type=float, default=0.4, help="Per-bin cap. Fraction (0,1] or int; None to disable.")
    p.add_argument("--min-samples", type=float, default=0.05, help="Per-bin floor. Fraction [0,1) or int; 0 to disable.")
    p.add_argument("--min-positives", type=float, default=0.05, help="Binary only: per-bin positives min. Fraction or int; 0 to disable.")
    p.add_argument("--p0", type=float, default=0.4, help="Initial p-value threshold.")
    p.add_argument("--maximize", action="store_true", default=True, help="If set, enforce <= max-bins (default).")
    p.add_argument("--no-maximize", dest="maximize", action="store_false", help="Use min-bins regime instead.")
    p.add_argument("--plot", action="store_true", help="Render plots (saved to ./_out).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load CSV and adapt 'default' to {0,1} as per your legacy script
    df = pd.read_csv(args.csv)
    if args.y in df.columns and df[args.y].dropna().nunique() == 2 and set(df[args.y].unique()) == {1, 2}:
        df[args.y] = df[args.y] - 1

    # Constraints
    cons = BinningConstraints(
        max_bins=args.max_bins,
        min_bins=args.min_bins,
        max_samples=args.max_samples,
        min_samples=args.min_samples,
        min_positives=args.min_positives,
        initial_pvalue=args.p0,
        maximize_bins=args.maximize,
    )

    # Fit binner
    binner = MonotonicBinner(
        df=df, x=args.x, y=args.y,
        metric="mean", sign="auto",
        constraints=cons,
        exclude_values=args.exclude,
    ).fit()

    print("\n=== CLEAN BINS ===")
    print(binner.bins_())

    print("\n=== FULL SUMMARY ===")
    print(binner.summary_())

    # Optional plots
    if args.plot:
        out_dir = pathlib.Path("_out")
        out_dir.mkdir(exist_ok=True)
        MOBPlot.plot_bins_summary(binner.summary_(), savepath=str(out_dir / "bins_summary.png"))
        plot_csd_gcm_from_binner(binner, savepath=str(out_dir / "csd_gcm.png"))
        print(f"\nSaved plots in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
