<h1><p align = 'center'><strong> Monotonic-Optimal-Binning </strong> </p></h1>

[![Run Tests](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/actions/workflows/RunTests.yml/badge.svg?branch=main)](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/actions/workflows/RunTests.yml)
[![Formatting Tests](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/actions/workflows/FormattingTests.yml/badge.svg)](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/actions/workflows/FormattingTests.yml)

<h3><p align = 'center'><strong> Python implementation (MOBPY) </strong> </p></h3>

`Monotone optimal binning` for numeric features, built on an efficient PAVA (isotonic regression) core with constraint-aware merging — plus optional MOB-style summaries (WoE/IV) and plots.

* **Single unified engine:** `PAVA` does isotonic pooling; **MOB** is just the special case where the metric is the **mean of a (binary) target**.
* **Constraints-aware merging:** after PAVA, adjacent blocks are merged using two-sample tests with **p-value annealing** and **min/max samples** (and **min positives** for binary).
* **Robust bin edges:** left-closed, right-open intervals `"[left, right)"`, first bin starts at `(-inf, …)`, last ends at `(..., inf)`.
* **Missing & special values:** handled as **separate display bins** (“Missing” and literal values) and excluded from the “clean” fit.
* **MOB summary:** for binary targets, computes **WoE** and **IV** per bin with smoothing.

---

## Install

### Minimal runtime

```bash
# pip (src/ layout; install your package)
pip install -e .

# or install just runtime deps
pip install -r requirements.txt
```

### With plotting and dev/test tools

```bash
# extras
pip install -e ".[plot,dev]"
```

### Conda (recommended for examples & tests)

```bash
conda env create -f conda-env.yml
conda activate mobpy-2.0
```

---

## Quickstart

```python
import pandas as pd
from MOBPY.binning.mob import MonotonicBinner
from MOBPY.core.constraints import BinningConstraints

# 1) Load your data (example: German credit)
df = pd.read_csv("./data/german_data_credit_cat.csv")

# The dataset has target in {1, 2}. Convert to {0, 1} if needed:
df["default"] = df["default"] - 1

# 2) Define constraints (classic MOB defaults shown)
cons = BinningConstraints(
    max_bins=6,         # cap number of bins
    min_bins=4,         # or ensure at least this many when maximize_bins=False
    max_samples=0.4,    # per-bin cap (fraction or absolute)
    min_samples=0.05,   # per-bin floor (fraction or absolute)
    min_positives=0.05, # for binary targets only (fraction or absolute)
    initial_pvalue=0.4, # start high, anneal down if needed
    maximize_bins=True, # “fit up to max bins” (classic) vs “at least min bins”
)

# 3) Fit binner (MOB is “mean of target”; sign can be '+', '-', or 'auto')
binner = MonotonicBinner(
    df=df, x="Durationinmonth", y="default",
    metric="mean", sign="auto",
    constraints=cons,
    exclude_values=[999]  # optional “special” x values that get their own rows
).fit()

# 4) Inspect bins (clean portion only)
bins = binner.bins_()
print(bins)
# columns: left, right, n, sum, mean, std, min, max

# 5) Full MOB-style summary (adds Missing & Excluded rows; WoE/IV for binary)
summary = binner.summary_()
print(summary)

# 6) Apply transform to new values
out = binner.transform(df["Durationinmonth"], assign="interval")  # or "left"/"right"
print(out.head())
```

---

## Plotting

CSD/GCM (group means & PAVA step) and MOB-style plot are optional (install with `.[plot]`).

```python
from MOBPY.plot.csd_gcm import plot_csd_gcm_from_binner
from MOBPY.plot.mob_plot import MOBPlot

# CSD/GCM (one point per unique x; step function = PAVA fit)
plot_csd_gcm_from_binner(binner, savepath="csd_gcm.png")

# WoE bars + bad-rate line (works when y is binary)
MOBPlot.plot_bins_summary(summary, savepath="mob_bins.png")
```

---

## How it works (high level)

1. **Partition input** by `x`:

   * **clean**: non-missing and not in `exclude_values`
   * **missing**: `x` is NaN
   * **excluded**: `x ∈ exclude_values`
2. **Detect binary target** on the **clean** partition (so WoE/IV only when `y ∈ {0,1}`).
3. **PAVA on clean** (metric = mean):

   * group by unique `x` (sorted) → atomic blocks with stats (n, sum, sum², min, max)
   * **pool adjacent violators** to enforce monotonicity (stack-based; equivalent to DLL merging)
   * if `strict=True`, merge **plateaus** (equal means) to enforce strictly monotone steps.
4. **Constraints-aware merging** on the PAVA blocks:

   * Greedily merge the **best adjacent pair** (two-sample test p-value, with penalties for constraint pressure).
   * **Anneal the p-value threshold** (“start high; if bins still violate constraints like `max_bins`, reduce threshold and merge again”).
   * Enforce **min/max samples** and **min positives** (binary).
5. **Materialize bins** as left-closed, right-open intervals; set first left to `-inf`, last right to `+inf`.
6. **Summary rows**: append Missing and each special value as their own display bins.
7. **WoE/IV** (binary): smoothed to avoid log(0).

---

## API

### `MOBPY.binning.mob.MonotonicBinner`

```python
MonotonicBinner(
    df: pd.DataFrame,
    x: str,
    y: str,
    metric: Literal["mean"] = "mean",
    sign: Literal["+", "-", "auto"] = "auto",
    strict: bool = True,
    constraints: BinningConstraints | None = None,
    exclude_values: Iterable | None = None,
    sort_kind: str | None = None,
)
```

* `fit()` → returns self
* `bins_()` → clean bins as DataFrame (`left`, `right`, `n`, `sum`, `mean`, `std`, `min`, `max`)
* `summary_()` → full summary; for binary: `nsamples`, `bads`, `goods`, `bad_rate`, `woe`, `iv_grp`
* `transform(series, assign="interval"|"left"|"right")` → map raw x to fitted bin labels or edges

### `MOBPY.core.pava.PAVA`

Lower-level PAVA (used internally by the binner). Exposes:

* `fit()` → creates `blocks_` and `groups_`
* `export_blocks(as_dict=True)` → list of dicts with stable fields (left, right, n, sum, sum2, ymin, ymax)

### `MOBPY.core.constraints.BinningConstraints`

Holds user knobs; `resolve(total_n, total_pos)` converts fractional limits to absolutes on the **clean** partition and validates cross-relations (e.g., `min_samples` cannot exceed `max_samples`).

---

## Testing

Property-based tests (via **Hypothesis**) stress every edge case we could think of: constant signals, plateaus, extremes of constraints, missing/special-value combos, and tiny/large sample blocks.

```bash
pytest -vv -q
```

---

## Reproducible environments

* **Minimal**: `requirements.txt`
* **Conda**: `conda-env.yml` (includes plotting and test deps, installs package `-e .`)
* **Exact locks (optional)**:

  ```bash
  python -m pip freeze --exclude-editable > requirements-lock.txt
  conda env export --no-builds > conda-env-lock.yml
  ```

---

## FAQ / Tips

* **Binary target detection**: we check the **clean** `y` values are subset of `{0,1}`. If your labels are `{1,2}`, convert first (e.g., `y = y - 1`).
* **Interpreting edges**: bins are `[left, right)`. The first interval is `"(-inf, R)"`, last is `"[L, inf)"`.
* **Special values**: values you pass in `exclude_values` are **not** part of the fitted PAVA. They appear as their own rows in the summary.
* **Plateaus**: with `strict=True`, equal-mean adjacent blocks are merged so the final step function is strictly monotone.

---

## Versioning & License

* Version: **2.0.0**
* License: MIT (see `LICENSE`)

---

## Citation

If this project helped your work, please consider citing:

> MOBPY 2.0.0 — Monotone Optimal Binning, [https://github.com/ChenTaHung/Monotonic-Optimal-Binning](https://github.com/ChenTaHung/Monotonic-Optimal-Binning)



## Reference 

- [Mironchyk, Pavel, and Viktor Tchistiakov. "Monotone optimal binning algorithm for credit risk modeling." Utr. Work. Pap (2017).](https://www.researchgate.net/profile/Viktor-Tchistiakov/publication/322520135_Monotone_optimal_binning_algorithm_for_credit_risk_modeling/links/5a5dd1a8458515c03edf9a97/Monotone-optimal-binning-algorithm-for-credit-risk-modeling.pdf)
  
- [Smalbil, P. J. "The choices of weights in the iterative convex minorant algorithm." (2015).](https://repository.tudelft.nl/islandora/object/uuid:5a111157-1a92-4176-9c8e-0b848feb7c30)

- Testing Dataset 1 : [German Credit Risk](https://www.kaggle.com/datasets/uciml/german-credit) from [Kaggle](https://www.kaggle.com/)

- Testing Dataset 2 : [US Health Insurance Dataset](https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset) from [Kaggle](https://www.kaggle.com/)

- GitHub Project : [Monotone Optimal Binning (SAS 9.4 version)](https://github.com/cdfq384903/MonotonicOptimalBinning)

## Authors 


1. Ta-Hung (Denny) Chen
    - LinkedIn Profile : https://www.linkedin.com/in/dennychen-tahung/
    - E-Mail : denny20700@gmail.com
2. Yu-Cheng (Darren) Tsai
    - LindedIn Profile : https://www.linkedin.com/in/darren-yucheng-tsai/
    - E-Mail : 
3. Peter Chen
   - LinkedIn Profile : https://www.linkedin.com/in/peterchentsungwei/
   - E-Mail : peterwei20700@gmail.com

