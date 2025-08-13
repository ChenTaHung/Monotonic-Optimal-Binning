<h1><p align="center"><strong>Monotonic-Optimal-Binning</strong></p></h1>

[![Run Tests](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/actions/workflows/RunTests.yml/badge.svg?branch=main)](https://github.com/ChenTaHung/Monotonic-Optimal-Binning/actions/workflows/RunTests.yml)
[![PyPI version](https://img.shields.io/pypi/v/MOBPY.svg)](https://pypi.org/project/MOBPY/)

Monotonic Optimal Binning (MOB) for Python.  
**MOBPY** builds monotone bins for a numeric feature `x` with respect to the mean of a target `y` (binary or numeric). It:

- runs a fast **PAVA** (Pool-Adjacent-Violators Algorithm) to create monotone “atomic” blocks,
- **greedily merges adjacent blocks** using a simple two-sample test and constraint penalties,
- outputs clean, half-open bins **covering the full real line** with edges `(-∞, …) … […, +∞)`,
- optionally adds **MOB-style summary** (WoE/IV, bad/good rates) when `y` is binary,
- provides **plots** for the PAVA fit and the final bin summary.

---

## Why MOBPY?

- **Deterministic & fast**: stack-based PAVA on grouped `x`, then O(k) adjacent merges.
- **Robust constraints**: min/max samples (fractional or absolute), min positives (binary), min/max #bins, maximize vs. satisfy min-bins modes.
- **Safe edges by construction**: first bin starts at **−∞**, last bin ends at **+∞**.  
  This guarantees every future value of `x` can be assigned to some bin.
- **Well-tested**: unit tests + Hypothesis property-based tests (stress) validate key invariants.

---

## Installation

```bash
pip install MOBPY
# or from your repo (editable):
pip install -e .
````

### Requirements

* Python 3.9–3.12
* numpy, pandas, matplotlib (for plotting), pytest & hypothesis (for tests)

If you keep a pinned environment, include `requirements.txt` in your project and `pip install -r requirements.txt`.

---

## Project structure

```
.
├── src/
│   └── MOBPY/
│       ├── __init__.py
│       ├── binning/
│       │   ├── __init__.py
│       │   └── mob.py             # Orchestrator (partition → PAVA → merge → bins/summary)
│       ├── core/
│       │   ├── __init__.py
│       │   ├── constraints.py     # BinningConstraints (fractional → absolute resolution)
│       │   ├── merge.py           # Block + adjacent merges + min-samples sweep
│       │   ├── pava.py            # Stack-based PAVA on grouped x
│       │   └── utils.py           # helpers (partition_df, woe/iv, type checks)
│       └── plot/
│           ├── __init__.py
│           ├── csd_gcm.py         # CSD/GCM plot (group means + PAVA step)
│           └── mob_plot.py        # WoE bars + bad-rate line (binary y)
├── tests/
│   ├── conftest.py
│   ├── test_binner.py
│   ├── test_constraints.py
│   ├── test_merge.py
│   ├── test_pava.py
│   ├── test_plots.py
│   └── test_property_based.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Quickstart

```python
import pandas as pd
import numpy as np
from MOBPY.binning.mob import MonotonicBinner
from MOBPY.core.constraints import BinningConstraints

# Toy data: binary target
rng = np.random.default_rng(7)
n = 500
x = np.linspace(-2, 3, n) + rng.normal(0, 0.15, n)
p = 1 / (1 + np.exp(-1.4 * x))
y = rng.binomial(1, p)

df = pd.DataFrame({"x": x, "y": y})

cons = BinningConstraints(
    max_bins=6,      # allow up to 6 bins
    min_bins=2,      # keep at least 2 in non-maximize mode (if feasible)
    min_samples=0.05 # per-bin min samples (fraction of clean rows)
)

binner = MonotonicBinner(
    df=df, x="x", y="y",
    metric="mean", sign="auto", strict=True,
    constraints=cons, exclude_values=None
).fit()

bins = binner.bins_()       # clean numeric bins (no Missing/Excluded rows)
summary = binner.summary_() # WoE/IV columns included when y is binary

print(bins.head())
print(summary.head())
```

### Transforming new data to bins

```python
# Map raw x to interval labels like "(-inf, 0.42)" etc.
labels = binner.transform(df["x"], assign="interval")

# Or retrieve the assigned left/right edges
left_edges  = binner.transform(df["x"], assign="left")
right_edges = binner.transform(df["x"], assign="right")
```

> **Edge convention**
> Bins are half-open intervals `[left, right)`. The **first** bin is `(-∞, right)`,
> the **last** bin is `[left, +∞)`. This ensures complete coverage for any future `x`.

---

## Plotting

Two convenient plotting utilities live under `MOBPY.plot`:

* **CSD/GCM plot**: visualize PAVA group means and the fitted step function.
* **MOB summary plot**: WoE bars + bad-rate line (binary `y` only).

```python
from MOBPY.plot.csd_gcm import plot_csd_gcm
from MOBPY.plot.mob_plot import MOBPlot

# CSD/GCM from the fitted binner
plot_csd_gcm(
    groups_df=binner._pava.groups_,          # from PAVA
    blocks=binner._pava.export_blocks(True), # list of dicts
    x_name="x", y_name="y",
    savepath="csd_gcm.png"
)

# WoE + bad-rate plot (binary y)
MOBPlot.plot_bins_summary(binner.summary_(), savepath="mob_summary.png")
```

### GIF demo (optional)

> Place the GIF at `docs/plots_demo.gif` (or any path you like) and it will render here:

<p align="center">
  <img src="docs/plots_demo.gif" alt="MOBPY demo plots" width="800"/>
</p>

**Make a GIF quickly**

1. Save a few PNGs from your notebook or script (e.g., different constraints):

```python
# Example: save multiple PNGs after plotting
for i, p in enumerate([0.4, 0.2, 0.1], start=1):
    cons.initial_pvalue = p
    MonotonicBinner(df=df, x="x", y="y", constraints=cons).fit()
    plot_csd_gcm(groups_df=binner._pava.groups_,
                 blocks=binner._pava.export_blocks(True),
                 x_name="x", y_name="y",
                 savepath=f"docs/plots_demo_{i:02d}.png")
```

2. Use ImageMagick to create a GIF:

```bash
# macOS / Linux
convert -delay 120 -loop 0 docs/plots_demo_*.png docs/plots_demo.gif
```

---

## API Overview

### `MonotonicBinner`

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

* **`sign`**: `+` for non-decreasing means, `-` for non-increasing, `auto` infers from grouped correlation.
* **`strict`**: if `True`, equal-mean plateaus are merged in PAVA to enforce strict monotonicity.
* **`constraints`**: see below; automatically **resolved** inside `fit()` on the **clean** subset.
* **`exclude_values`**: special `x` values excluded from “clean” bins and reported as separate rows in the summary.

**Key methods**

* `fit()`: runs the full pipeline, caches bins and summary.
* `bins_() -> pd.DataFrame`: clean numeric bins (no Missing/Excluded rows).
* `summary_() -> pd.DataFrame`: full summary. If `y` is binary, includes `nsamples`, `bads`, `goods`, `bad_rate`, `woe`, `iv_grp`.
* `transform(x_values, assign="interval"|"left"|"right")`: map raw `x` to the fitted bins.

### `BinningConstraints`

```python
BinningConstraints(
    max_bins: int = 6,
    min_bins: int = 4,
    max_samples: float | None = None,  # fraction in (0,1] or absolute
    min_samples: float | None = None,  # fraction in (0,1] or absolute
    min_positives: float | None = None,# binary-only; fraction or absolute
    initial_pvalue: float = 0.4,       # merge threshold (annealed by search)
    maximize_bins: bool = True,
)
```

Resolved attributes after `resolve(...)` (called by `MonotonicBinner.fit()`):

* `abs_max_samples: int | None`
* `abs_min_samples: int`
* `abs_min_positives: int`

The library ensures `min_samples <= max_samples` (when both provided) and `min_bins <= max_bins` in `maximize_bins=True` mode.

---

## Datasets & Notebooks

Explore MOBPY with the example datasets:

* `german_data_credit_cat.csv` (binary) — target is `default - 1` (0/1),
* `insurance3r2.csv` (binary) — target is `insuranceclaim`.

A demonstration notebook is provided in the repo (or build your own via the Quickstart and Plotting examples).

---

## Testing

Run the full test suite:

```bash
pip install -r requirements.txt
pytest -q
```

We include both **unit tests** and **Hypothesis property-based tests**.
Property tests stress the pipeline with random datasets and constraint combinations, asserting invariants such as:

* monotone means under the resolved sign,
* first bin starts at **−∞**, last bin ends at **+∞** (full coverage),
* bins respect min/max samples or stop merging at min-bins,
* post-merge bin count doesn’t exceed the PAVA baseline when not maximizing bins.

---

## Versioning & Changelog

* Current: **2.0.0**
* We follow semantic-ish versioning. Breaking API changes bump the major version.

---

## Contributing

Issues and PRs are welcome!

1. Create a virtual environment
2. `pip install -r requirements.txt`
3. Add tests for any code changes
4. `pytest -q` before pushing

---

## License

MIT (see `LICENSE`)

---

## Acknowledgements

This package implements a clean, testable variant of PAVA + adjacent merges tailored to monotone binning use-cases common in risk scoring and credit modeling.

---

## Reference

* [Mironchyk, Pavel, and Viktor Tchistiakov. *Monotone optimal binning algorithm for credit risk modeling.* (2017)](https://www.researchgate.net/profile/Viktor-Tchistiakov/publication/322520135_Monotone_optimal_binning_algorithm_for_credit_risk_modeling/links/5a5dd1a8458515c03edf9a97/Monotone-optimal-binning-algorithm-for-credit-risk-modeling.pdf)
* [Smalbil, P. J. *The choices of weights in the iterative convex minorant algorithm.* (2015)](https://repository.tudelft.nl/islandora/object/uuid:5a111157-1a92-4176-9c8e-0b848feb7c30)
* Testing Dataset 1: [German Credit Risk](https://www.kaggle.com/datasets/uciml/german-credit) from Kaggle
* Testing Dataset 2: [US Health Insurance Dataset](https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset) from Kaggle
* GitHub Project: [Monotone Optimal Binning (SAS 9.4 version)](https://github.com/cdfq384903/MonotonicOptimalBinning)

---

## Authors

1. Ta-Hung (Denny) Chen

   * LinkedIn: [https://www.linkedin.com/in/dennychen-tahung/](https://www.linkedin.com/in/dennychen-tahung/)
   * E-mail: [denny20700@gmail.com](mailto:denny20700@gmail.com)
2. Yu-Cheng (Darren) Tsai

   * LinkedIn: [https://www.linkedin.com/in/darren-yucheng-tsai/](https://www.linkedin.com/in/darren-yucheng-tsai/)
   * E-mail:
3. Peter Chen

   * LinkedIn: [https://www.linkedin.com/in/peterchentsungwei/](https://www.linkedin.com/in/peterchentsungwei/)
   * E-mail: [peterwei20700@gmail.com](mailto:peterwei20700@gmail.com)


