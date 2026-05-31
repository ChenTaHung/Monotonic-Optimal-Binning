# Changelog

All notable changes to MOBPY will be documented in this file.

## [2.3.0] - 2026-02-27

### Added

- **Categorical binning path** (`x_type='categorical'`): chi-square-based block merging with multiple comparison correction (Holm by default), O(k²) pair-result caching, three-phase merging (statistical → min_samples → class-count), and full `BinningConstraints` enforcement
- **`MonotonicBinner` parameters for categorical path**:
  - `x_type` — `'numeric'` (default) or `'categorical'`
  - `categorical_alpha` — significance level for chi-square merging (default `0.05`)
  - `categorical_correction` — `'holm'` (default), `'bonferroni'`, or `'fdr_bh'`
  - `unseen_categories` — `'error'` (default) raises `ValueError` on unseen values; `'unknown'` returns `"Unknown"` / NaN WoE
  - `max_label_cats` — truncate long bin labels: `{A, B, C, ...+N}`
- **`bin_assignment()` method** on `MonotonicBinner` — returns a Series mapping each original category to its 0-based bin index
- **`plot_categorical_merge()`** visualization — one bar per original category coloured by final bin assignment, grouped with gaps, per-bin dashed pooled-rate hlines, overall mean dotted line
- **`tick_labels` parameter** on `plot_woe_bars` and `plot_event_rate` — `None` (verbatim), `list[str]` (explicit), or `'auto'` (compact `"Bin N\n(XX.X%)"` labels for categorical set labels)
- **`ensure_categorical_series()`** validation utility — rejects numeric dtype for the categorical path
- **`woe_iv()` `return_components` parameter** — returns `{"woe": arr, "iv": arr}` dict when `True`
- **`CategoryBlock`** and **`merge_categorical()`** in new `MOBPY.core.categorical_merge` module, also re-exported from `MOBPY.core`
- **E-Commerce Fraud categorical binning example** notebook (`examples/E-Commerce Fraud - Categorical Binning.ipynb`)

### Changed

- **Dropped Python 3.9–3.12 support** — minimum Python version is now **3.13**
- **NumPy minimum version bumped to 2.0.0** — NumPy 1.x has no Python 3.13+ wheels
- **CI matrix** now runs on Python 3.13 and 3.14 in parallel
- `get_diagnostics()` returns `n_initial_categories` and `n_final_bins` for the categorical path (in addition to existing numeric-path keys)
- `bins_()` returns a `categories` column (sorted list of original values) instead of `left`/`right` edges for the categorical path

### Fixed

- **`MatplotlibDeprecationWarning`** — replaced `plt.cm.get_cmap()` (deprecated in 3.7, removed in 3.11) with `matplotlib.colormaps[name].resampled(N)` in `plot_categorical_merge`

---

## [2.2.0] - 2025-02-18

### Added
- **`min_negatives` parameter** in `BinningConstraints` for controlling minimum negative samples (y=0) per bin, ensuring stable WoE calculations
- **`positives` and `negatives` properties** on `Block` class for easier access to class counts in binary classification
- **Feasibility warnings** during `resolve()` when `min_positives` or `min_negatives` constraints cannot be mathematically satisfied with the given `min_bins`
- **`_enforce_min_class_counts()` function** for unified enforcement of both `min_positives` and `min_negatives` constraints in a single pass
- New tests for class count constraint enforcement and feasibility warnings

### Fixed
- **`min_positives` constraint is now actively enforced** - Previously it was only a soft penalty (1.4x score multiplier) in the merge scorer; now bins violating `min_positives` are actively merged until the constraint is satisfied or `min_bins` floor is reached
- `_validate_merge_result()` now properly checks and warns about `min_positives` and `min_negatives` violations when constraints cannot be fully satisfied

### Changed
- `merge_adjacent()` now includes Phase 3 for class count enforcement after min_samples enforcement
- `_validate_merge_result()` now accepts `is_binary_y` parameter to enable class count validation
- `Block.as_dict()` now includes `positives` and `negatives` keys in the exported dictionary
- `MergeScorer._apply_penalties()` now applies bonuses for merging bins with insufficient negatives (matching existing positives behavior)
- `BinningConstraints.__repr__()` now includes `min_negatives` in the string representation
- `BinningConstraints.copy()` now properly copies `min_negatives` parameter

---

## [2.1.0] - 2025-11-06

### Fixed
- Fixed exluded bins don't calculate woe and iv. 
- Fixed `TypeError` when comparing constraints with None values in `_check_constraints_satisfied()`
- Fixed `TypeError: 'NoneType' object is not iterable` in `transform()` method
- Now properly uses `abs_min_samples` and `abs_min_positives` instead of nullable constraint values
- Added None checks for `exclude_values` in transform operations

### Changed
- Now the excluded bin will show the WoE and iv calcualted.
- Improved constraint validation to use resolved absolute values
- Enhanced `__init__` parameter validation and error messages

## [2.0.0] - 2025-08-28

### 🎉 Major Release

This is a complete rewrite of the monotonic optimal binning library with significant improvements in performance, reliability, and usability.

### Added
- **Stack-based PAVA algorithm** with O(n) complexity for monotone fitting
- **Comprehensive constraint system** supporting both fractional and absolute constraints
- **Rich visualization suite** including:
  - PAVA process visualization (CSD/GCM plots)
  - Weight of Evidence (WoE) bar charts
  - Event rate analysis plots
  - Comprehensive multi-panel statistics dashboard
  - Binning stability comparison plots
- **Automatic WoE/IV calculation** for binary classification targets
- **Safe edge handling** with -∞ and +∞ boundaries for complete coverage
- **Special value handling** for excluded codes (e.g., -999, -1)
- **Global configuration system** via `MOBPYConfig`
- **Comprehensive logging** with adjustable verbosity levels
- **Progress tracking** for long-running operations
- **Type hints** throughout the codebase
- **Extensive test suite** with property-based testing using Hypothesis

### Changed
- **Complete architecture redesign** with modular structure:
  - `core/` - Algorithm implementations (PAVA, merging, utilities)
  - `binning/` - Main orchestration layer
  - `plot/` - Visualization tools
- **Improved API** with cleaner, more intuitive interface
- **Better error handling** with custom exception hierarchy
- **Enhanced documentation** with detailed API references and examples

---

### Key Migration Points:
1. Package name changed from `mob` to `MOBPY`
2. Main class renamed from `MOB` to `MonotonicBinner`
3. Constraints now use dedicated `BinningConstraints` class
4. Results accessed via `bins_()` and `summary_()` methods
5. Plotting functions moved to `MOBPY.plot` module