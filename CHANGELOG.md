# Changelog

All notable changes to MOBPY will be documented in this file.

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