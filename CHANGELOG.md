# Changelog

All notable changes to MOBPY will be documented in this file.

## [2.0.0] - 2022025-08-28

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