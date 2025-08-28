# Changelog

All notable changes to MOBPY will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-01-XX

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
- **Optimized performance** for datasets from 10² to 10⁶ samples

### Fixed
- Numerical stability issues in edge cases
- Memory leaks in large dataset processing
- Monotonicity violations in certain constraint combinations
- Handling of missing values and infinite values

### Performance
- **10x faster** PAVA implementation compared to v1.x
- **50% reduction** in memory usage for large datasets
- **Deterministic** results (no randomness in algorithm)

## [1.0.0] - 2023-XX-XX

### Initial Release
- Basic monotonic binning functionality
- Simple PAVA implementation
- Basic constraint support

---

## Upgrade Guide

### From 1.x to 2.0

The API has changed significantly. Here's how to migrate:

**Old (1.x):**
```python
from mob import MOB
mob = MOB(data, x='feature', y='target')
result = mob.fit()
```

**New (2.0):**
```python
from MOBPY import MonotonicBinner, BinningConstraints

constraints = BinningConstraints(max_bins=6, min_samples=0.05)
binner = MonotonicBinner(df, x='feature', y='target', constraints=constraints)
binner.fit()
result = binner.summary_()
```

### Key Migration Points:
1. Package name changed from `mob` to `MOBPY`
2. Main class renamed from `MOB` to `MonotonicBinner`
3. Constraints now use dedicated `BinningConstraints` class
4. Results accessed via `bins_()` and `summary_()` methods
5. Plotting functions moved to `MOBPY.plot` module