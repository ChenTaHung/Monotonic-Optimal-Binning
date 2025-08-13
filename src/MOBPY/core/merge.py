# src/MOBPY/core/merge.py
"""
Adjacent-merge engine with statistical tests and constraint handling.

This module consumes **monotone blocks** (e.g., from PAVA) and merges adjacent
pairs until the bin-count and bin-size constraints are satisfied.

Design goals
------------
- General: works for binary y (MOB case, metric="mean") and numeric y.
- Robust: uses sufficient statistics to merge without data access.
- Faithful: reproduces your legacy behavior (p-value schedule, penalties).
- Clean API: decoupled from the PAVA internals.

Key functions
-------------
- `merge_adjacent(blocks, constraints, is_binary_y)`:
    Main entry. Greedily merges the adjacent pair with the largest p-value
    above the current threshold; if none, lowers the threshold per schedule.

- `as_blocks(items)`:
    Adapter to convert dicts/objects to `Block` dataclasses.

Statistical tests
-----------------
- Binary y:
    Two-proportion z-test (two-sided) with pooled variance.
- Numeric y:
    Welch's t-test (two-sided) with Satterthwaite df.

Penalties and preferences (match legacy logic)
----------------------------------------------
- Singleton bins (n==1): set p=3.0 to strongly encourage merging.
- Min samples: +1.0 if either side < min_samples; +1.0 if merged still < min_samples.
- Min positives (binary only): +1.0 if either side < min_positives.
- WoE safety (binary only): +2.0 if either side has rate in {0, 1}.
- Oversized merged bin (> max_samples if defined):
    Downweight p by *0.01* if both sides are "healthy" (n>1, rate∉{0,1}),
    else by *0.1*. This discourages huge bins unless tiny/degenerate bins involved.

P-value threshold schedule
--------------------------
Start at `constraints.initial_pvalue`. If no pair exceeds it while the bin
count still violates limits, reduce the threshold:
- subtract 0.05 down to 0.01, then
- multiply by 0.1 on each step (floor at 1e-8).

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import norm, t as student_t  # requires SciPy

from .constraints import BinningConstraints


# ------------------------------- Data model ---------------------------------- #

@dataclass
class Block:
    """A contiguous bin with sufficient statistics.

    Attributes:
        left: Left edge (inclusive).
        right: Right edge (exclusive). The last block should have +inf.
        n: Number of rows in the block.
        y_sum: Sum of y.
        y_sum2: Sum of y^2 (sample variance reconstruction).
        y_min: Minimum y in the block.
        y_max: Maximum y in the block.
    """
    left: float
    right: float
    n: int
    y_sum: float
    y_sum2: float
    y_min: float
    y_max: float

    # ---- derived stats ----------------------------------------------------- #

    def mean(self) -> float:
        """Mean of y."""
        return self.y_sum / self.n if self.n else float("nan")

    def var(self) -> float:
        """Unbiased sample variance."""
        if self.n <= 1:
            return 0.0
        num = self.y_sum2 - (self.y_sum * self.y_sum) / self.n
        return max(num / (self.n - 1.0), 0.0)

    def std(self) -> float:
        """Sample standard deviation."""
        return float(np.sqrt(self.var()))

    # ---- pool with neighbor ------------------------------------------------ #

    def merged_with(self, other: "Block") -> "Block":
        """Return a new Block that merges this block followed by `other`."""
        return Block(
            left=self.left,
            right=other.right,
            n=self.n + other.n,
            y_sum=self.y_sum + other.y_sum,
            y_sum2=self.y_sum2 + other.y_sum2,
            y_min=min(self.y_min, other.y_min),
            y_max=max(self.y_max, other.y_max),
        )


def as_blocks(items: Iterable[Union[Block, dict, object]]) -> List[Block]:
    """Convert an iterable of dicts/objects into `Block` instances.

    Accepts:
        - Block instances (returned as-is),
        - dicts with keys: left,right,n,y_sum,y_sum2,y_min,y_max,
        - any object with attributes of the same names.

    Returns:
        List[Block]

    Raises:
        ValueError: If an item cannot be adapted.
    """
    out: List[Block] = []
    for it in items:
        if isinstance(it, Block):
            out.append(it)
            continue
        if isinstance(it, dict):
            try:
                out.append(
                    Block(
                        left=float(it["left"]),
                        right=float(it["right"]),
                        n=int(it["n"]),
                        y_sum=float(it["y_sum"]),
                        y_sum2=float(it["y_sum2"]),
                        y_min=float(it["y_min"]),
                        y_max=float(it["y_max"]),
                    )
                )
                continue
            except Exception as e:  # pragma: no cover (defensive)
                raise ValueError(f"Bad block dict: {it}") from e
        # try attribute access
        try:
            out.append(
                Block(
                    left=float(getattr(it, "left")),
                    right=float(getattr(it, "right")),
                    n=int(getattr(it, "n")),
                    y_sum=float(getattr(it, "y_sum")),
                    y_sum2=float(getattr(it, "y_sum2")),
                    y_min=float(getattr(it, "y_min")),
                    y_max=float(getattr(it, "y_max")),
                )
            )
        except Exception as e:  # pragma: no cover (defensive)
            raise ValueError(f"Cannot adapt item to Block: {it!r}") from e
    return out


# --------------------------- Statistical primitives -------------------------- #

def _pvalue_two_proportions(a: Block, b: Block) -> float:
    """Two-sided p-value for difference in proportions (binary y).

    Uses pooled standard error (large-sample normal approximation).
    Falls back to p=1.0 if SE is zero.

    Returns:
        float in [0, 1]
    """
    n1, n2 = a.n, b.n
    x1, x2 = a.y_sum, b.y_sum  # number of positives
    if n1 <= 0 or n2 <= 0:
        return 1.0
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2) if (n1 + n2) > 0 else 0.0
    se = np.sqrt(max(p_pool * (1.0 - p_pool), 0.0) * (1.0 / n1 + 1.0 / n2))
    if se <= 0:
        return 1.0
    z = (p1 - p2) / se
    return float(2.0 * norm.sf(abs(z)))


def _pvalue_welch_t(a: Block, b: Block) -> float:
    """Two-sided p-value for difference in means (numeric y) via Welch's t-test.

    Robust to unequal variances and sample sizes. If SE or df <= 0, returns 1.0.
    """
    n1, n2 = a.n, b.n
    if n1 <= 0 or n2 <= 0:
        return 1.0

    m1, m2 = a.mean(), b.mean()
    s1_sq, s2_sq = a.var(), b.var()

    # Standard error for difference in means
    se_sq = (s1_sq / n1) + (s2_sq / n2)
    if se_sq <= 0:
        return 1.0

    # Welch–Satterthwaite degrees of freedom
    num = se_sq * se_sq
    den = ((s1_sq / n1) ** 2) / (n1 - 1 if n1 > 1 else np.inf) + ((s2_sq / n2) ** 2) / (n2 - 1 if n2 > 1 else np.inf)
    df = num / den if den > 0 else 0.0
    if df <= 0 or not np.isfinite(df):
        return 1.0

    t_stat = (m1 - m2) / np.sqrt(se_sq)
    return float(2.0 * student_t.sf(abs(t_stat), df))


def _pair_pvalue(a: Block, b: Block, is_binary_y: bool) -> float:
    """Dispatch to the appropriate two-sample test."""
    return _pvalue_two_proportions(a, b) if is_binary_y else _pvalue_welch_t(a, b)


# ------------------------------- Merge helpers -------------------------------- #

def _adjacent_pvals(
    blocks: List[Block],
    cons: BinningConstraints,
    is_binary_y: bool,
) -> Tuple[np.ndarray, List[Block]]:
    """Compute penalized p-values for all adjacent pairs and their merged blocks.

    Mirrors legacy behavior by adding penalties / downweighting.

    Returns:
        pvals: shape (len(blocks)-1,) penalized p-values
        merged: list of would-be merged blocks for each adjacent pair
    """
    n = len(blocks)
    if n < 2:
        return np.array([]), []

    pvals = np.zeros(n - 1, dtype=float)
    merged: List[Block] = []

    for i in range(n - 1):
        a, b = blocks[i], blocks[i + 1]
        m = a.merged_with(b)

        # Base p-value from the relevant test
        p = _pair_pvalue(a, b, is_binary_y=is_binary_y)

        # ---- penalties / preferences (match legacy) ----------------------- #

        # Singleton bins → strongly prefer merging
        if a.n == 1 or b.n == 1:
            p = 3.0

        # Min samples → encourage merging small bins
        if a.n < cons.abs_min_samples or b.n < cons.abs_min_samples:
            p += 1.0
            if m.n < cons.abs_min_samples:
                p += 1.0

        # Binary only: min positives (a.k.a. min_bads) and WoE safety
        if is_binary_y:
            a_pos = a.y_sum
            b_pos = b.y_sum
            if a_pos < cons.abs_min_positives or b_pos < cons.abs_min_positives:
                p += 1.0

            # Avoid bins that would cause ±∞ WoE (rates 0 or 1)
            ar, br = a.mean(), b.mean()
            if ar in (0.0, 1.0) or br in (0.0, 1.0):
                p += 2.0

        # Oversized merged bin → downweight the p-value
        if cons.abs_max_samples is not None and m.n > cons.abs_max_samples:
            # If both sides are "healthy", downweight more strongly
            healthy = (a.n > 1 and b.n > 1)
            if is_binary_y:
                healthy = healthy and (a.mean() not in (0.0, 1.0)) and (b.mean() not in (0.0, 1.0))
            p *= 0.01 if healthy else 0.1

        pvals[i] = p
        merged.append(m)

    return pvals, merged


def _merge_once(blocks: List[Block], merged: List[Block], idx: int) -> List[Block]:
    """Merge blocks[idx] and blocks[idx+1] into `merged[idx]` and fix edges."""
    new_blocks = blocks[:idx] + [merged[idx]] + blocks[idx + 2:]

    # Recompute right edges to maintain left-closed/right-open invariant.
    for j in range(len(new_blocks) - 1):
        new_blocks[j].right = new_blocks[j + 1].left
    new_blocks[-1].right = float("inf")
    return new_blocks


def _enforce_min_samples(
    blocks: List[Block],
    cons: BinningConstraints,
    is_binary_y: bool,
) -> List[Block]:
    """Sweep undersized bins and merge them into neighbors (while we can).

    Strategy:
        Recompute p-values; pick the pair with highest p that involves an
        undersized bin (n < min_samples) and merge; repeat until all bins are
        >= min_samples or we hit `min_bins`.
    """
    while len(blocks) > cons.min_bins:
        small_idx = [i for i, b in enumerate(blocks) if b.n < cons.abs_min_samples]
        if not small_idx:
            break
        pvals, merged = _adjacent_pvals(blocks, cons, is_binary_y)
        if pvals.size == 0:
            break

        # Candidate pairs that involve at least one undersized bin
        cand = []
        for i in range(len(blocks) - 1):
            if i in small_idx or (i + 1) in small_idx:
                cand.append((pvals[i], i))

        if not cand:
            break

        # Merge the candidate pair with the largest p-val
        cand.sort(key=lambda x: x[0], reverse=True)
        _, idx = cand[0]
        blocks = _merge_once(blocks, merged, idx)
    return blocks


# --------------------------------- Public API -------------------------------- #

def merge_adjacent(
    blocks_in: Sequence[Block],
    constraints: BinningConstraints,
    is_binary_y: bool,
) -> List[Block]:
    """Merge adjacent monotone blocks under binning constraints.

    The algorithm greedily merges the adjacent pair with the *largest p-value*
    above a threshold. If no such pair exists while violating the bin-count
    constraint, the threshold is reduced according to the schedule in
    `BinningConstraints.next_threshold`.

    Args:
        blocks_in: Initial monotone blocks (left-closed/right-open except last).
        constraints: Resolved binning constraints (call `resolve(...)` first).
        is_binary_y: Whether y is binary {0,1} (affects tests & penalties).

    Returns:
        A new list of `Block` representing the merged, constraint-compliant bins.

    Raises:
        ValueError: If inputs are invalid or constraints unresolved.
    """
    if not isinstance(constraints, BinningConstraints):
        raise ValueError("`constraints` must be a BinningConstraints instance.")
    if not constraints.resolved:
        raise ValueError("Constraints must be resolved. Call constraints.resolve(...).")

    blocks = as_blocks(blocks_in)
    if len(blocks) == 0:
        return []

    # Ensure blocks are sorted and edges make sense; set last right to +inf.
    blocks.sort(key=lambda b: b.left)
    for i in range(len(blocks) - 1):
        blocks[i].right = blocks[i + 1].left
    blocks[-1].right = float("inf")

    # Choose the operating regime based on constraints.maximize_bins
    pthr = float(constraints.initial_pvalue)

    if constraints.maximize_bins:
        # Merge down to <= max_bins, lowering p-threshold as needed.
        while len(blocks) > constraints.max_bins:
            pvals, merged = _adjacent_pvals(blocks, constraints, is_binary_y)
            if pvals.size == 0:
                break

            max_idx = int(np.argmax(pvals))
            max_p = pvals[max_idx]

            if max_p > pthr:
                blocks = _merge_once(blocks, merged, max_idx)
                continue

            # No pair exceeds current threshold; reduce and retry
            new_pthr = BinningConstraints.next_threshold(pthr)
            if new_pthr == pthr:  # floor reached; nothing else to do
                break
            pthr = new_pthr

        # After reaching the cap, sweep undersized bins if any
        blocks = _enforce_min_samples(blocks, constraints, is_binary_y)
        return blocks

    # Alternative regime: try to keep >= min_bins
    while len(blocks) > constraints.min_bins:
        pvals, merged = _adjacent_pvals(blocks, constraints, is_binary_y)
        if pvals.size == 0:
            break

        max_idx = int(np.argmax(pvals))
        max_p = pvals[max_idx]

        if max_p > pthr:
            blocks = _merge_once(blocks, merged, max_idx)
            continue

        # Greedy bulk-merge: if merging all pairs with p > pthr still keeps us >= min_bins
        over = np.where(pvals > pthr)[0]
        if over.size > 0 and (len(blocks) - over.size) >= constraints.min_bins:
            # Merge in descending p-value order; indices shift after each merge
            order = over[np.argsort(-pvals[over])]
            shift = 0
            for idx in order:
                real_idx = int(idx - shift)
                blocks = _merge_once(blocks, merged, real_idx)
                shift += 1
            continue

        # Otherwise lower threshold and try again
        new_pthr = BinningConstraints.next_threshold(pthr)
        if new_pthr == pthr:
            break
        pthr = new_pthr

    # Final sweep to satisfy min_samples if possible
    blocks = _enforce_min_samples(blocks, constraints, is_binary_y)
    return blocks
