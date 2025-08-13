from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union, Any
import math
import numpy as np

from .constraints import BinningConstraints


@dataclass
class Block:
    """Contiguous block with sufficient statistics (mean-only pipeline).

    Attributes:
        left: Left (inclusive) x-edge for the block.
        right: Right (exclusive) x-edge for the block.
        n: Number of rows in the block.
        sum: Sum of y over the block.
        sum2: Sum of y**2 over the block (for variance).
        ymin: Minimum y in the block.
        ymax: Maximum y in the block.
    """

    left: float
    right: float
    n: int
    sum: float
    sum2: float
    ymin: float
    ymax: float

    # -------- derived stats -------- #

    @property
    def mean(self) -> float:
        """Sample mean (0.0 for empty blocks)."""
        return self.sum / self.n if self.n > 0 else 0.0

    @property
    def var(self) -> float:
        """Unbiased sample variance from sufficient statistics (>= 0)."""
        if self.n <= 1:
            return 0.0
        num = self.sum2 - (self.sum * self.sum) / self.n
        v = num / (self.n - 1)
        return max(v, 0.0)

    @property
    def std(self) -> float:
        """Sample standard deviation (>= 0)."""
        return math.sqrt(self.var)

    # -------- operations -------- #

    def merge_with(self, other: "Block") -> "Block":
        """Return a new block that is the union of ``self`` and ``other``.

        The merged block spans from ``self.left`` to ``other.right`` and pools
        all sufficient statistics for O(1) updates.
        """
        n = self.n + other.n
        s = self.sum + other.sum
        s2 = self.sum2 + other.sum2
        ymin = min(self.ymin, other.ymin)
        ymax = max(self.ymax, other.ymax)
        return Block(
            left=self.left,
            right=other.right,
            n=n,
            sum=s,
            sum2=s2,
            ymin=ymin,
            ymax=ymax,
        )

    # Export for safe history snapshots (plots/GIFs)
    def as_dict(self) -> dict:
        return dict(
            left=float(self.left),
            right=float(self.right),
            n=int(self.n),
            sum=float(self.sum),
            sum2=float(self.sum2),
            ymin=float(self.ymin),
            ymax=float(self.ymax),
            mean=float(self.mean),
            std=float(self.std),
        )


# --------------------------------------------------------------------------- #
# Internal helpers (two-sample test + constraint-aware scoring)
# --------------------------------------------------------------------------- #

def _phi_cdf(z: float) -> float:
    """Standard normal CDF via erfc (stable for tails)."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _two_sample_pvalue(a: Block, b: Block) -> float:
    """Two-sample (pooled) z/t-like test on means → p-value (symmetric).

    Heuristic used to prioritize merges: higher p-value ⇒ weaker evidence
    the two blocks differ ⇒ more attractive to merge.

    Notes:
        * Uses pooled variance; returns 1.0 for degenerate cases.
        * Mean-only pipeline by design; future metrics (median/quantile)
          would require a different scoring/test strategy.
    """
    na, nb = a.n, b.n
    if na == 0 or nb == 0:
        return 1.0
    va, vb = a.var, b.var
    df = max(na + nb - 2, 1)
    pooled = ((na - 1) * va + (nb - 1) * vb) / df
    if pooled <= 0:
        return 1.0
    denom = math.sqrt(pooled * (1.0 / na + 1.0 / nb))
    if denom == 0:
        return 1.0
    z = abs(a.mean - b.mean) / denom
    return 2.0 * (1.0 - _phi_cdf(z))


def _penalize_for_constraints(
    p: float,
    a: Block,
    b: Block,
    constraints: BinningConstraints,
    is_binary_y: bool,
) -> float:
    """Adjust a base score using constraint heuristics.

    Larger value is a stronger merge signal. We up/down-scale by:
      * small bins vs. abs_min_samples (discourage tiny bins),
      * binary extreme means 0/1 (nudge merges to smooth),
      * exceeding abs_max_samples when merged (encourage splitting or other merges).
    """
    out = p
    if constraints.abs_min_samples:
        if a.n < constraints.abs_min_samples or b.n < constraints.abs_min_samples:
            out *= 1.5  # encourage merging small bins
    if is_binary_y:
        if (a.mean in (0.0, 1.0)) or (b.mean in (0.0, 1.0)):
            out *= 1.25  # encourage merging saturated-rate bins
    if constraints.abs_max_samples:
        if a.n + b.n > constraints.abs_max_samples:
            out *= 0.5  # merging would violate max-samples → downweight
    return out


def _pair_score(a: Block, b: Block, constraints: BinningConstraints, is_binary_y: bool) -> float:
    """Score a candidate adjacent merge (higher = more mergeable)."""
    base = _two_sample_pvalue(a, b)
    return _penalize_for_constraints(base, a, b, constraints, is_binary_y)


def _best_adjacent_index(
    blocks: Sequence[Block],
    constraints: BinningConstraints,
    is_binary_y: bool,
) -> Optional[int]:
    """Return index i of the best (i, i+1) adjacent pair to merge, or None."""
    if len(blocks) < 2:
        return None
    best_idx: Optional[int] = None
    best_score = -1.0
    for i in range(len(blocks) - 1):
        sc = _pair_score(blocks[i], blocks[i + 1], constraints, is_binary_y)
        if sc > best_score:
            best_score = sc
            best_idx = i
    return best_idx


def _merge_at(blocks: List[Block], idx: int) -> List[Block]:
    """Merge blocks[idx] and blocks[idx+1] and return a new list."""
    merged = blocks[idx].merge_with(blocks[idx + 1])
    return blocks[:idx] + [merged] + blocks[idx + 2:]


def _snapshot(blocks: Sequence[Block]) -> List[dict]:
    """Export a snapshot of the current block list (for history/GIFs)."""
    return [b.as_dict() for b in blocks]


def _sweep_min_samples(
    blocks: List[Block],
    constraints: BinningConstraints,
    is_binary_y: bool,
    history: Optional[List[List[dict]]] = None,
) -> List[Block]:
    """Enforce abs_min_samples when possible, stopping at min_bins.

    Property-style rule honored by tests:
      - Either all bins meet abs_min_samples, or we stop once we hit min_bins.

    We greedily merge an undersized bin with its better-scored neighbor.
    """
    if not constraints.abs_min_samples:
        return blocks

    while True:
        # Stop if we’ve hit min_bins; caller may allow some undersized bins then.
        if len(blocks) <= max(1, constraints.min_bins):
            break

        small = [i for i, b in enumerate(blocks) if b.n < constraints.abs_min_samples]
        if not small:
            break

        i = small[0]
        # If at either edge, we have only one merge direction.
        if i == 0:
            blocks = _merge_at(blocks, 0)
            if history is not None:
                history.append(_snapshot(blocks))
            continue
        if i == len(blocks) - 1:
            blocks = _merge_at(blocks, i - 1)
            if history is not None:
                history.append(_snapshot(blocks))
            continue

        # Choose side by higher merge score (stat + penalties).
        left_sc = _pair_score(blocks[i - 1], blocks[i], constraints, is_binary_y)
        right_sc = _pair_score(blocks[i], blocks[i + 1], constraints, is_binary_y)
        blocks = _merge_at(blocks, i if right_sc >= left_sc else i - 1)
        if history is not None:
            history.append(_snapshot(blocks))

    return blocks


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #

def blocks_from_dicts(rows: Iterable[dict[str, Any]]) -> List[Block]:
    """Coerce an iterable of dicts to a typed list of ``Block`` instances.

    Each dict must provide: ``left, right, n, sum, sum2`` and either
    ``ymin/ymax`` or legacy aliases ``min/max``.
    """
    out: List[Block] = []
    for r in rows:
        out.append(
            Block(
                left=float(r["left"]),
                right=float(r["right"]),
                n=int(r["n"]),
                sum=float(r["sum"]),
                sum2=float(r["sum2"]),
                ymin=float(r.get("ymin", r.get("min", float("inf")))),
                ymax=float(r.get("ymax", r.get("max", float("-inf")))),
            )
        )
    return out


def as_blocks(rows: Union[List[dict[str, Any]], List[Block]]) -> List[Block]:
    """Accept list of dicts or list of ``Block`` and return list of ``Block``."""
    if not rows:
        return []
    if isinstance(rows[0], Block):  # type: ignore[index]
        return list(rows)  # already typed
    return blocks_from_dicts(rows)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Main API
# --------------------------------------------------------------------------- #

def merge_adjacent(
    blocks: Union[List[Block], List[dict[str, Any]]],
    constraints: BinningConstraints,
    is_binary_y: bool,
    *,
    history: Optional[List[List[dict]]] = None,
) -> List[Block]:
    """Greedy adjacent merges followed by a min-samples sweep.

    This implements the “adjacent merge” phase after PAVA:
      1) While a good merge exists:
         - If ``maximize_bins`` and we're over ``max_bins``, keep merging best pairs.
         - Else merge when the scored p-value (with penalties) ≥ ``initial_pvalue``.
      2) Run a hard sweep to enforce ``abs_min_samples``, but never go below ``min_bins``.

    Args:
        blocks: Input blocks (as ``Block`` objects or dicts). Typically the
            output of `PAVA.export_blocks(as_dict=True)` coerced here.
        constraints: Resolved binning constraints.
        is_binary_y: If True, apply a few binary-specific score nudges.
        history: Optional list that will be **appended** with snapshots (each is
            a list of dicts) after each merge *and* after the final min-samples
            sweep. Passing None disables history capture.

    Returns:
        List[Block]: The merged, contiguous blocks.

    Notes:
        * Metric is **mean-only** in this codebase. Median/quantile support is
          a common request and is tracked as future work because it requires
          different tests and merge scoring.
    """
    # Coerce to List[Block] (accept dict form from PAVA export).
    blocks_typed: List[Block] = as_blocks(blocks)

    # Work on a copy to avoid mutating caller-provided lists.
    blocks_cur: List[Block] = list(blocks_typed)
    if not blocks_cur:
        return []

    # Greedy adjacent merges
    while True:
        if len(blocks_cur) <= 1:
            break

        best = _best_adjacent_index(blocks_cur, constraints, is_binary_y)
        if best is None:
            break

        # If we still exceed max_bins, keep merging the best pair outright.
        if constraints.maximize_bins and len(blocks_cur) > constraints.max_bins:
            blocks_cur = _merge_at(blocks_cur, best)
            if history is not None:
                history.append(_snapshot(blocks_cur))
            continue

        score = _pair_score(blocks_cur[best], blocks_cur[best + 1], constraints, is_binary_y)

        # Merge by threshold OR if still over max_bins.
        if score >= constraints.initial_pvalue or (
            constraints.maximize_bins and len(blocks_cur) > constraints.max_bins
        ):
            blocks_cur = _merge_at(blocks_cur, best)
            if history is not None:
                history.append(_snapshot(blocks_cur))
            continue

        # No more merges at current threshold.
        break

    # Enforce min-samples, but do not go below min_bins
    blocks_cur = _sweep_min_samples(blocks_cur, constraints, is_binary_y, history=history)

    return blocks_cur


__all__ = ["Block", "merge_adjacent", "blocks_from_dicts", "as_blocks"]
