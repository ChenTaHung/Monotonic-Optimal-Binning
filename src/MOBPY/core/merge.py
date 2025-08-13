from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union
import math
import numpy as np

from .constraints import BinningConstraints


@dataclass
class Block:
    """Contiguous bin with pooled statistics (sufficient stats for O(1) merges).

    Attributes:
        left: Left edge (inclusive).
        right: Right edge (exclusive). (PAVA gives us “next-left”; final last right becomes +inf.)
        n: Row count in the block.
        sum: Sum of target y in the block.
        sum2: Sum of squares of y in the block (for pooled variance).
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

    @property
    def mean(self) -> float:
        return self.sum / self.n if self.n > 0 else 0.0

    @property
    def var(self) -> float:
        """Unbiased pooled sample variance from aggregated stats."""
        if self.n <= 1:
            return 0.0
        num = self.sum2 - (self.sum * self.sum) / self.n
        v = num / (self.n - 1)
        return max(v, 0.0)

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def merge_with(self, other: "Block") -> "Block":
        """Return a new Block representing self ∪ other (adjacent)."""
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


def _phi_cdf(z: float) -> float:
    """Standard normal CDF via erfc (stable for tails)."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _two_sample_pvalue(a: Block, b: Block) -> float:
    """Two-sample z-test (pooled variance) p-value for mean difference.

    For very small/degenerate bins, returns 1.0 (no evidence to split).
    """
    na, nb = a.n, b.n
    if na == 0 or nb == 0:
        return 1.0
    va, vb = a.var, b.var
    df = max(na + nb - 2, 1)  # pooled df protection
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
    """Inflate/deflate the p-value based on constraint hints.

    * Smaller-than-min bins → inflate p to encourage merging.
    * Binary degenerate means (0 or 1) → moderate inflation (encourage merge).
    * Exceeding abs_max_samples if merged → deflate p to discourage merging.
    """
    out = p
    if constraints.abs_min_samples:
        if a.n < constraints.abs_min_samples or b.n < constraints.abs_min_samples:
            out *= 1.5
    if is_binary_y:
        if (a.mean in (0.0, 1.0)) or (b.mean in (0.0, 1.0)):
            out *= 1.25
    if constraints.abs_max_samples:
        if a.n + b.n > constraints.abs_max_samples:
            out *= 0.5
    return out


def _pair_score(a: Block, b: Block, constraints: BinningConstraints, is_binary_y: bool) -> float:
    """Score two adjacent blocks by p-value with constraint penalties."""
    base = _two_sample_pvalue(a, b)
    return _penalize_for_constraints(base, a, b, constraints, is_binary_y)


def _best_adjacent_index(
    blocks: Sequence[Block],
    constraints: BinningConstraints,
    is_binary_y: bool,
) -> Optional[int]:
    """Return index i of best pair (i, i+1) to merge, by highest score."""
    if len(blocks) < 2:
        return None
    best_idx = None
    best_score = -1.0
    for i in range(len(blocks) - 1):
        sc = _pair_score(blocks[i], blocks[i + 1], constraints, is_binary_y)
        if sc > best_score:
            best_score = sc
            best_idx = i
    return best_idx


def _merge_at(blocks: List[Block], idx: int) -> List[Block]:
    """Return a new list with blocks[idx] merged into blocks[idx+1]."""
    merged = blocks[idx].merge_with(blocks[idx + 1])
    return blocks[:idx] + [merged] + blocks[idx + 2 :]


def _sweep_min_samples(
    blocks: List[Block],
    constraints: BinningConstraints,
    is_binary_y: bool,
) -> List[Block]:
    """Enforce `abs_min_samples` greedily without going below `min_bins`.

    Property we aim to satisfy for tests:
      - Either **all** bins meet min-samples, or we have exactly `min_bins` bins.
    """
    if not constraints.abs_min_samples:
        return blocks

    while True:
        if len(blocks) <= max(1, constraints.min_bins):
            break

        small = [i for i, b in enumerate(blocks) if b.n < constraints.abs_min_samples]
        if not small:
            break

        i = small[0]
        if i == 0:
            blocks = _merge_at(blocks, 0)
            continue
        if i == len(blocks) - 1:
            blocks = _merge_at(blocks, i - 1)
            continue

        # Merge towards side with weaker separation (higher p-value)
        left_sc = _pair_score(blocks[i - 1], blocks[i], constraints, is_binary_y)
        right_sc = _pair_score(blocks[i], blocks[i + 1], constraints, is_binary_y)
        blocks = _merge_at(blocks, i if right_sc >= left_sc else i - 1)

    return blocks


def blocks_from_dicts(rows: Iterable[dict]) -> List[Block]:
    """Coerce list of dicts (e.g., from PAVA export) into Block objects."""
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


def as_blocks(rows):
    """Alias used throughout the code/tests: dicts → Block list."""
    return blocks_from_dicts(rows)


def merge_adjacent(
    blocks: Union[List[Block], List[dict]],
    constraints: BinningConstraints,
    is_binary_y: bool,
) -> List[Block]:
    """Greedy adjacent merges + hard min-samples sweep.

    Behavior:
      * If `blocks` are dicts, we coerce to `Block`.
      * If we exceed `max_bins` (maximize mode), we keep merging best pairs
        regardless of p-value until `len(blocks) <= max_bins`.
      * Otherwise, we merge only if the scored p-value exceeds
        `constraints.initial_pvalue`.
      * Finally, we run `_sweep_min_samples` to lift undersized bins, stopping
        at `min_bins` (so some undersized bins can remain if we’re at the floor).
    """
    # Accept dict blocks too (some tests pass dicts explicitly)
    if blocks and isinstance(blocks[0], dict):
        blocks = blocks_from_dicts(blocks)  # type: ignore[assignment]

    blocks = list(blocks)  # shallow copy
    if not blocks:
        return []

    while True:
        if len(blocks) <= 1:
            break
        best = _best_adjacent_index(blocks, constraints, is_binary_y)
        if best is None:
            break

        # If we still exceed max_bins, merge best pair outright
        if constraints.maximize_bins and len(blocks) > constraints.max_bins:
            blocks = _merge_at(blocks, best)
            continue

        score = _pair_score(blocks[best], blocks[best + 1], constraints, is_binary_y)
        if score >= constraints.initial_pvalue:
            blocks = _merge_at(blocks, best)
            continue

        # No eligible merges at this threshold
        break

    # Enforce absolute min-samples, but do not go below min_bins
    blocks = _sweep_min_samples(blocks, constraints, is_binary_y)
    return blocks


__all__ = ["Block", "merge_adjacent", "blocks_from_dicts", "as_blocks"]
