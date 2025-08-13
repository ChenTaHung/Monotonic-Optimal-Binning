from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union
import math
import numpy as np

from .constraints import BinningConstraints


@dataclass
class Block:
    """A contiguous bin (closed on the left, open on the right).

    We store sufficient statistics to merge in O(1):
    - n: count
    - sum, sum2: for mean/variance
    - ymin, ymax: min/max within the block

    Note: `right` is the *raw* GCM/PAVA edge. The public bins table uses
    “next-left” as the effective `right` and sets the last `right` to +inf.
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
        """Unbiased sample variance computed from (sum, sum2, n)."""
        if self.n <= 1:
            return 0.0
        num = self.sum2 - (self.sum * self.sum) / self.n
        v = num / (self.n - 1)
        return max(v, 0.0)

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def merge_with(self, other: "Block") -> "Block":
        """Return a new block equal to the union of self and other (adjacent)."""
        n = self.n + other.n
        s = self.sum + other.sum
        s2 = self.sum2 + other.sum2
        ymin = min(self.ymin, other.ymin)
        ymax = max(self.ymax, other.ymax)
        return Block(
            left=self.left,
            right=other.right,  # keep the outer-right boundary
            n=n,
            sum=s,
            sum2=s2,
            ymin=ymin,
            ymax=ymax,
        )


# ------------------------- stats helpers (Welch-ish) ------------------------- #

def _phi_cdf(z: float) -> float:
    """Standard normal CDF via erfc for numeric stability."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _two_sample_pvalue(a: Block, b: Block) -> float:
    """Two-sample test on means with pooled variance (Welch-lite).

    We keep it simple/fast because this is used many times in greedy merging.
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


def _penalize_for_constraints(p: float, a: Block, b: Block,
                              constraints: BinningConstraints,
                              is_binary_y: bool) -> float:
    """Heuristic penalty/bonus to bias the merge choice under constraints."""
    out = p

    # Encourage fixing tiny bins (push p upward => looks “more mergeable”)
    if constraints.abs_min_samples:
        if a.n < constraints.abs_min_samples or b.n < constraints.abs_min_samples:
            out *= 1.5

    # For binary targets, extreme purity tends to be noisy; nudge to merge
    if is_binary_y:
        if (a.mean in (0.0, 1.0)) or (b.mean in (0.0, 1.0)):
            out *= 1.25

    # Discourage producing an oversized bin (reduce p so it’s harder to merge)
    if constraints.abs_max_samples:
        if a.n + b.n > constraints.abs_max_samples:
            out *= 0.5

    return out


def _pair_score(a: Block, b: Block,
                constraints: BinningConstraints,
                is_binary_y: bool) -> float:
    base = _two_sample_pvalue(a, b)
    return _penalize_for_constraints(base, a, b, constraints, is_binary_y)


def _best_adjacent_index(blocks: Sequence[Block],
                         constraints: BinningConstraints,
                         is_binary_y: bool) -> Optional[int]:
    """Pick the most “mergeable” adjacent pair by highest adjusted score."""
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
    merged = blocks[idx].merge_with(blocks[idx + 1])
    return blocks[:idx] + [merged] + blocks[idx + 2:]


# -------------------------- hard min-samples sweep --------------------------- #

def _sweep_min_samples(blocks: List[Block],
                       constraints: BinningConstraints,
                       is_binary_y: bool) -> List[Block]:
    """Make every bin meet abs_min_samples if possible, but stop at min_bins.

    Property-test rule we honor:
      Either all bins meet min-samples OR len(bins) <= min_bins (we stop).
    """
    if not constraints.abs_min_samples:
        return blocks

    while True:
        # Stop if we’ve hit the floor; caller may allow undersized bins at that point.
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

        left_sc = _pair_score(blocks[i - 1], blocks[i], constraints, is_binary_y)
        right_sc = _pair_score(blocks[i], blocks[i + 1], constraints, is_binary_y)
        blocks = _merge_at(blocks, i if right_sc >= left_sc else i - 1)

    return blocks


# ------------------------- adapters + main merge loop ------------------------ #

def blocks_from_dicts(rows: Iterable[dict]) -> List[Block]:
    """Build Block list from a list of dictionaries (safe copy)."""
    out: List[Block] = []
    for r in rows:
        out.append(Block(
            left=float(r["left"]),
            right=float(r["right"]),
            n=int(r["n"]),
            sum=float(r["sum"]),
            sum2=float(r["sum2"]),
            ymin=float(r.get("ymin", r.get("min", float("inf")))),
            ymax=float(r.get("ymax", r.get("max", float("-inf")))),
        ))
    return out


def as_blocks(rows: Union[List[dict], List[Block]]) -> List[Block]:
    """Accept either prebuilt Blocks or dicts; always return a fresh List[Block]."""
    if rows and isinstance(rows[0], dict):
        return blocks_from_dicts(rows)  # type: ignore[return-value]
    # Copy to avoid side effects
    return [Block(**vars(b)) for b in rows]  # shallow copy of dataclass fields


def merge_adjacent(blocks: Union[List[Block], List[dict]],
                   constraints: BinningConstraints,
                   is_binary_y: bool) -> List[Block]:
    """Greedy adjacent merges + hard min-samples sweep.

    Behavior by mode:
      - maximize_bins=True  → reduce until <= max_bins (p-value threshold helps choose which).
      - maximize_bins=False → *preserve at least min_bins* (never merge past this floor),
                              unless you’re already down to 1 bin.

    The “anneal p” behavior (lower p if still violating limits) is implemented at
    a higher level by the caller if needed; here we implement the core greedy pass.
    """
    # Accept dict blocks too (some unit tests pass dicts)
    if blocks and isinstance(blocks[0], dict):
        blocks = blocks_from_dicts(blocks)  # type: ignore[assignment]

    blocks = list(blocks)  # copy
    if not blocks:
        return []

    while True:
        if len(blocks) <= 1:
            break
        best = _best_adjacent_index(blocks, constraints, is_binary_y)
        if best is None:
            break

        # If we still exceed max_bins, keep merging the best pair outright
        if constraints.maximize_bins and len(blocks) > constraints.max_bins:
            blocks = _merge_at(blocks, best)
            continue

        # Evaluate statistical score for the chosen pair
        score = _pair_score(blocks[best], blocks[best + 1], constraints, is_binary_y)

        # ------------------------------------------------------------------
        # NEW GUARD: In "minimize merges" mode (maximize_bins=False),
        # do not merge if that would drop us *below* min_bins.
        # (Property tests expect: len(bins) >= min_bins or len(bins) == 1.)
        # ------------------------------------------------------------------
        if not constraints.maximize_bins:
            would_len = len(blocks) - 1
            # Never go below the requested floor (unless we’re already at 1)
            if would_len < max(1, constraints.min_bins):
                break

        # Merge by threshold OR if still over max_bins (rare when maximize_bins=False)
        if score >= constraints.initial_pvalue or len(blocks) > (constraints.max_bins or float("inf")):
            blocks = _merge_at(blocks, best)
            continue

        # No more merges by p-value
        break

    # Enforce min-samples, but do not go below min_bins
    blocks = _sweep_min_samples(blocks, constraints, is_binary_y)
    return blocks


__all__ = ["Block", "merge_adjacent", "blocks_from_dicts", "as_blocks"]
