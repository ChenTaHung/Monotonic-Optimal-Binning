from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union, overload, cast
import math
import numpy as np

from .constraints import BinningConstraints


@dataclass
class Block:
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
        if self.n <= 1:
            return 0.0
        num = self.sum2 - (self.sum * self.sum) / self.n
        v = num / (self.n - 1)
        return max(v, 0.0)

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def merge_with(self, other: "Block") -> "Block":
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
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _two_sample_pvalue(a: Block, b: Block) -> float:
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


def _pair_score(a: Block, b: Block,
                constraints: BinningConstraints,
                is_binary_y: bool) -> float:
    base = _two_sample_pvalue(a, b)
    return _penalize_for_constraints(base, a, b, constraints, is_binary_y)


def _best_adjacent_index(blocks: Sequence[Block],
                         constraints: BinningConstraints,
                         is_binary_y: bool) -> Optional[int]:
    if len(blocks) < 2:
        return None
    best_idx: Optional[int] = None
    best_score = -1.0
    for i in range(len(blocks) - 1):
        sc = _pair_score(blocks[i], blocks[i+1], constraints, is_binary_y)
        if sc > best_score:
            best_score = sc
            best_idx = i
    return best_idx


def _merge_at(blocks: List[Block], idx: int) -> List[Block]:
    merged = blocks[idx].merge_with(blocks[idx+1])
    return blocks[:idx] + [merged] + blocks[idx+2:]


def _sweep_min_samples(blocks: List[Block],
                       constraints: BinningConstraints,
                       is_binary_y: bool) -> List[Block]:
    """Make every bin meet abs_min_samples if possible, but stop at min_bins."""
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

        left_sc = _pair_score(blocks[i-1], blocks[i], constraints, is_binary_y)
        right_sc = _pair_score(blocks[i], blocks[i+1], constraints, is_binary_y)
        blocks = _merge_at(blocks, i if right_sc >= left_sc else i - 1)

    return blocks


def blocks_from_dicts(rows: Iterable[Dict[str, Any]]) -> List[Block]:
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


@overload
def as_blocks(rows: Iterable[Dict[str, Any]]) -> List[Block]: ...
@overload
def as_blocks(rows: Iterable[Block]) -> List[Block]: ...
def as_blocks(rows: Iterable[Union[Dict[str, Any], Block]]) -> List[Block]:
    """Coerce either dict-rows or Block-rows into List[Block]."""
    lst = list(rows)
    if not lst:
        return []
    if isinstance(lst[0], Block):
        return cast(List[Block], lst)
    return blocks_from_dicts(cast(Iterable[Dict[str, Any]], lst))


def _coerce_blocks(blocks: Union[Iterable[Block], Iterable[Dict[str, Any]]]) -> List[Block]:
    """Internal: always return a concrete List[Block] for downstream funcs."""
    return as_blocks(blocks)  # as_blocks already handles both cases


def merge_adjacent(blocks: Union[Iterable[Block], Iterable[Dict[str, Any]]],
                   constraints: BinningConstraints,
                   is_binary_y: bool) -> List[Block]:
    """Greedy adjacent merges + hard min-samples sweep.

    Accepts either iterables of Block or dict rows; returns concrete List[Block].
    """
    blks: List[Block] = _coerce_blocks(blocks)
    if not blks:
        return []

    while True:
        if len(blks) <= 1:
            break
        best = _best_adjacent_index(blks, constraints, is_binary_y)
        if best is None:
            break
        
        # In "min-bins" mode, never merge once we are at or below the target.
        if not constraints.maximize_bins and len(blks) <= max(1, constraints.min_bins):
            break
        
        if constraints.maximize_bins and len(blks) > constraints.max_bins:
            blks = _merge_at(blks, best)
            continue

        score = _pair_score(blks[best], blks[best+1], constraints, is_binary_y)
        if score >= constraints.initial_pvalue or len(blks) > constraints.max_bins:
            blks = _merge_at(blks, best)
            continue
        
        break

    blks = _sweep_min_samples(blks, constraints, is_binary_y)
    return blks


__all__ = ["Block", "merge_adjacent", "blocks_from_dicts", "as_blocks"]
