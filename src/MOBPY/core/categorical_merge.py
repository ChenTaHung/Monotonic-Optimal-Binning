"""Categorical x merging using chi-square tests with multiple comparison correction.

This module implements the merging phase for categorical x features. It uses
chi-square contingency tests to measure similarity between category groups and
applies multiple comparison correction (Holm by default) to control false merges.

Key design choices:
- Pair-result cache with local invalidation achieves O(k²) total chi-square calls
  instead of the naive O(k³) (where k = initial number of categories).
- After merging (u, v) → w, only pairs touching u or v are invalidated; the
  O(k) new pairs for w are recomputed. All other cached results remain valid.
- Multiple comparison correction is reapplied each iteration to the current
  active set of pairs, ensuring globally controlled error rates.
- Three merging phases mirror the numeric path:
    Phase 1: Statistical merging (chi-square + correction).
    Phase 2: Enforce min_samples per bin.
    Phase 3: Enforce min_positives / min_negatives per bin (binary only).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
import scipy.stats

from MOBPY.core.constraints import BinningConstraints
from MOBPY.core.merge import _BlockStatsBase
from MOBPY.exceptions import FittingError
from MOBPY.config import get_config
from MOBPY.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChiResult:
    """Result of a chi-square test between two CategoryBlocks.

    Attributes:
        chi2: Chi-square test statistic.
        pvalue: Raw p-value (higher = more similar = merge candidate).
        dof: Degrees of freedom (always 1 for 2×2 binary table).
        n_obs: Total observations in both blocks.
    """

    chi2: float
    pvalue: float
    dof: int
    n_obs: int


@dataclass
class CategoryBlock(_BlockStatsBase):
    """Block for a group of categories with aggregated y statistics.

    Inherits shared sufficient statistics (n, sum, sum2, ymin, ymax) and all
    derived properties (mean, var, std, positives, negatives) from
    _BlockStatsBase.

    Attributes:
        categories: Frozenset of original category values in this block.
        merge_history: Sequence of (left_cats, right_cats) merges that formed
            this block, for debugging.
        pvalue_history: Adjusted p-values recorded at each merge step.
    """

    categories: FrozenSet[Any]
    merge_history: List[Tuple[FrozenSet[Any], FrozenSet[Any]]] = field(
        default_factory=list, compare=False
    )
    pvalue_history: List[float] = field(default_factory=list, compare=False)

    def merge_with(self, other: "CategoryBlock") -> "CategoryBlock":
        """Merge with another CategoryBlock, pooling statistics.

        Args:
            other: Block to merge with.

        Returns:
            CategoryBlock: New merged block with combined statistics and
                union of categories.
        """
        merged = CategoryBlock(
            n=self.n + other.n,
            sum=self.sum + other.sum,
            sum2=self.sum2 + other.sum2,
            ymin=min(self.ymin, other.ymin),
            ymax=max(self.ymax, other.ymax),
            categories=self.categories | other.categories,
        )
        merged.merge_history = (
            self.merge_history
            + [(self.categories, other.categories)]
            + other.merge_history
        )
        return merged

    def as_dict(self) -> Dict[str, Any]:
        """Export block as dictionary.

        Returns:
            Dict with all block statistics, sorted category list, and derived values.
        """
        return {
            "categories": sorted(str(c) for c in self.categories),
            "n": int(self.n),
            "sum": float(self.sum),
            "sum2": float(self.sum2),
            "ymin": float(self.ymin),
            "ymax": float(self.ymax),
            "mean": float(self.mean),
            "var": float(self.var),
            "std": float(self.std),
            "positives": float(self.positives),
            "negatives": float(self.negatives),
        }

    def __repr__(self) -> str:
        cats = sorted(str(c) for c in self.categories)
        return (
            f"CategoryBlock({cats}, n={self.n}, "
            f"mean={self.mean:.4f}, std={self.std:.4f})"
        )


# ---------------------------------------------------------------------------
# Statistical test
# ---------------------------------------------------------------------------

def _chi2_test(a: CategoryBlock, b: CategoryBlock) -> ChiResult:
    """Compute chi-square test of independence for two CategoryBlocks.

    Builds a 2×2 contingency table:

        .. code-block::

                  y=0          y=1
            a:  a.negatives  a.positives
            b:  b.negatives  b.positives

    Args:
        a: First block.
        b: Second block.

    Returns:
        ChiResult: Chi-square statistic, p-value, dof, and total n.
        Returns pvalue=1.0 for degenerate tables (all-zero rows/columns).
    """
    n_obs = int(a.n + b.n)

    # Degenerate: empty blocks
    if a.n == 0 or b.n == 0:
        return ChiResult(chi2=0.0, pvalue=1.0, dof=1, n_obs=n_obs)

    table = np.array(
        [[a.negatives, a.positives], [b.negatives, b.positives]],
        dtype=float,
    )

    # Degenerate: zero-sum row (one block has no data) — already handled above
    # Degenerate: zero-sum column (no events or no non-events in either block)
    if table[:, 0].sum() == 0 or table[:, 1].sum() == 0:
        return ChiResult(chi2=0.0, pvalue=1.0, dof=1, n_obs=n_obs)

    try:
        # correction=False: Yates' correction is not applied; multiple comparison
        # correction via Holm/Bonferroni is handled separately.
        chi2, pvalue, dof, _ = scipy.stats.chi2_contingency(table, correction=False)
        return ChiResult(
            chi2=float(chi2),
            pvalue=float(np.clip(pvalue, 0.0, 1.0)),
            dof=int(dof),
            n_obs=n_obs,
        )
    except Exception:
        return ChiResult(chi2=0.0, pvalue=1.0, dof=1, n_obs=n_obs)


# ---------------------------------------------------------------------------
# Multiple comparison correction
# ---------------------------------------------------------------------------

def _adjust_pvalues(pvalues: List[float], correction: str) -> np.ndarray:
    """Apply multiple comparison correction to a list of raw p-values.

    Supported methods:

    * ``'bonferroni'``: Multiply each p-value by the number of tests.
    * ``'holm'`` (default): Holm-Bonferroni step-down procedure. Strictly
      more powerful than Bonferroni while controlling FWER.
    * ``'fdr_bh'``: Benjamini-Hochberg FDR control.

    Args:
        pvalues: List of raw p-values, one per active pair.
        correction: Name of the correction method.

    Returns:
        np.ndarray: Adjusted p-values in the same order as ``pvalues``,
        clipped to [0, 1].

    Raises:
        ValueError: If correction name is not recognised.
    """
    n = len(pvalues)
    if n == 0:
        return np.array([], dtype=float)

    p_arr = np.array(pvalues, dtype=float)

    if correction == "bonferroni":
        return np.clip(p_arr * n, 0.0, 1.0)

    elif correction == "holm":
        # Sort ascending; multiply p_(i) by (n - i); take running maximum
        # to enforce non-decreasing adjusted values; then unsort.
        order = np.argsort(p_arr)
        adjusted = np.empty(n, dtype=float)
        running_max = 0.0
        for rank, idx in enumerate(order):
            adj = (n - rank) * p_arr[idx]
            running_max = max(running_max, adj)
            adjusted[idx] = running_max
        return np.clip(adjusted, 0.0, 1.0)

    elif correction == "fdr_bh":
        # Sort ascending; process from largest rank down; multiply p_(i)
        # by n/(i+1); take running minimum to enforce non-increasing then
        # unsort.
        order = np.argsort(p_arr)
        adjusted = np.empty(n, dtype=float)
        running_min = 1.0
        for rank in range(n - 1, -1, -1):
            idx = order[rank]
            adj = (n / (rank + 1)) * p_arr[idx]
            running_min = min(running_min, adj)
            adjusted[idx] = running_min
        return np.clip(adjusted, 0.0, 1.0)

    else:
        raise ValueError(
            f"Unknown correction method {correction!r}. "
            f"Valid options: 'bonferroni', 'holm', 'fdr_bh'."
        )


# ---------------------------------------------------------------------------
# Constraint enforcement helpers (phases 2 & 3)
# ---------------------------------------------------------------------------

def _find_best_cat_partner(blocks: List[CategoryBlock], idx: int) -> Optional[int]:
    """Find the block most similar to ``blocks[idx]`` by chi-square p-value.

    Used during constraint-enforcement phases where a block must be merged
    but there is no ordering (unlike the numeric adjacent-only case).

    Args:
        blocks: Current list of CategoryBlocks.
        idx: Index of the block that needs a merge partner.

    Returns:
        Index of the best partner, or None if only one block remains.
    """
    if len(blocks) < 2:
        return None

    target = blocks[idx]
    best_p = -1.0
    best_partner: Optional[int] = None

    for i, b in enumerate(blocks):
        if i == idx:
            continue
        result = _chi2_test(target, b)
        if result.pvalue > best_p:
            best_p = result.pvalue
            best_partner = i

    return best_partner


def _enforce_cat_min_samples(
    blocks: List[CategoryBlock],
    constraints: BinningConstraints,
    history: Optional[List[List[Dict]]],
) -> List[CategoryBlock]:
    """Phase 2: Enforce minimum samples per bin for categorical blocks.

    Finds the smallest undersized block and merges it with its most
    statistically similar partner (highest chi-square p-value).

    Args:
        blocks: Current CategoryBlocks.
        constraints: Resolved binning constraints.
        history: Optional list to append merge snapshots to.

    Returns:
        Blocks with min_samples enforced where possible.
    """
    current = list(blocks)

    if constraints.abs_min_samples <= 0:
        return current

    logger.debug(f"Enforcing cat min_samples={constraints.abs_min_samples}")

    max_iterations = len(blocks) * 2
    iteration = 0

    while iteration < max_iterations:
        if len(current) <= max(1, constraints.min_bins):
            break

        undersized = [i for i, b in enumerate(current) if b.n < constraints.abs_min_samples]
        if not undersized:
            break

        # Choose the smallest violating block
        idx = min(undersized, key=lambda i: current[i].n)
        partner = _find_best_cat_partner(current, idx)
        if partner is None:
            break

        lo, hi = min(idx, partner), max(idx, partner)
        merged = current[lo].merge_with(current[hi])
        current = [b for i, b in enumerate(current) if i not in (lo, hi)] + [merged]

        if history is not None:
            history.append([b.as_dict() for b in current])

        logger.debug(
            f"Cat min-samples merge: ({lo},{hi}) -> 1 block, "
            f"{len(current)} blocks remain"
        )
        iteration += 1

    if iteration >= max_iterations:
        warnings.warn(
            f"Categorical min-samples enforcement reached max iterations ({max_iterations})",
            UserWarning,
        )

    return current


def _enforce_cat_min_class_counts(
    blocks: List[CategoryBlock],
    constraints: BinningConstraints,
    history: Optional[List[List[Dict]]],
) -> List[CategoryBlock]:
    """Phase 3: Enforce min_positives / min_negatives per bin.

    Finds the first bin violating either class-count constraint and merges
    it with its most statistically similar partner.

    Args:
        blocks: Current CategoryBlocks.
        constraints: Resolved binning constraints (abs_min_positives /
            abs_min_negatives must be set).
        history: Optional list to append merge snapshots to.

    Returns:
        Blocks with class-count constraints enforced where possible.
    """
    current = list(blocks)

    min_pos = constraints.abs_min_positives
    min_neg = constraints.abs_min_negatives

    if min_pos <= 0 and min_neg <= 0:
        return current

    logger.debug(f"Enforcing cat min_positives={min_pos}, min_negatives={min_neg}")

    max_iterations = len(blocks) * 2
    iteration = 0

    while iteration < max_iterations:
        if len(current) <= max(1, constraints.min_bins):
            logger.debug(
                f"Reached min_bins={constraints.min_bins} floor, "
                "stopping categorical class count enforcement"
            )
            break

        violating = [
            i
            for i, b in enumerate(current)
            if (min_pos > 0 and b.positives < min_pos)
            or (min_neg > 0 and b.negatives < min_neg)
        ]
        if not violating:
            logger.debug("All categorical bins satisfy class count constraints")
            break

        idx = violating[0]
        partner = _find_best_cat_partner(current, idx)
        if partner is None:
            break

        lo, hi = min(idx, partner), max(idx, partner)
        merged = current[lo].merge_with(current[hi])
        current = [b for i, b in enumerate(current) if i not in (lo, hi)] + [merged]

        if history is not None:
            history.append([b.as_dict() for b in current])

        logger.debug(
            f"Cat class-count merge: ({lo},{hi}) -> 1 block, "
            f"{len(current)} blocks remain"
        )
        iteration += 1

    if iteration >= max_iterations:
        warnings.warn(
            f"Categorical class count enforcement reached max iterations ({max_iterations})",
            UserWarning,
        )

    still_violating_pos = sum(1 for b in current if min_pos > 0 and b.positives < min_pos)
    still_violating_neg = sum(1 for b in current if min_neg > 0 and b.negatives < min_neg)
    if still_violating_pos > 0 or still_violating_neg > 0:
        logger.warning(
            f"Could not satisfy all categorical class count constraints: "
            f"{still_violating_pos} bins below min_positives, "
            f"{still_violating_neg} bins below min_negatives "
            f"(reached min_bins={constraints.min_bins} floor)"
        )

    return current


def _validate_cat_merge_result(
    blocks: List[CategoryBlock],
    constraints: BinningConstraints,
    is_binary_y: bool,
) -> None:
    """Validate the final categorical merge result and emit warnings.

    Args:
        blocks: Final CategoryBlocks after merging.
        constraints: Resolved binning constraints.
        is_binary_y: Whether the target is binary.

    Raises:
        FittingError: If max_bins is exceeded.
    """
    n_blocks = len(blocks)

    if constraints.maximize_bins and n_blocks > constraints.max_bins:
        raise FittingError(
            f"Categorical merge failed: {n_blocks} bins exceeds "
            f"max_bins={constraints.max_bins}"
        )

    undersized = [b for b in blocks if b.n < constraints.abs_min_samples]
    if undersized and n_blocks > constraints.min_bins:
        logger.warning(
            f"{len(undersized)} categorical bins have fewer than "
            f"{constraints.abs_min_samples} samples"
        )
    elif undersized:
        warnings.warn(
            f"{len(undersized)} categorical bins have fewer than "
            f"min_samples={constraints.abs_min_samples}, but cannot merge "
            f"further without violating min_bins={constraints.min_bins}. "
            f"Consider relaxing min_samples or min_bins.",
            UserWarning,
        )

    if is_binary_y and constraints.abs_min_positives > 0:
        low_pos = [b for b in blocks if b.positives < constraints.abs_min_positives]
        if low_pos:
            msg = (
                f"{len(low_pos)} categorical bins have fewer than "
                f"min_positives={constraints.abs_min_positives}."
            )
            if n_blocks > constraints.min_bins:
                logger.warning(msg)
            else:
                warnings.warn(
                    msg + " Cannot merge further (min_bins floor). "
                    "WoE calculations may be unstable.",
                    UserWarning,
                )

    if is_binary_y and constraints.abs_min_negatives > 0:
        low_neg = [b for b in blocks if b.negatives < constraints.abs_min_negatives]
        if low_neg:
            msg = (
                f"{len(low_neg)} categorical bins have fewer than "
                f"min_negatives={constraints.abs_min_negatives}."
            )
            if n_blocks > constraints.min_bins:
                logger.warning(msg)
            else:
                warnings.warn(
                    msg + " Cannot merge further (min_bins floor). "
                    "WoE calculations may be unstable.",
                    UserWarning,
                )


# ---------------------------------------------------------------------------
# Main merge function
# ---------------------------------------------------------------------------

def merge_categorical(
    blocks: List[CategoryBlock],
    constraints: BinningConstraints,
    is_binary_y: bool,
    *,
    alpha: float = 0.05,
    correction: str = "holm",
    history: Optional[List[List[Dict]]] = None,
) -> List[CategoryBlock]:
    """Merge categorical blocks using chi-square tests with multiple comparison correction.

    Algorithm overview
    ------------------
    **Phase 1 – statistical merging:**

    1. Compute chi-square p-values for all pairs once: O(k²).
    2. Apply multiple comparison correction across all active pairs.
    3. Find the pair with the highest adjusted p-value (most similar).
    4. If adjusted p ≥ alpha (or max_bins is exceeded), merge the pair.
    5. After merging (u, v) → w:

       * Remove all cached results touching u or v.
       * Recompute only the O(k) new pairs (w, t) for each remaining t.
       * All unaffected pair results stay cached.

    6. Repeat until no pair qualifies or min_bins floor is reached.

    **Phase 2** – enforce min_samples per bin.

    **Phase 3** – enforce min_positives / min_negatives per bin (binary only).

    Complexity
    ----------
    * Naive recompute-all: O(k³) chi-square calls total.
    * Cached incremental: O(k²) chi-square calls total.

    Args:
        blocks: Initial CategoryBlocks — typically one per unique category
            value, built from the clean data partition.
        constraints: Resolved binning constraints (max_bins, min_bins,
            min_samples, min_positives, min_negatives).
        is_binary_y: Whether the target is binary (y ∈ {0, 1}). Must be
            True; non-binary y is not supported for categorical x.
        alpha: Significance level. Pairs with adjusted p-value ≥ alpha are
            merge candidates. Default 0.05.
        correction: Multiple comparison correction method: 'holm' (default),
            'bonferroni', or 'fdr_bh'.
        history: Optional list; merge snapshots (list of dicts) are appended
            after each merge for debugging / visualisation.

    Returns:
        List[CategoryBlock]: Merged blocks satisfying constraints.

    Raises:
        ValueError: If ``is_binary_y`` is False (chi-square requires binary y).
        FittingError: If merging produces zero blocks or violates max_bins.

    Examples:
        >>> blocks = [CategoryBlock(n=50, sum=10.0, sum2=10.0,
        ...                         ymin=0.0, ymax=1.0,
        ...                         categories=frozenset(['A']))]
        >>> constraints = BinningConstraints(max_bins=4, min_bins=2)
        >>> constraints.resolve(total_n=200, total_pos=40)
        >>> merged = merge_categorical(blocks, constraints, is_binary_y=True)
    """
    if not is_binary_y:
        raise ValueError(
            "Categorical merging with chi-square requires a binary target "
            "(y ∈ {0, 1}). Non-binary continuous y is not supported for "
            "categorical x."
        )

    if not blocks:
        return []

    if len(blocks) == 1:
        return list(blocks)

    # ------------------------------------------------------------------
    # Phase 1: Statistical merging with pair cache
    # ------------------------------------------------------------------

    # Assign integer IDs to blocks so cache keys are stable across merges.
    _next_id = [0]

    def _new_id() -> int:
        bid = _next_id[0]
        _next_id[0] += 1
        return bid

    active: Dict[int, CategoryBlock] = {}
    for b in blocks:
        active[_new_id()] = b

    # pair_cache[(id_a, id_b)] with id_a < id_b -> ChiResult
    pair_cache: Dict[Tuple[int, int], ChiResult] = {}
    active_pairs: Set[Tuple[int, int]] = set()

    ids = list(active.keys())
    for i, ia in enumerate(ids):
        for ib in ids[i + 1 :]:
            key = (ia, ib)  # ia < ib because ids is sorted ascending
            pair_cache[key] = _chi2_test(active[ia], active[ib])
            active_pairs.add(key)

    logger.info(
        f"Starting categorical merge: {len(active)} blocks, "
        f"target range [{constraints.min_bins}, {constraints.max_bins}], "
        f"alpha={alpha}, correction={correction}"
    )

    max_iter = len(blocks) * 2
    iteration = 0

    while iteration < max_iter and len(active) > 1:
        must_merge = len(active) > constraints.max_bins

        if len(active) <= constraints.min_bins and not must_merge:
            # Already at the min_bins floor; stop statistical merging.
            break

        pairs_list = sorted(active_pairs)
        if not pairs_list:
            break

        raw_pvals = [pair_cache[p].pvalue for p in pairs_list]
        adj_pvals = _adjust_pvalues(raw_pvals, correction)

        best_pos = int(np.argmax(adj_pvals))
        best_pair = pairs_list[best_pos]
        best_adj_p = float(adj_pvals[best_pos])

        # Decide whether to merge this pair.
        if must_merge:
            should_merge = True
        elif len(active) > constraints.min_bins and best_adj_p >= alpha:
            should_merge = True
        else:
            # No pair qualifies (all significantly different) or at floor.
            break

        if not should_merge:
            break

        ia, ib = best_pair
        merged_block = active[ia].merge_with(active[ib])
        merged_block.pvalue_history.append(best_adj_p)

        new_id = _new_id()

        # Invalidate only pairs that touch ia or ib.
        to_remove = {p for p in active_pairs if ia in p or ib in p}
        active_pairs -= to_remove
        for p in to_remove:
            pair_cache.pop(p, None)

        # Remove old blocks and register merged block.
        del active[ia]
        del active[ib]
        active[new_id] = merged_block

        # Recompute only the new pairs (new_id, t) for each remaining t.
        for existing_id in active:
            if existing_id == new_id:
                continue
            key = (min(new_id, existing_id), max(new_id, existing_id))
            pair_cache[key] = _chi2_test(merged_block, active[existing_id])
            active_pairs.add(key)

        if history is not None:
            history.append([b.as_dict() for b in active.values()])

        logger.debug(
            f"Iteration {iteration}: merged pair {best_pair} "
            f"(adj_p={best_adj_p:.4f}), {len(active)} blocks remain"
        )

        iteration += 1

    if iteration >= max_iter:
        warnings.warn(
            f"Categorical statistical merge reached max iterations ({max_iter})",
            UserWarning,
        )

    result = list(active.values())

    # ------------------------------------------------------------------
    # Phase 2: Enforce min_samples
    # ------------------------------------------------------------------
    if constraints.abs_min_samples > 0:
        result = _enforce_cat_min_samples(result, constraints, history)

    # ------------------------------------------------------------------
    # Phase 3: Enforce min_positives / min_negatives
    # ------------------------------------------------------------------
    if is_binary_y and (
        constraints.abs_min_positives > 0 or constraints.abs_min_negatives > 0
    ):
        result = _enforce_cat_min_class_counts(result, constraints, history)

    if not result:
        raise FittingError("Categorical merging produced zero blocks")

    _validate_cat_merge_result(result, constraints, is_binary_y)

    logger.info(
        f"Categorical merge complete: {len(blocks)} -> {len(result)} blocks"
    )

    return result


__all__ = [
    "CategoryBlock",
    "ChiResult",
    "merge_categorical",
]
