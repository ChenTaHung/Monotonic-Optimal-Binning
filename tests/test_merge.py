import numpy as np

from MOBPY.core.merge import merge_adjacent, blocks_from_dicts, Block
from MOBPY.core.constraints import BinningConstraints


def _mk_block(left, right, ys):
    ys = np.asarray(ys, dtype=float)
    return dict(
        left=float(left),
        right=float(right),
        n=int(len(ys)),
        sum=float(ys.sum()),
        sum2=float((ys**2).sum()),
        ymin=float(ys.min()),
        ymax=float(ys.max()),
    )


def test_merge_reduces_violations_by_pvalue():
    # Three tiny adjacent blocks with distinct means; large p-value threshold encourages merging
    rows = [
        _mk_block(0.0, 1.0, [0.2, 0.3, 0.1]),
        _mk_block(1.0, 2.0, [0.25, 0.35, 0.15]),
        _mk_block(2.0, 3.0, [0.8, 0.9, 0.7]),
    ]
    cons = BinningConstraints(max_bins=2, min_bins=1, initial_pvalue=0.5, maximize_bins=True)
    merged = merge_adjacent(rows, constraints=cons, is_binary_y=False)
    assert len(merged) <= 2  # at or below max_bins
    # final right edge of last block should be +inf after materialization in binner; here we just test merging returns Blocks
    assert isinstance(merged[0], Block)


def test_merge_min_samples_sweep():
    # Middle block is undersized; the sweep should merge until bins >= abs_min_samples or hit min_bins
    rows = [
        _mk_block(0.0, 1.0, [0.2]*20),   # n=20
        _mk_block(1.0, 2.0, [0.3]*5),    # n=5 (undersized)
        _mk_block(2.0, 3.0, [0.4]*20),   # n=20
    ]
    cons = BinningConstraints(max_bins=10, min_bins=2, min_samples=10, maximize_bins=False)
    cons.resolve(total_n=45, total_pos=0)
    merged = merge_adjacent(rows, constraints=cons, is_binary_y=False)
    # Either all bins >= 10 or len == min_bins (2)
    all_ok = all(b.n >= cons.abs_min_samples for b in merged)
    assert all_ok or len(merged) == cons.min_bins


def test_merge_accepts_dict_or_block():
    rows = [
        _mk_block(0.0, 1.0, [0.0, 1.0]),
        _mk_block(1.0, 2.0, [1.0, 1.0]),
    ]
    cons = BinningConstraints(max_bins=1, min_bins=1, initial_pvalue=0.0)
    merged_from_dict = merge_adjacent(rows, constraints=cons, is_binary_y=True)
    merged_from_block = merge_adjacent(blocks_from_dicts(rows), constraints=cons, is_binary_y=True)
    assert len(merged_from_dict) == 1
    assert len(merged_from_block) == 1
