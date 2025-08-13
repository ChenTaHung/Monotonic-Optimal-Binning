import pytest

from MOBPY.core.constraints import BinningConstraints


def test_constraints_resolve_happy():
    cons = BinningConstraints(
        max_bins=6, min_bins=4,
        max_samples=0.5,    # fraction of total_n
        min_samples=0.1,    # fraction of total_n
        min_positives=0.2,  # fraction of total_pos
        initial_pvalue=0.4,
    )
    cons.resolve(total_n=1_000, total_pos=200)
    assert cons.abs_min_samples == 100
    # abs_max_samples is bounded above by total_n
    assert cons.abs_max_samples == 500
    # 0.2 * 200 = 40
    assert cons.abs_min_positives == 40
    assert cons._resolved is True


def test_constraints_resolve_cross_error():
    # min_samples > max_samples after resolution -> raise
    cons = BinningConstraints(max_bins=6, min_bins=4, max_samples=0.1, min_samples=0.2)
    with pytest.raises(ValueError, match="min_samples.*exceed"):
        cons.resolve(total_n=10_000, total_pos=0)


def test_constraints_sanity_on_bins():
    # min_bins > max_bins with maximize_bins=True should raise
    cons = BinningConstraints(max_bins=3, min_bins=4, maximize_bins=True)
    with pytest.raises(ValueError, match="min_bins cannot exceed max_bins"):
        cons.resolve(total_n=100, total_pos=0)
