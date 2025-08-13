import numpy as np
import pandas as pd
import pytest

from MOBPY.core.pava import PAVA


def test_pava_monotone_strict_plateau():
    # x has three unique values; y means (2, 2, 3) -> with strict=True, first two merge
    df = pd.DataFrame({"x": [0]*5 + [1]*5 + [2]*5, "y": [2]*5 + [2]*5 + [3]*5})
    p = PAVA(df=df, x="x", y="y", strict=True).fit()
    blocks = p.export_blocks(as_dict=True)
    # expect two blocks: [0,2) mean ~2, [2, +inf) mean ~3
    assert len(blocks) == 2
    assert blocks[0]["left"] == 0.0 and np.isfinite(blocks[0]["right"])
    assert np.isposinf(blocks[1]["right"])
    # monotone increasing
    m0 = blocks[0]["sum"] / blocks[0]["n"]
    m1 = blocks[1]["sum"] / blocks[1]["n"]
    assert m0 < m1


def test_pava_auto_sign_inference():
    # decreasing means -> expect sign '-'
    df = pd.DataFrame({"x": [0]*5 + [1]*5 + [2]*5, "y": [3]*5 + [2]*5 + [1]*5})
    p = PAVA(df=df, x="x", y="y", sign="auto").fit()
    assert p.resolved_sign_ == "-"


def test_pava_input_validation():
    # non-numeric y must raise TypeError/ValueError in ensure_numeric_series
    df = pd.DataFrame({"x": [0, 1, 2], "y": ["a", "b", "c"]})
    with pytest.raises((TypeError, ValueError)):
        PAVA(df=df, x="x", y="y").fit()

    # missing x/y => drop; if all drop -> ValueError
    df2 = pd.DataFrame({"x": [np.nan, np.nan], "y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="No rows with non-missing x and y"):
        PAVA(df=df2, x="x", y="y").fit()
