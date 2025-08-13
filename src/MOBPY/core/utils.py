from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def ensure_numeric_series(s: pd.Series, name: str) -> None:
    """Validate that a pandas Series is numeric and finite.

    Args:
        s: Series to validate.
        name: Human-readable name for error messages.

    Raises:
        TypeError: If dtype is not numeric.
        ValueError: If values contain non-finite numbers (NaN/inf).
    """
    if not pd.api.types.is_numeric_dtype(s):
        raise TypeError(f"{name!r} must be numeric.")
    if not np.isfinite(s.dropna().to_numpy()).all():
        raise ValueError(f"{name!r} contains non-finite values.")


def is_binary_series(s: pd.Series) -> bool:
    """Return True if Series looks binary {0,1} (ignoring NaN)."""
    vals = pd.Series(s.dropna().unique())
    # tolerate e.g., int/float dtypes; treat exactly {0,1}
    try:
        as_int = vals.astype(int).tolist()
    except Exception:
        return False
    return len(vals) <= 2 and set(as_int).issubset({0, 1})


def woe_iv(
    goods: np.ndarray,
    bads: np.ndarray,
    smoothing: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute WoE and IV per bin with simple additive smoothing.

    Args:
        goods: Number of goods per bin.
        bads: Number of bads per bin.
        smoothing: Additive smoothing to avoid log(0).

    Returns:
        Tuple of (woe, iv_grp) arrays.
    """
    goods = goods.astype(float)
    bads = bads.astype(float)
    G = goods.sum()
    B = bads.sum()

    # smoothed shares
    g = (goods + smoothing) / (G + smoothing * (goods.size > 0))
    b = (bads + smoothing) / (B + smoothing * (bads.size > 0))

    w = np.log(np.clip(g / b, 1e-12, np.inf))
    iv = (g - b) * w
    return w, iv


@dataclass(frozen=True)
class Parts:
    """Partitioned dataset by the x-column condition."""
    clean: pd.DataFrame
    missing: pd.DataFrame
    excluded: pd.DataFrame


def partition_df(df: pd.DataFrame, x: str, exclude_values: Iterable | None) -> Parts:
    """Split df into clean/missing/excluded w.r.t. x.

    Rules:
        * Missing: rows where x is NaN.
        * Excluded: rows where x is non-missing and in `exclude_values`.
        * Clean: rows where x is non-missing and not excluded.

    Args:
        df: Input DataFrame.
        x: Column to check.
        exclude_values: Values to exclude from clean buckets.

    Returns:
        Parts(clean, missing, excluded)
    """
    missing = df[df[x].isna()]
    if exclude_values is None:
        excluded = df.iloc[0:0]
        clean = df[df[x].notna()]
    else:
        ex = set(exclude_values)
        mask_ex = df[x].notna() & df[x].isin(ex)
        excluded = df[mask_ex]
        clean = df[df[x].notna() & ~df[x].isin(ex)]
    return Parts(clean=clean, missing=missing, excluded=excluded)
