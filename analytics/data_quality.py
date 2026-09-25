"""Price-panel data-quality helpers (vectorised versions of the notebook loops)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame


def find_single_day_glitches(prices: DataFrame, threshold: float = 0.5) -> DataFrame:
    """Boolean mask of one-day prints that move > ``threshold`` and revert the next session.

    A genuine crash or split leaves the price at the new level; a bad print
    craters (or spikes) for one day and comes straight back. The mask marks
    the glitch day only.
    """
    r = prices.pct_change(fill_method=None)
    big = r.abs() > threshold
    next_r = r.shift(-1)
    # a down glitch (r <= -thr) reverts with a large up move; an up glitch reverts down
    reverts = ((r < 0) & (next_r > threshold)) | ((r > 0) & (next_r < -threshold / (1 + threshold)))
    return (big & reverts).fillna(False)


def neutralize_single_day_glitches(prices: DataFrame, threshold: float = 0.5) -> tuple[DataFrame, int]:
    """Replace single-day glitch prints with the previous price (forward fill).

    Nothing is deleted from the source; the cleaned copy keeps every genuine
    price. Returns ``(clean_prices, n_glitches)``.
    """
    mask = find_single_day_glitches(prices, threshold)
    clean = prices.mask(mask).ffill()
    return clean, int(mask.to_numpy().sum())


def coverage_by_date(prices: DataFrame) -> pd.Series:
    """Share of columns with a non-NaN price on each date."""
    return prices.notna().mean(axis=1)


def drop_sparse_securities(prices: DataFrame, min_obs: int) -> DataFrame:
    """Drop securities with fewer than ``min_obs`` priced days."""
    keep = prices.notna().sum() >= min_obs
    return prices.loc[:, keep]


def zero_or_negative_prices(prices: DataFrame) -> pd.Series:
    """Count of non-positive prices per security (log returns are undefined there)."""
    return (prices <= 0).sum()


def implausible_return_mask(prices: DataFrame, threshold: float = 0.25) -> DataFrame:
    """Boolean mask of day-over-day moves above ``threshold`` (no reversion test)."""
    return (prices.pct_change(fill_method=None).abs() > threshold).fillna(False)


def describe_panel(prices: DataFrame) -> dict[str, float]:
    """Quick health summary of a price panel."""
    cov = coverage_by_date(prices)
    return {
        "dates": int(len(prices.index)),
        "securities": int(prices.shape[1]),
        "mean_coverage": float(cov.mean()) if len(cov) else np.nan,
        "min_coverage": float(cov.min()) if len(cov) else np.nan,
        "non_positive_prices": int((prices <= 0).sum().sum()),
        "single_day_glitches": int(find_single_day_glitches(prices).to_numpy().sum()),
    }
