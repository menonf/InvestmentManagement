"""Rebalance schedules derived from the trading calendar in a price panel."""

from __future__ import annotations

import pandas as pd

_PERIOD = {"D": None, "W": "W", "M": "M", "Q": "Q", "A": "Y", "Y": "Y"}


def rebalance_dates(index: pd.Index, freq: str = "M") -> pd.DatetimeIndex:
    """First trading day of each period present in ``index``.

    Args:
        index: trading dates (any order; converted to ``DatetimeIndex``).
        freq: ``"D"`` (every day), ``"W"``, ``"M"`` (default), ``"Q"`` or ``"A"``.
    """
    idx = pd.DatetimeIndex(index).sort_values().unique()
    key = freq.upper()
    if key not in _PERIOD:
        raise ValueError(f"unknown rebalance frequency '{freq}'")
    if _PERIOD[key] is None:
        return pd.DatetimeIndex(idx)
    periods = idx.to_period(_PERIOD[key])
    firsts = pd.Series(idx, index=periods).groupby(level=0).first()
    return pd.DatetimeIndex(firsts.to_numpy())


def month_starts(index: pd.Index) -> pd.DatetimeIndex:
    """First trading day of each month."""
    return rebalance_dates(index, "M")


def quarter_starts(index: pd.Index) -> pd.DatetimeIndex:
    """First trading day of each quarter."""
    return rebalance_dates(index, "Q")
