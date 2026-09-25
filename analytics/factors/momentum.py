"""Price momentum factors."""

from __future__ import annotations

from typing import Optional

import numpy as np
from pandas import DataFrame

from .base import Factor


class MomentumFactor(Factor):
    """Trailing total return over ``lookback`` days, skipping the most recent ``skip`` days.

    The classic "12-1" momentum (Jegadeesh & Titman) is ``lookback=252,
    skip=21``: the return from t-273 to t-21. Skipping the latest month avoids
    the short-term reversal effect.

    Args:
        lookback: trading days in the measurement window.
        skip: most-recent trading days excluded from the window.
        vol_scale: divide the return by its trailing volatility (``vol_window``
            days of daily returns ending at t-skip). Risk-managed momentum
            (Barroso & Santa-Clara, 2015) reduces momentum crashes.
        vol_window: window for the volatility estimate when ``vol_scale``.
    """

    name = "momentum"

    def __init__(self, lookback: int = 252, skip: int = 21, vol_scale: bool = False, vol_window: int = 63):
        """Store the window parameters."""
        if lookback <= 0 or skip < 0:
            raise ValueError("lookback must be > 0 and skip >= 0")
        self.lookback = lookback
        self.skip = skip
        self.vol_scale = vol_scale
        self.vol_window = vol_window

    def compute(self, prices: DataFrame, fundamentals: Optional[DataFrame] = None) -> DataFrame:
        """Return ``P[t-skip] / P[t-skip-lookback] - 1`` (optionally vol-scaled)."""
        end = prices.shift(self.skip)
        start = prices.shift(self.skip + self.lookback)
        mom = (end / start - 1.0).where(start.notna() & end.notna())
        if self.vol_scale:
            daily = prices.pct_change()
            vol = daily.rolling(self.vol_window, min_periods=max(5, self.vol_window // 2)).std().shift(self.skip)
            mom = mom / vol.replace(0, np.nan)
        return mom
