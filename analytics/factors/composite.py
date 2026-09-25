"""Blend several factors into one signal on a rebalance schedule."""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd
from pandas import DataFrame

from .base import Factor
from .transforms import combine, sample_on


class CompositeFactor(Factor):
    """Weighted blend of component signals, refreshed on rebalance dates.

    Components can be :class:`Factor` instances (computed from ``prices`` /
    ``fundamentals`` at call time) or pre-computed date x security frames,
    which is how a point-in-time ML signal is passed in.

    Args:
        components: ``name -> Factor | DataFrame``.
        weights: ``name -> weight`` (default equal).
        method: ``"zscore"`` or ``"rank"`` normalisation before blending.
        rebalance_dates: dates on which the composite is recomputed; between
            them it is forward-filled. ``None`` = every date.
    """

    name = "composite"

    def __init__(
        self,
        components: dict[str, Union[Factor, DataFrame]],
        weights: Optional[dict[str, float]] = None,
        method: str = "zscore",
        rebalance_dates: Optional[pd.DatetimeIndex] = None,
    ):
        """Store the blend definition."""
        if not components:
            raise ValueError("CompositeFactor needs at least one component")
        self.components = components
        self.weights = weights
        self.method = method
        self.rebalance_dates = rebalance_dates

    def compute(self, prices: DataFrame, fundamentals: Optional[DataFrame] = None) -> DataFrame:
        """Compute / align every component, blend, then apply the rebalance schedule."""
        signals: dict[str, DataFrame] = {}
        for name, comp in self.components.items():
            sig = comp.compute(prices, fundamentals) if isinstance(comp, Factor) else comp
            signals[name] = sig.reindex(index=prices.index, columns=prices.columns)
        blended = combine(signals, self.weights, method=self.method)
        if self.rebalance_dates is not None:
            blended = sample_on(blended, pd.DatetimeIndex(self.rebalance_dates))
        return blended
