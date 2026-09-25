"""Transparent (non-ML) value factor built from the fundamentals ratio panel."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from pandas import DataFrame, Series

from data_engineering.fundamentals.ratios import VALUE_MULTIPLES_LOWER_IS_BETTER

from .base import PanelFactor
from .transforms import winsorize, zscore

#: Default cheapness measures: price multiples where lower is cheaper.
DEFAULT_VALUE_METRICS: tuple[str, ...] = tuple(VALUE_MULTIPLES_LOWER_IS_BETTER)


class ValueFactor(PanelFactor):
    """Composite "cheapness" score: mean z-score of inverted price multiples.

    Each multiple in ``metrics`` (``P/E``, ``P/B``, ``P/S``, ``EV/EBIT`` by
    default) is inverted into a yield (``E/P`` etc.), non-positive multiples are
    treated as missing (negative earnings are not "cheap"), yields are
    winsorised and z-scored across the cross-section, and the available
    component scores are averaged. Higher = cheaper.

    This gives a readable baseline to compare against the ML value factor.
    """

    name = "value"

    def __init__(
        self,
        metrics: Sequence[str] = DEFAULT_VALUE_METRICS,
        winsor: Optional[tuple[float, float]] = (0.01, 0.99),
        min_components: int = 1,
        fundamentals: Optional[DataFrame] = None,
    ):
        """Configure which multiples to use and how to clean them."""
        super().__init__(fundamentals)
        self.metrics = tuple(metrics)
        self.winsor = winsor
        self.min_components = min_components

    def score_panel(self, panel: DataFrame) -> Series:
        """Return the mean z-scored yield per security (NaN if too few components)."""
        comps = []
        for m in self.metrics:
            if m not in panel.columns:
                continue
            mult = panel[m].astype(float)
            yld = (1.0 / mult).where(mult > 0)
            if self.winsor is not None and yld.notna().sum() > 2:
                yld = winsorize(yld, *self.winsor)
            comps.append(zscore(yld))
        if not comps:
            return Series(np.nan, index=panel.index, dtype=float)
        stacked = DataFrame(comps).T
        score = stacked.mean(axis=1)
        return score.where(stacked.notna().sum(axis=1) >= self.min_components)
