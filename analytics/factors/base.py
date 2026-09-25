"""Factor contracts.

A *factor* turns standardised inputs into a cross-sectional signal: one score
per security per date, higher = stronger. Two flavours exist:

- :class:`Factor` - computes from the price panel alone (e.g. momentum).
- :class:`PanelFactor` - scores a *fundamentals panel* (``security_id`` x
  ratios) and broadcasts / refreshes it across dates (value, ML value).

Input contract (produced by ``data_engineering``):
    prices        wide DataFrame, index = date, columns = security_id, values = adj_close
    fundamentals  wide DataFrame, index = security_id, columns = RATIO_COLUMNS
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


class Factor(ABC):
    """Abstract base class for cross-sectional factors."""

    #: Human-readable factor name (used in outputs and the factor_scores table).
    name: str = "base"

    @abstractmethod
    def compute(self, prices: DataFrame, fundamentals: Optional[DataFrame] = None) -> DataFrame:
        """Return a score matrix aligned to ``prices`` (NaN where undefined)."""
        raise NotImplementedError

    @staticmethod
    def rank(scores: DataFrame, axis: int = 1) -> DataFrame:
        """Convert raw scores to cross-sectional percentile ranks in ``(0, 1]``."""
        return scores.rank(axis=axis, pct=True)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({params})"


class PanelFactor(Factor):
    """A factor that scores one fundamentals panel at a time.

    Subclasses implement :meth:`score_panel`. The base class provides the two
    ways of turning panel scores into a date x security matrix:

    * :meth:`compute` - one static panel broadcast across every price date
      (the score is constant over the holding period);
    * :meth:`compute_dynamic` - a *different*, point-in-time panel on each
      rebalance date, forward-filled in between. This is the no-look-ahead
      path used by the backtests.
    """

    def __init__(self, fundamentals: Optional[DataFrame] = None):
        """Optionally attach a static fundamentals panel for single-argument ``compute``."""
        self._fundamentals = fundamentals

    @abstractmethod
    def score_panel(self, panel: DataFrame) -> Series:
        """Score one cross-section: ``security_id -> score`` (NaN allowed)."""
        raise NotImplementedError

    def compute(self, prices: DataFrame, fundamentals: Optional[DataFrame] = None) -> DataFrame:
        """Score a single panel and broadcast it across all ``prices`` dates."""
        panel = fundamentals if fundamentals is not None else self._fundamentals
        if panel is None or panel.empty:
            return DataFrame(np.nan, index=prices.index, columns=prices.columns, dtype=float)
        scores = self.score_panel(panel).reindex(prices.columns).to_numpy(dtype=float)
        return DataFrame(np.tile(scores, (len(prices.index), 1)), index=prices.index, columns=prices.columns)

    def compute_dynamic(
        self,
        prices: DataFrame,
        fundamentals_fn: Callable[[Any], Optional[DataFrame]],
        dates: Optional[Iterable[Any]] = None,
        forward_fill: bool = True,
    ) -> DataFrame:
        """Score a point-in-time panel on each date in ``dates`` and forward-fill.

        Args:
            prices: date x security_id panel defining the output shape.
            fundamentals_fn: ``f(date) -> panel`` returning only fundamentals
                *available* on that date (point-in-time). This is what keeps a
                2020 rebalance from seeing a 2025 filing.
            dates: rebalance dates on which to re-score (default: every price
                date, which is exact but slow; monthly rebalance dates are
                usually sufficient because signals are only read there).
            forward_fill: carry each score forward until the next re-score.
        """
        score_dates = pd.DatetimeIndex(prices.index if dates is None else dates).intersection(pd.DatetimeIndex(prices.index))
        out = DataFrame(np.nan, index=prices.index, columns=prices.columns, dtype=float)
        for dt in score_dates:
            panel = fundamentals_fn(dt)
            if panel is None or len(panel) == 0:
                continue
            out.loc[dt] = self.score_panel(panel).reindex(prices.columns).to_numpy(dtype=float)
        return out.ffill() if forward_fill else out
