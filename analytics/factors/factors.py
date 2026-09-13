"""Factor engine scaffolding for the multi-vendor backtesting engine.

A *factor* turns a standardized price/fundamental panel into a cross-sectional
signal (one score per asset per date) that a backtest can rank and weight.

The base class is intentionally minimal: subclasses implement :meth:`compute`
and return a wide DataFrame indexed by date with asset identifiers as columns.

Standardized input contract (what vendors/loaders produce upstream):
    prices: DataFrame indexed by ``as_of_date`` with ``security_id`` columns,
            values are the chosen price (e.g. ``adj_close``).
    fundamentals (optional): DataFrame or dict of per-security metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class Factor(ABC):
    """Abstract base class for cross-sectional factors."""

    #: Human-readable factor name (used in outputs/debugging).
    name: str = "base"

    @abstractmethod
    def compute(self, prices: pd.DataFrame, fundamentals: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Return a cross-sectional score matrix.

        Args:
            prices: Wide DataFrame, index = date, columns = security_id,
                values = price (or other per-date observation).
            fundamentals: Optional wide/long fundamentals (per-security metrics).

        Returns:
            DataFrame with the same index as ``prices`` and columns = security_id,
            values = factor score (higher = stronger signal). NaN where undefined.
        """
        raise NotImplementedError

    def rank(self, scores: pd.DataFrame, axis: int = 1) -> pd.DataFrame:
        """Convert raw scores to percentile ranks (0..1) cross-sectionally."""
        return scores.rank(axis=axis, pct=True)


class MomentumFactor(Factor):
    """Trailing return over a lookback window (default 12 months, monthly step)."""

    name = "momentum"

    def __init__(self, lookback: int = 252, skip: int = 21):
        """Initialize the momentum factor.

        Args:
            lookback: Trading days of price history to measure momentum over.
            skip: Trading days to skip from the most recent window (avoid reversal).
        """
        self.lookback = lookback
        self.skip = skip

    def compute(self, prices: pd.DataFrame, fundamentals: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Compute the momentum score matrix from prices.

        Args:
            prices: DataFrame indexed by date with ``security_id`` columns.
            fundamentals: Unused; accepted for interface compatibility.

        Returns:
            Wide DataFrame of period-over-period returns, NaN where unavailable.
        """
        # Price [t] / Price [t - lookback - skip] - 1, skipping the most recent
        # ``skip`` days to avoid short-term reversal contamination.
        shifted = prices.shift(self.lookback + self.skip)
        return (prices / shifted - 1.0).where(shifted.notna())


class ValueFactor(Factor):
    """Earnings-yield style value proxy: 1 / trailing price multiple.

    Requires a fundamentals frame with a ``metric_value`` per security (e.g.
    a per-share earnings or book value). Lower price relative to the metric
    scores higher.
    """

    name = "value"

    def __init__(self, metric: str = "earnings_per_share"):
        """Initialize the value factor.

        Args:
            metric: Fundamentals column used as the value metric.
        """
        self.metric = metric

    def compute(self, prices: pd.DataFrame, fundamentals: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Compute the value score matrix from fundamentals.

        Args:
            prices: DataFrame indexed by date with ``security_id`` columns.
            fundamentals: DataFrame indexed by ``security_id`` (or with that column).

        Returns:
            Wide DataFrame of the chosen metric broadcast across dates.
        """
        if fundamentals is None:
            raise ValueError("ValueFactor requires a fundamentals frame.")
        # Expect fundamentals indexed by security_id with a column == self.metric.
        if self.metric not in fundamentals.columns:
            raise ValueError(f"Fundamentals missing metric column '{self.metric}'.")
        metric_series = fundamentals[self.metric]
        # Align metric to the price columns; broadcast across dates.
        aligned = pd.DataFrame(
            {sec: [metric_series.get(sec, pd.NA)] * len(prices.index) for sec in prices.columns},
            index=prices.index,
        )
        with pd.option_context("mode.use_inf_as_na", True):
            yield_ratio = aligned / prices.replace(0, pd.NA)
        return yield_ratio
