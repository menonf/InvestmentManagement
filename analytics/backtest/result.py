"""Container for backtest outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pandas import DataFrame, Series

from .metrics import cumulative_returns, summary_table, turnover


@dataclass
class BacktestResult:
    """Daily results of one or more portfolios.

    Attributes:
        returns: date x portfolio simple returns (net of costs when modelled).
        weights: date x security weights per portfolio (``{name: frame}``);
            empty for reconstruction-style runs that pass returns directly.
        gross_returns: returns before transaction costs (same as ``returns``
            when no cost model was applied).
        costs: date x portfolio cost drag.
    """

    returns: DataFrame
    weights: dict[str, DataFrame] = field(default_factory=dict)
    gross_returns: Optional[DataFrame] = None
    costs: Optional[DataFrame] = None

    def cumulative(self) -> DataFrame:
        """Compounded cumulative returns per portfolio."""
        return cumulative_returns(self.returns)

    def growth_of_one(self) -> DataFrame:
        """Value of 1 unit invested at the start."""
        return 1.0 + self.cumulative()

    def turnover(self) -> DataFrame:
        """One-way turnover per date per portfolio with weights."""
        return DataFrame({name: turnover(w) for name, w in self.weights.items()})

    def summary(self, benchmark: Optional[str] = None) -> DataFrame:
        """Return performance statistics table (see :func:`~analytics.backtest.metrics.summary_table`)."""
        table = summary_table(self.returns, benchmark=benchmark)
        if self.weights:
            to = self.turnover()
            table["annual_turnover"] = Series({n: float(to[n].mean() * 252) for n in to.columns})
        return table
