"""Backtest engine: weights x returns -> portfolio returns, plus performance and signal metrics.

from analytics.backtest import run_backtest, prices_to_returns, summary_table

result = run_backtest(weights, prices_to_returns(prices), shift_weights=1, cost_bps=10)
result.summary()
"""

from .engine import (
    align_frames,
    portfolio_returns,
    portfolio_returns_from_constituents,
    prices_to_returns,
    reconstruction_backtest,
    run_backtest,
    to_datetime_index,
)
from .metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    cumulative_returns,
    forward_returns,
    hit_rate,
    ic_summary,
    information_coefficient,
    max_drawdown,
    quantile_returns,
    sharpe_ratio,
    sortino_ratio,
    summary_table,
    turnover,
)
from .result import BacktestResult

__all__ = [
    "run_backtest",
    "portfolio_returns",
    "portfolio_returns_from_constituents",
    "reconstruction_backtest",
    "prices_to_returns",
    "align_frames",
    "to_datetime_index",
    "BacktestResult",
    "cumulative_returns",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "hit_rate",
    "turnover",
    "summary_table",
    "forward_returns",
    "information_coefficient",
    "ic_summary",
    "quantile_returns",
]
