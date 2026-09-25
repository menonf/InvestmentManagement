"""Performance analytics facade.

Kept so existing callers can continue to write::

    from analytics.performance import performance_analytics as perf
    perf.calculate_portfolio_constituent_returns(...)

The implementations now live in :mod:`analytics.portfolio` (weights, returns)
and :mod:`analytics.performance.plots`.
"""

from analytics.portfolio.returns import calculate_portfolio_constituent_returns, merge_returns_with_weights
from analytics.portfolio.weights import calculate_held_shares, calculate_portfolio_constituent_weights

from .plots import plot_cumulative_returns, plot_returns

__all__ = [
    "calculate_portfolio_constituent_returns",
    "calculate_portfolio_constituent_weights",
    "calculate_held_shares",
    "plot_cumulative_returns",
    "plot_returns",
    "merge_returns_with_weights",
]
