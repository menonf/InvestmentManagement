"""Investment performance analytics.

This module is a thin facade over the focused submodules:
    - returns.py  : constituent return calculations
    - weights.py  : constituent weight calculations
    - plots.py    : matplotlib plotting helpers

All public function names are re-exported here so legacy callers
(``from analytics.performance import performance_analytics as perf``) continue
to work unchanged.
"""

from .plots import (
    plot_cumulative_returns,
    plot_returns,
)
from .returns import (
    calculate_portfolio_constituent_returns,
    merge_returns_with_weights,
)
from .weights import (
    calculate_held_shares,
    calculate_portfolio_constituent_weights,
)

__all__ = [
    "calculate_portfolio_constituent_returns",
    "calculate_portfolio_constituent_weights",
    "calculate_held_shares",
    "plot_cumulative_returns",
    "plot_returns",
    "merge_returns_with_weights",
]
