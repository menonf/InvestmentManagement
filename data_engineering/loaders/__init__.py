"""Reusable data-loading pipelines (previously inline notebook cells).

- :mod:`.portfolios`          get-or-create portfolios, generate daily holdings
- :mod:`.universe`            build the EOD universe for a backtest from index membership
- :mod:`.eod_loader`          batched / retried EOD price load with sanity guards, mop-up and coverage
- :mod:`.fundamentals_loader` fundamentals load with per-security retry and a point-in-time check
"""

from .eod_loader import (
    EodLoadSummary,
    coverage_report,
    drop_implausible_moves,
    fetch_in_batches,
    find_broken_securities,
    load_eod_prices,
    mop_up_missing_prices,
    prune_low_coverage_holdings,
)
from .fundamentals_loader import check_point_in_time, fundamentals_coverage, load_fundamentals_with_retry
from .portfolios import (
    business_days,
    ensure_security,
    get_or_create_portfolio,
    replace_portfolio_holdings,
    resolve_security_ids_by_ric,
    write_constant_holdings,
    write_index_holdings,
)
from .universe import build_backtest_universe

__all__ = [
    "EodLoadSummary",
    "coverage_report",
    "drop_implausible_moves",
    "fetch_in_batches",
    "find_broken_securities",
    "load_eod_prices",
    "mop_up_missing_prices",
    "prune_low_coverage_holdings",
    "check_point_in_time",
    "fundamentals_coverage",
    "load_fundamentals_with_retry",
    "business_days",
    "ensure_security",
    "get_or_create_portfolio",
    "replace_portfolio_holdings",
    "resolve_security_ids_by_ric",
    "write_constant_holdings",
    "write_index_holdings",
    "build_backtest_universe",
]
