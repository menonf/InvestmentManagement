"""Minimal example: backtest stored portfolios and print a performance summary.

Usage:
    python toolkit/scripts/portfolio_data_load.py --portfolios SP500 SPY --start 2025-01-01 --end 2025-06-30
"""

from __future__ import annotations

import argparse

from analytics.backtest import reconstruction_backtest
from analytics.portfolio import calculate_portfolio_constituent_returns, calculate_portfolio_constituent_weights
from data_engineering.database import database


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolios", nargs="+", default=["SP500", "SPY"])
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-06-30")
    parser.add_argument("--weighting", default="market_weighted", choices=["market_weighted", "equal_weighted", "price_weighted"])
    args = parser.parse_args(argv)

    engine, connection, _conn_str, session = database.get_db_connection()
    try:
        market_data = database.get_portfolio_market_data(session, engine, args.start, args.end, args.portfolios)
    finally:
        session.close()
        connection.close()
    returns = calculate_portfolio_constituent_returns(market_data, "adj_close")["returns"]
    weights = calculate_portfolio_constituent_weights(market_data, "adj_close", args.weighting)
    result = reconstruction_backtest(returns, weights)
    print(result.summary().round(4).to_string())


if __name__ == "__main__":
    main()
