"""Portfolio constituent return calculations.

Pure functions only — no plotting, no I/O. Safe for reuse in backtests and
batch analytics.

Column contract (must match legacy callers / unit tests):
    required: {as_of_date, port_id, security_id, portfolio_short_name, <price_type>}
"""

from __future__ import annotations

import numpy as np
from pandas import DataFrame


def calculate_portfolio_constituent_returns(portfolio_market_data: DataFrame, price_type: str) -> DataFrame:
    """Compute simple and log period returns per constituent.

    Args:
        portfolio_market_data: DataFrame with columns
            ``as_of_date``, ``port_id``, ``security_id``, ``portfolio_short_name``
            and a price column named ``price_type``.
        price_type: Name of the price column (e.g. ``"close"``, ``"adj_close"``).

    Returns:
        Pivot DataFrame indexed by ``as_of_date`` with a
        (portfolio_short_name, security_id) columns MultiIndex and
        ``returns`` / ``log_returns`` levels, sorted on both axis levels.
    """
    required_columns = {"as_of_date", "port_id", "security_id", "portfolio_short_name", price_type}
    missing_cols = required_columns - set(portfolio_market_data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in input data: {missing_cols}")

    # Sort once to avoid sorting in every group
    sorted_data = portfolio_market_data.sort_values(by=["port_id", "security_id", "as_of_date"])

    # Transform keeps the grouping columns in the result across pandas versions.
    grouped_prices = sorted_data.groupby(["port_id", "security_id"])[price_type]
    returns_df = sorted_data.copy()
    returns_df["returns"] = grouped_prices.transform("pct_change")
    # Log returns: guard against zero/negative prices (bad EOD rows with close=0,
    # or a price of 0 in the prior day) which make log(0) -> -inf and raise a
    # RuntimeWarning. Compute under errstate so the 0/negative ratios don't warn,
    # then mask them to NaN (dropped below with the first-row NaNs).
    ratio = grouped_prices.transform(lambda p: p / p.shift(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_r = np.log(ratio)
    returns_df["log_returns"] = np.where(ratio > 0, log_r, np.nan)

    # Drop rows where returns are NaN (typically the first row of each group,
    # or a bad price that yielded a non-finite log return)
    returns_df = returns_df.dropna(subset=["returns", "log_returns"])

    # Pivot to get multi-indexed return matrix
    return_matrix = returns_df.pivot_table(
        index="as_of_date",
        columns=["portfolio_short_name", "security_id"],
        values=["returns", "log_returns"],
    )

    return return_matrix.sort_index(axis=1, level=[0, 1])


def merge_returns_with_weights(portfolio_asset_returns: DataFrame, portfolio_asset_weights: DataFrame) -> tuple[DataFrame, DataFrame]:
    """Align returns and weights, filling missing weights with 0.

    Returns:
        ``(returns, weights)`` with a common (date, portfolio, security) shape.
    """
    returns = portfolio_asset_returns.copy()
    weights = portfolio_asset_weights.reindex(index=returns.index, columns=returns.columns, fill_value=0)
    return returns, weights
