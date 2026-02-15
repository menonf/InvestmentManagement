"""Module for investment performance analytics."""

from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame


def calculate_portfolio_constituent_returns(portfolio_market_data: DataFrame, price_type: str) -> DataFrame:
    """
    Calculate daily simple and logarithmic returns for each asset within each portfolio.

    Params:
        portfolio_market_data: DataFrame containing market price data with columns
                               ['as_of_date', 'port_id', 'security_id', 'portfolio_short_name', price_type].
        price_type: Column name to use for price (e.g., 'close', 'adj_close').

    Returns:
        DataFrame: MultiIndex (as_of_date x [portfolio_short_name, security_id]) DataFrame
                   with 'returns' and 'log_returns'.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = {"as_of_date", "port_id", "security_id", "portfolio_short_name", price_type}
    missing_cols = required_columns - set(portfolio_market_data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in input data: {missing_cols}")

    # Sort once to avoid sorting in every group
    sorted_data = portfolio_market_data.sort_values(by=["port_id", "security_id", "as_of_date"])

    # Group and compute returns more efficiently
    def compute_returns(group: DataFrame) -> DataFrame:
        group["returns"] = group[price_type].pct_change()
        group["log_returns"] = np.log(group[price_type] / group[price_type].shift(1))
        return group

    returns_df = sorted_data.groupby(["port_id", "security_id"], group_keys=False).apply(compute_returns)

    # Drop rows where returns are NaN (typically the first row of each group)
    returns_df = returns_df.dropna(subset=["returns", "log_returns"])

    # Pivot to get multi-indexed return matrix
    return_matrix = returns_df.pivot_table(
        index="as_of_date", columns=["portfolio_short_name", "security_id"], values=["returns", "log_returns"]
    )

    return return_matrix.sort_index(axis=1, level=[0, 1])


def calculate_portfolio_constituent_weights(
    portfolio_market_data: DataFrame,
    price_type: str,
    weightage_type: str = "equal_weighted",
    portfolio_specific_weights: Optional[Dict[str, Union[str, List[str]]]] = None,
) -> DataFrame:
    """
    Calculate asset weights in each portfolio for a given weightage methodology.

    Params:
        portfolio_market_data: DataFrame containing price and quantity data of assets.
        price_type: Column name for price (e.g. 'close', 'adj_close').
        weightage_type: Default weightage type for all portfolios. One of ['equal_weighted', 'market_weighted', 'price_weighted'].
        portfolio_specific_weights: Dict mapping portfolio_short_name(s) to specific weightage types.
                                  Can be: {'portfolio_name': 'weight_type'} or {'portfolio_name': ['weight_type1', 'weight_type2']}
                                  If a list is provided for a portfolio, the first valid weight type will be used.

    Returns:
        DataFrame: MultiIndex (as_of_date x [portfolio_short_name, security_id]) weights.

    """
    portfolio_specific_weights = portfolio_specific_weights or {}
    df = portfolio_market_data.copy()

    # ---------------------------------------------------
    # 1. Compute market_value & held shares (for benchmark indices reconstruction only)
    # ---------------------------------------------------
    if "market_cap" in df.columns:
        df.loc[df["portfolio_type"] == "Benchmark", "market_value"] = df["market_cap"]
    else:
        df.loc[df["portfolio_type"] == "Benchmark", "market_value"] = df["shares_outstanding"] * df[price_type]

    df = df.groupby(["as_of_date", "portfolio_short_name"], group_keys=False).apply(calculate_held_shares)

    # ---------------------------------------------------
    # 2. Pivot held_shares and prices
    # ---------------------------------------------------
    pivot = (
        df.pivot_table(
            index="as_of_date",
            columns=["portfolio_short_name", "security_id"],
            values=["held_shares", price_type],
        )
        .fillna(0)
        .sort_index(axis=1, level=[0, 1])
    )

    held_shares = pivot["held_shares"]
    prices = pivot[price_type]

    weights = pd.DataFrame(
        index=held_shares.index,
        columns=held_shares.columns,
        dtype=float,
    ).fillna(0.0)

    # ---------------------------------------------------
    # 3. Process each portfolio independently
    # ---------------------------------------------------
    for portfolio in df["portfolio_short_name"].unique():

        # Determine weight method for this portfolio
        weight_method = portfolio_specific_weights.get(portfolio, weightage_type)
        if isinstance(weight_method, list):
            weight_method = weight_method[0]

        portfolio_cols = [c for c in held_shares.columns if c[0] == portfolio]
        if not portfolio_cols:
            continue

        shares = held_shares[portfolio_cols]
        px = prices[portfolio_cols]

        # -------- Weight calculations --------
        if weight_method == "equal_weighted":
            active = (shares != 0).astype(int)
            counts = active.sum(axis=1)
            portfolio_weights = active.div(counts, axis=0).fillna(0)

        elif weight_method == "market_weighted":
            market_value = shares * px.shift(1)
            totals = market_value.sum(axis=1)
            portfolio_weights = market_value.div(totals, axis=0).fillna(0)

        elif weight_method == "price_weighted":
            totals = px.sum(axis=1)
            portfolio_weights = px.div(totals, axis=0).fillna(0)

        else:
            raise ValueError(f"Unsupported weightage_type '{weight_method}' " f"for portfolio '{portfolio}'")

        weights[portfolio_cols] = portfolio_weights

    return weights.sort_index(axis=1, level=[0, 1])


def calculate_held_shares(group_df: DataFrame) -> DataFrame:
    """
    Calculate held shares for benchmark portfolios based on market value.

    For rows where portfolio_type is 'Benchmark', held_shares is computed as:
        market_value / total_market_value (i.e., weight of each security in the benchmark).

    Parameters:
        group_df (DataFrame): A DataFrame representing a single group (e.g., grouped by
                              as_of_date and portfolio_short_name).

    Returns:
        DataFrame: Same DataFrame with an additional 'held_shares' column added
                   (only for benchmark portfolios; unchanged otherwise).

    Example:
        >>> df.groupby(["as_of_date", "portfolio_short_name"]).apply(calculate_held_shares)
    """
    # Work on a copy to avoid modifying original data
    df = group_df.copy()

    if (df["portfolio_type"] == "Benchmark").all():
        total_market_value = df["market_value"].sum()

        if total_market_value == 0:
            df["held_shares"] = 0.0  # Prevent division by zero
        else:
            df["held_shares"] = df["market_value"] / total_market_value
    else:
        # Optionally initialize held_shares as NaN or leave untouched
        df["held_shares"] = df.get("held_shares", pd.NA)

    return df


def plot_cumulative_returns(asset_returns: DataFrame, start_at_zero: bool = True) -> None:
    """
    Plot cumulative returns over time for each portfolio in the given returns DataFrame.

    Parameters:
        asset_returns (DataFrame): MultiIndex or wide-format DataFrame with time index
                                   and portfolios (or asset identifiers) as columns.
                                   Values should be periodic (log returns).
        start_at_zero (bool): If True, cumulative returns will start at 0.

    Returns:
        None
    """
    import matplotlib.dates as mdates
    import mplcursors

    # Ensure datetime index
    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        asset_returns = asset_returns.copy()
        asset_returns.index = pd.to_datetime(asset_returns.index)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each series
    for column in asset_returns.columns:
        cumulative_returns = np.exp(asset_returns[column].cumsum()) - 1
        if start_at_zero:
            cumulative_returns = cumulative_returns - cumulative_returns.iloc[0]
        cumulative_returns *= 100
        ax.plot(asset_returns.index, cumulative_returns, label=str(column), linewidth=2)

    # Style plot
    ax.set_title("Cumulative Returns Over Time", fontsize=14)
    ax.set_xlabel("")  # no x-axis label
    ax.set_ylabel("Cumulative Returns (%)", fontsize=12)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Show monthly x-axis labels
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", which="major", labelrotation=45, labelsize=9)
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))  # optional minor ticks

    fig.tight_layout()
    fig.autofmt_xdate()

    # Interactive hover: show date + cumulative return
    cursor = mplcursors.cursor(ax, hover=True)

    @cursor.connect("add")
    def on_add(sel) -> None:  # type: ignore
        x, y = sel.target
        date = mdates.num2date(x).strftime("%Y-%m-%d")
        sel.annotation.set_text(f"{sel.artist.get_label()}\n" f"Date: {date}\n" f"Cumulative Return: {y:.2f}%")

    plt.show()


def plot_returns(asset_returns: DataFrame) -> None:
    """
    Plot periodic (non-cumulative) returns over time for each portfolio in the given returns DataFrame.

    Parameters:
        asset_returns (DataFrame): DataFrame with time index and portfolios
                                   (or asset identifiers) as columns.
                                   Values should be periodic (log or simple) returns.

    Returns:
        None
    """
    import matplotlib.dates as mdates
    import mplcursors

    # Ensure datetime index
    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        asset_returns = asset_returns.copy()
        asset_returns.index = pd.to_datetime(asset_returns.index)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each series
    for column in asset_returns.columns:
        returns_pct = asset_returns[column] * 100
        ax.plot(asset_returns.index, returns_pct, label=str(column), linewidth=1.5)

    # Style plot
    ax.set_title("Periodic Returns Over Time", fontsize=14)
    ax.set_xlabel("")  # no x-axis label
    ax.set_ylabel("Returns (%)", fontsize=12)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Hide x-axis ticks & labels entirely
    ax.xaxis.set_visible(False)

    fig.tight_layout()

    # Interactive hover: show date + value
    cursor = mplcursors.cursor(ax, hover=True)

    @cursor.connect("add")
    def on_add(sel) -> None:  # type: ignore
        x, y = sel.target
        date = mdates.num2date(x).strftime("%Y-%m-%d")
        sel.annotation.set_text(f"{sel.artist.get_label()}\n" f"Date: {date}\n" f"Return: {y:.2f}%")

    plt.show()
