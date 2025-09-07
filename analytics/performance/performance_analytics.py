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
    if portfolio_specific_weights is None:
        portfolio_specific_weights = {}

    # Get all unique portfolios in the data
    unique_portfolios = portfolio_market_data["portfolio_short_name"].unique()

    # Determine required columns based on all weight types that will be used
    all_weight_types = {weightage_type}
    for portfolio, weight_spec in portfolio_specific_weights.items():
        if isinstance(weight_spec, list):
            all_weight_types.update(weight_spec)
        else:
            all_weight_types.add(weight_spec)

    required_columns = {"as_of_date", "portfolio_short_name", "security_id", "portfolio_type", price_type}
    if "market_weighted" in all_weight_types:
        required_columns.add("shares_outstanding")

    missing_cols = required_columns - set(portfolio_market_data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = portfolio_market_data.copy()

    # Compute market_value for Benchmark portfolios
    if "market_cap" in df.columns:
        df.loc[df["portfolio_type"] == "Benchmark", "market_value"] = df["market_cap"]
    else:
        df.loc[df["portfolio_type"] == "Benchmark", "market_value"] = df["shares_outstanding"].shift(1) * df[price_type].shift(1)

    # Compute held_shares using user-defined function
    df = df.groupby(["as_of_date", "portfolio_short_name"], group_keys=False).apply(calculate_held_shares)

    # Pivot held_shares and price
    pivoted = df.pivot_table(
        index="as_of_date", columns=["portfolio_short_name", "security_id"], values=["held_shares", price_type]
    ).fillna(0)

    # Reorder and sort columns
    pivoted = pivoted.sort_index(axis=1, level=[0, 1])
    held_shares = pivoted["held_shares"]
    prices = pivoted[price_type]

    # Initialize weights DataFrame with the same structure
    weights = pd.DataFrame(index=held_shares.index, columns=held_shares.columns).fillna(0.0)

    # Process each portfolio with its specific weight type
    for portfolio in unique_portfolios:
        # Determine weight type for this portfolio
        if portfolio in portfolio_specific_weights:
            portfolio_weight_type = portfolio_specific_weights[portfolio]
            # If it's a list, use the first one (you could add logic to validate and pick the best one)
            if isinstance(portfolio_weight_type, list):
                portfolio_weight_type = portfolio_weight_type[0]
        else:
            portfolio_weight_type = weightage_type

        # Get columns for this portfolio
        portfolio_cols = [col for col in held_shares.columns if col[0] == portfolio]

        if not portfolio_cols:
            continue

        portfolio_held_shares = held_shares[portfolio_cols]
        portfolio_prices = prices[portfolio_cols]

        # Calculate weights based on the specific weight type
        if portfolio_weight_type == "equal_weighted":
            nonzero_counts = portfolio_held_shares.apply(lambda row: (row != 0).sum(), axis=1)
            portfolio_weights = portfolio_held_shares.apply(lambda row: (row != 0).astype(int), axis=1)
            portfolio_weights = portfolio_weights.div(nonzero_counts, axis=0).fillna(0)

        elif portfolio_weight_type == "market_weighted":
            market_value = portfolio_held_shares * portfolio_prices.shift(1)
            portfolio_totals = market_value.sum(axis=1)
            portfolio_weights = market_value.div(portfolio_totals, axis=0).fillna(0)

        elif portfolio_weight_type == "price_weighted":
            portfolio_totals = portfolio_prices.sum(axis=1)
            portfolio_weights = portfolio_prices.div(portfolio_totals, axis=0).fillna(0)

        else:
            raise ValueError(f"Unsupported weightage_type: {portfolio_weight_type} for portfolio: {portfolio}")

        # Assign the calculated weights back to the main weights DataFrame
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
                                   Values should be periodic (log) returns.
        start_at_zero (bool): If True, cumulative returns will start at 0.

    Returns:
        None

    Example:
        >>> plot_cumulative_returns(log_returns_df)
    """
    # --- Step 1: Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Step 2: Loop over each portfolio or asset column
    for column in asset_returns.columns:
        cumulative_returns = np.exp(asset_returns[column].cumsum()) - 1
        if start_at_zero:
            cumulative_returns = cumulative_returns - cumulative_returns.iloc[0]

        cumulative_returns *= 100

        # Plot the series
        ax.plot(asset_returns.index, cumulative_returns, label=str(column), linewidth=2)

    # --- Step 3: Style plot
    ax.set_title("Cumulative Returns Over Time", fontsize=14)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Returns (%)", fontsize=12)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # --- Step 4: Show plot
    plt.tight_layout()
    plt.show()
