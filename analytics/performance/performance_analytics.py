"""Module for investment performance analytics."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame


def calculate_portfolio_asset_returns(portfolio_market_data: DataFrame, price_type: str) -> DataFrame:
    """Calculate asset returns for each asset in a portfolio.

    Params:
        portfolio_market_data: Dataframe of asset Market Prices grouped by Portfolio.
        price_type: String specifying which type of OHLC price.

    Returns:
        DataFrame containing the logarithmic & price return.

    Raises:
        ValueError: If the dataframe does not contain SecurityName and price_type

    """
    if "SecurityName" not in portfolio_market_data.columns or price_type not in portfolio_market_data.columns:
        raise ValueError(f"DataFrame must contain 'SecurityName' and {price_type} columns for calculating returns.")

    returns_df = pd.DataFrame()
    grouped = portfolio_market_data.groupby("SecurityName")
    for asset_name, asset_data in grouped:
        asset_data = asset_data.sort_values(by="AsOfDate")
        asset_data["Returns"] = asset_data[price_type].pct_change()
        asset_data["LogReturns"] = np.log(asset_data[price_type] / asset_data[price_type].shift(1))
        returns_df = pd.concat([returns_df, asset_data])

    asset_returns_pivot = pd.pivot(
        data=returns_df.dropna(),
        index="AsOfDate",
        columns=["PortfolioShortName", "SecurityName"],
        values=["Returns", "LogReturns"],
    )

    return asset_returns_pivot.sort_index(axis=1, level=[0, 1])


def plot_cumulative_returns(asset_returns: DataFrame) -> None:
    """Plot portfolio return over time based on asset returns per period.

    Params:
        asset_returns: Dataframe of asset Mreturns

    """
    # Create a figure and axis for the plot
    fig, ax = plt.subplots()

    # Plot cumulative returns for each portfolio
    for portfolio_name in asset_returns.columns:
        cumulative_returns = np.exp(asset_returns[portfolio_name].cumsum()) * 100
        ax.plot(
            asset_returns.index,
            cumulative_returns,
            label=portfolio_name,
            linewidth=2,
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns")
    ax.set_title("Cumulative Returns for Each Portfolio")
    ax.legend(loc="upper left")

    # Set the initial value of cumulative returns to 100
    for line in ax.lines:
        ydata = line.get_ydata()
        ydata[0] = 100  # type: ignore
        line.set_ydata(ydata)

    plt.show()


def calculate_portfolio_asset_weights(portfolio_market_data: DataFrame, price_type: str, weightage_type: str) -> DataFrame:
    """Calculate weight for each asset in a portfolio.

    Params:
        portfolio_market_data: Dataframe of asset Market Prices grouped by Portfolio.
        price_type: String specifying which type of OHLC price.

    Returns:
        DataFrame containing weights of each asset in a portfolio.

    """
    pivot_values = ["HeldShares", price_type]

    portfolio_market_data_pivot = pd.pivot(
        data=portfolio_market_data,
        index="AsOfDate",
        columns=["PortfolioShortName", "SecurityName"],
        values=pivot_values,
    )

    weights_by_portfolio = 0
    portfolio_market_data_pivot = portfolio_market_data_pivot.fillna(0)

    security_count_by_portfolio = portfolio_market_data_pivot["HeldShares"].apply(lambda x: x[x != 0].groupby(level=0).count(), axis=1)
    weights_by_portfolio = security_count_by_portfolio

    if weightage_type == "market_weighted":
        market_value = portfolio_market_data_pivot["HeldShares"].mul(portfolio_market_data_pivot[price_type])
        market_value = market_value.reorder_levels(["PortfolioShortName", "SecurityName"], axis=1)
        market_values_sum_by_portfolio = market_value.T.groupby(level=[0]).sum().T
        weights_by_portfolio = market_value.div(market_values_sum_by_portfolio, axis=1)

    elif weightage_type == "price_weighted":
        market_value = portfolio_market_data_pivot[price_type]
        market_value = market_value.reorder_levels(["PortfolioShortName", "SecurityName"], axis=1)
        market_value = market_value.sort_index(axis=1)
        market_values_sum_by_portfolio = market_value.T.groupby(level=[0]).sum().T
        weights_by_portfolio = market_value.div(market_values_sum_by_portfolio, axis=1)

    else:
        portfolio_sum = portfolio_market_data_pivot["HeldShares"].groupby(level=0, axis=1).sum()
        weights_by_portfolio = portfolio_market_data_pivot["HeldShares"].div(portfolio_sum, axis=0)

    return weights_by_portfolio
