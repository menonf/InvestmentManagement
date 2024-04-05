"""Module for investment risk analytics."""
import numpy as np
import scipy as sp
from pandas import DataFrame


def calculate_portfolio_var(portfolio_returns: DataFrame, portfolio_latest_weights: DataFrame, PortfolioShortName: str) -> DataFrame:
    """Calculate Historical, Parametric Value at Risk of a Portfolio .

    Params:
        portfolio_returns: Dataframe of prtfolio returns.
        portfolio_latest_weights: T-1 asset weights in a portfolio
        PortfolioShortName: String specifying Portfolio Short Name.

    Returns:
        DataFrame containing VaR at 95 & 99 Confidence Interval.

    """
    # Historic Simulation
    historicSimulatedPrice = portfolio_returns.mul(portfolio_latest_weights.values, axis=1)
    historicSimulatedPortfolio = historicSimulatedPrice.sum(axis=1)
    historicSimulationVaR95 = historicSimulatedPortfolio.quantile(0.05)
    historicSimulationVaR99 = historicSimulatedPortfolio.quantile(0.01)

    # Analytical
    portfolioMean = (portfolio_returns.mean() * portfolio_latest_weights).sum(axis=1)
    portfolioVariance = np.dot(np.dot(portfolio_latest_weights, portfolio_returns.cov()), portfolio_latest_weights.T)
    portfolioStdDeviation = np.sqrt(portfolioVariance)
    parametricVaR95 = sp.stats.norm.ppf(0.05, portfolioMean, portfolioStdDeviation)
    parametricVaR99 = sp.stats.norm.ppf(0.01, portfolioMean, portfolioStdDeviation)

    data = {
        "AsOfDate": [portfolio_returns.index.max()] * 4,
        "PortfolioShortName": [PortfolioShortName] * 4,
        "MetricName": ["1 Day 95% Historical VaR", "1 Day 99% Historical VaR", "1 Day 95% Parametric VaR", "1 Day 99% Parametric VaR"],
        "MetricType": ["Risk"] * 4,
        "MetricLevel": ["Portfolio"] * 4,
        "MetricValue": [historicSimulationVaR95, historicSimulationVaR99, parametricVaR95[0][0], parametricVaR99[0][0]],
    }

    return DataFrame.from_dict(data, orient="columns")
