"""Module for investment risk analytics."""

from typing import Any

import numpy as np
from pandas import DataFrame
from scipy import stats


class PortfolioVaR:
    """Class to calculate value at risk."""

    def __init__(
        self,
        portfolio_returns: DataFrame,
        portfolio_latest_weights: DataFrame,
        PortfolioShortName: str,
        lookback_days: int = 250,
        horizon_days: int = 1,
        confidence_interval: float = 0.95,
    ):
        """
        Initialize the VaR calculator with portfolio data.

        Params:
            portfolio_returns: Dataframe of portfolio returns.
            portfolio_latest_weights: T-1 asset weights in a portfolio.
            PortfolioShortName: String specifying Portfolio Short Name.
            lookback_years: The lookback period in years (default is 1 years).
            horizon_days: Forecast horizon in days (default is 1 day).
            confidence_interval: Confidence interval for VaR (default is 95%).
        """
        self.portfolio_returns = portfolio_returns
        self.portfolio_latest_weights = portfolio_latest_weights
        self.PortfolioShortName = PortfolioShortName
        self.lookback_days = lookback_days
        self.horizon_days = horizon_days
        self.confidence_interval = confidence_interval
        self.lookback_days = lookback_days

    def get_recent_returns(self) -> DataFrame:
        """
        Get the returns for the lookback period.

        Returns:
        DataFrame: A dataframe of returns over the lookback period.
        """
        if len(self.portfolio_returns) < self.lookback_days:
            raise ValueError("Not enough data for the specified lookback period.")
        return self.portfolio_returns[-self.lookback_days :]  # noqa

    def calculate_historical_var(self, recent_returns: DataFrame) -> float:
        """
        Calculate Historical VaR at the specified confidence level.

        Params:
        recent_returns: DataFrame of returns for the lookback period.

        Returns:
        float: Historical VaR at the specified confidence interval.
        """
        alpha = 1 - self.confidence_interval
        historic_simulated_price = recent_returns.mul(self.portfolio_latest_weights.values, axis=1)
        historic_simulated_portfolio = historic_simulated_price.sum(axis=1)
        var = float(historic_simulated_portfolio.quantile(alpha))
        return var

    def calculate_parametric_var(self, recent_returns: DataFrame) -> Any:
        """
        Calculate Parametric VaR at the specified confidence level.

        Params:
        recent_returns: DataFrame of returns for the lookback period.

        Returns:
        float: Parametric VaR at the specified confidence interval.
        """
        portfolio_mean = (recent_returns.mean().values * self.portfolio_latest_weights.values).sum()
        portfolio_variance = np.dot(np.dot(self.portfolio_latest_weights, recent_returns.cov()), self.portfolio_latest_weights.T)[0][0]
        portfolio_std_dev = np.sqrt(portfolio_variance)

        # Adjust for the horizon by scaling the standard deviation
        adjusted_std_dev = portfolio_std_dev * np.sqrt(self.horizon_days)
        alpha = 1 - self.confidence_interval
        var = stats.norm.ppf(alpha, portfolio_mean, adjusted_std_dev)

        return var

    def calculate_var(self) -> DataFrame:
        """
        Calculate both Historical and Parametric VaR.

        Returns:
        DataFrame: A DataFrame containing VaR at the specified confidence interval.
        """
        recent_returns = self.get_recent_returns()[self.PortfolioShortName]

        # Calculate Historical VaR
        hist_var = self.calculate_historical_var(recent_returns)

        # Calculate Parametric VaR
        param_var = self.calculate_parametric_var(recent_returns)

        # Prepare the result in a DataFrame
        data = {
            "AsOfDate": [self.portfolio_returns.index.max()] * 2,
            "PortfolioShortName": [self.PortfolioShortName] * 2,
            "MetricName": [
                f"{self.horizon_days}-Day {int(self.confidence_interval * 100)}% Historical VaR",
                f"{self.horizon_days}-Day {int(self.confidence_interval * 100)}% Parametric VaR",
            ],
            "MetricType": ["Risk"] * 2,
            "MetricLevel": ["Portfolio"] * 2,
            "MetricValue": [hist_var, param_var],
        }

        return DataFrame.from_dict(data, orient="columns")
