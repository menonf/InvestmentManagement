"""Portfolio value-at-risk (historical simulation and parametric)."""

from __future__ import annotations

from typing import Any

import numpy as np
from pandas import DataFrame
from scipy import stats


def historical_var(asset_returns: DataFrame, weights: DataFrame, confidence: float) -> float:
    """Historical-simulation VaR: the ``1 - confidence`` quantile of simulated portfolio returns."""
    simulated = asset_returns.mul(weights.values, axis=1).sum(axis=1)
    return float(simulated.quantile(1.0 - confidence))


def parametric_var(asset_returns: DataFrame, weights: DataFrame, confidence: float, horizon_days: int = 1) -> Any:
    """Variance-covariance (normal) VaR scaled to ``horizon_days``."""
    mean = (asset_returns.mean().values * weights.values).sum()
    variance = np.dot(np.dot(weights, asset_returns.cov()), weights.T)[0][0]
    std = np.sqrt(variance) * np.sqrt(horizon_days)
    return stats.norm.ppf(1.0 - confidence, mean, std)


class PortfolioVaR:
    """Value at risk for one portfolio from its constituents' returns and latest weights.

    Args:
        portfolio_returns: constituent returns with a (portfolio, security)
            column MultiIndex (as produced by
            :func:`analytics.portfolio.returns.calculate_portfolio_constituent_returns`).
        portfolio_latest_weights: one-row frame of the latest weights for the
            portfolio's securities (same column order as the returns).
        PortfolioShortName: portfolio to evaluate.
        lookback_days: number of most recent observations used.
        horizon_days: forecast horizon for the parametric VaR.
        confidence_interval: e.g. ``0.95`` or ``0.99``.
    """

    def __init__(
        self,
        portfolio_returns: DataFrame,
        portfolio_latest_weights: DataFrame,
        PortfolioShortName: str,
        lookback_days: int = 250,
        horizon_days: int = 1,
        confidence_interval: float = 0.95,
    ):
        """Store the inputs."""
        self.portfolio_returns = portfolio_returns
        self.portfolio_latest_weights = portfolio_latest_weights
        self.PortfolioShortName = PortfolioShortName
        self.lookback_days = lookback_days
        self.horizon_days = horizon_days
        self.confidence_interval = confidence_interval

    @property
    def portfolio_short_name(self) -> str:
        """Snake-case alias of ``PortfolioShortName``."""
        return self.PortfolioShortName

    def get_recent_returns(self) -> DataFrame:
        """Return the last ``lookback_days`` rows (raises if history is too short)."""
        if len(self.portfolio_returns) < self.lookback_days:
            raise ValueError("Not enough data for the specified lookback period.")
        return self.portfolio_returns.iloc[-self.lookback_days :]

    def calculate_historical_var(self, recent_returns: DataFrame) -> float:
        """Historical VaR at the configured confidence level."""
        return historical_var(recent_returns, self.portfolio_latest_weights, self.confidence_interval)

    def calculate_parametric_var(self, recent_returns: DataFrame) -> Any:
        """Parametric VaR at the configured confidence level and horizon."""
        return parametric_var(recent_returns, self.portfolio_latest_weights, self.confidence_interval, self.horizon_days)

    def calculate_var(self) -> DataFrame:
        """Both VaR figures as a tidy metrics frame (one row per measure)."""
        recent = self.get_recent_returns()[self.PortfolioShortName]
        hist = self.calculate_historical_var(recent)
        param = self.calculate_parametric_var(recent)
        label = f"{self.horizon_days}-Day {int(self.confidence_interval * 100)}%"
        return DataFrame(
            {
                "AsOfDate": [self.portfolio_returns.index.max()] * 2,
                "PortfolioShortName": [self.PortfolioShortName] * 2,
                "MetricName": [f"{label} Historical VaR", f"{label} Parametric VaR"],
                "MetricType": ["Risk"] * 2,
                "MetricLevel": ["Portfolio"] * 2,
                "MetricValue": [hist, param],
            }
        )
