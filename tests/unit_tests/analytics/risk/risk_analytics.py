"""Unit Tests for risk_analytics."""
import pathlib

import pandas as pd

from analytics.performance import performance_analytics as perf
from analytics.risk import risk_analytics as risk


def test_value_at_risk() -> None:
    """This function test whether the Value at Risk result is as expected."""
    webData = pd.read_csv(f"{pathlib.Path().resolve()}\\tests\\unit_tests\\analytics\\risk\\MSCIRiskMetricsPrices.csv", header=0)
    portfolio_asset_returns = perf.calculate_portfolio_asset_returns(webData, "Close")["LogReturns"]
    portfolio_asset_weights = perf.calculate_portfolio_asset_weights(webData, "Close", "price_weighted")
    max_date_index = portfolio_asset_weights.index.max()
    portfolio_latest_weights = portfolio_asset_weights[max_date_index:max_date_index]

    portfolio_var = risk.PortfolioVaR(portfolio_asset_returns,
                                      portfolio_latest_weights,
                                      "Portfolio_ABC",
                                      lookback_days=252,
                                      horizon_days=1,
                                      confidence_interval=0.95)

    df_var = pd.DataFrame.from_dict(portfolio_var.calculate_var(), orient="columns")
    var_values = [-0.03265852459105973, -0.03059793484961368]
    assert df_var["MetricValue"].to_list() == var_values
