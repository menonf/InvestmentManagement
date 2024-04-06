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
    portfolio_var = risk.calculate_portfolio_var(portfolio_asset_returns, portfolio_latest_weights, "TVAR")
    df_var = pd.DataFrame.from_dict(portfolio_var, orient="columns")
    var_values = [-0.032606116579867096, -0.05581146041933242, -0.030116069430500835, -0.043124354820760574]
    assert df_var["MetricValue"].to_list() == var_values

