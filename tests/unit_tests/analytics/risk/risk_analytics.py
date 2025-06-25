"""Unit Tests for risk_analytics."""
import os

import pandas as pd
import pytest

from analytics.performance import performance_analytics as perf
from analytics.risk import risk_analytics as risk

# Dynamically construct the file path
test_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(test_dir, "MSCIRiskMetricsPrices.csv")

@pytest.mark.skipif(not os.path.exists(file_path), reason="Test file missing")
def test_value_at_risk() -> None:
    """This function tests whether the Value at Risk result is as expected."""
    webData = pd.read_csv(file_path, header=0) 

    portfolio_asset_returns = perf.calculate_portfolio_constituent_returns(webData, "close")["log_returns"]
    portfolio_asset_weights = perf.calculate_portfolio_constituent_weights(webData, "close", "price_weighted")
    
    max_date_index = portfolio_asset_weights.index.max()
    portfolio_latest_weights = portfolio_asset_weights.loc[max_date_index:max_date_index]

    portfolio_var = risk.PortfolioVaR(
        portfolio_asset_returns,
        portfolio_latest_weights,
        "TVAR",
        lookback_days=252,
        horizon_days=1,
        confidence_interval=0.95
    )

    df_var = pd.DataFrame.from_dict(portfolio_var.calculate_var(), orient="columns")
    var_values = [-0.03265852459105973, -0.03059793484961368]
    assert df_var["MetricValue"].to_list() == var_values
