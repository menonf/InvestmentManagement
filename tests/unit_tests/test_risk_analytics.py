"""Value-at-risk regression test (values carried over from the original test-suite)."""

import os

import numpy as np
import pandas as pd
import pytest

from analytics.performance import performance_analytics as perf
from analytics.risk import risk_analytics as risk

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "MSCIRiskMetricsPrices.csv")


@pytest.mark.skipif(not os.path.exists(DATA), reason="Test file missing")
def test_value_at_risk() -> None:
    web_data = pd.read_csv(DATA, header=0)
    asset_returns = perf.calculate_portfolio_constituent_returns(web_data, "close")["log_returns"]
    asset_weights = perf.calculate_portfolio_constituent_weights(web_data, "close", "price_weighted")
    latest = asset_weights.loc[asset_weights.index.max() : asset_weights.index.max()]

    var = risk.PortfolioVaR(asset_returns, latest, "TVAR", lookback_days=252, horizon_days=1, confidence_interval=0.95)
    df_var = var.calculate_var()
    np.testing.assert_almost_equal(df_var["MetricValue"].to_numpy(), np.array([-0.03265852459105972, -0.03059793484961367]), decimal=6)
    assert list(df_var["MetricName"]) == ["1-Day 95% Historical VaR", "1-Day 95% Parametric VaR"]


def test_lookback_guard() -> None:
    returns = pd.DataFrame({("P", 1): [0.01] * 10}, index=pd.bdate_range("2024-01-01", periods=10))
    returns.columns = pd.MultiIndex.from_tuples(returns.columns)
    weights = pd.DataFrame([[1.0]], columns=pd.MultiIndex.from_tuples([("P", 1)]))
    with pytest.raises(ValueError):
        risk.PortfolioVaR(returns, weights, "P", lookback_days=252).calculate_var()
