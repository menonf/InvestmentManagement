"""Constituent weights and returns from the long (date, portfolio, security) frame."""

import numpy as np
import pandas as pd
import pytest

from analytics.backtest import portfolio_returns_from_constituents
from analytics.portfolio import calculate_portfolio_constituent_returns, calculate_portfolio_constituent_weights
from tests.unit_tests.synthetic import long_market_frame, synthetic_prices


def _frame() -> pd.DataFrame:
    prices = synthetic_prices(n_securities=5, n_days=30)
    port = long_market_frame(prices, "PORT", 1, "Portfolio", held_shares=1.0)
    bench = long_market_frame(prices, "BENCH", 2, "Benchmark", held_shares=1.0)
    bench["shares_outstanding"] = bench["security_id"].map({c: 1e6 * (i + 1) for i, c in enumerate(prices.columns)})
    port["shares_outstanding"] = np.nan
    return pd.concat([port, bench], ignore_index=True)


def test_returns_pivot_shape_and_values() -> None:
    frame = _frame()
    out = calculate_portfolio_constituent_returns(frame, "adj_close")
    rets = out["returns"]
    assert set(rets.columns.get_level_values(0)) == {"PORT", "BENCH"}
    px = frame[(frame["portfolio_short_name"] == "PORT") & (frame["security_id"] == 1001)].set_index("as_of_date")["adj_close"]
    expected = px.pct_change().dropna()
    np.testing.assert_allclose(rets[("PORT", 1001)].dropna().to_numpy(), expected.to_numpy())
    assert np.allclose(out["log_returns"].to_numpy(dtype=float), np.log1p(out["returns"].to_numpy(dtype=float)), equal_nan=True)


def test_returns_requires_columns() -> None:
    with pytest.raises(ValueError):
        calculate_portfolio_constituent_returns(pd.DataFrame({"as_of_date": []}), "close")


def test_equal_weights_sum_to_one() -> None:
    w = calculate_portfolio_constituent_weights(_frame(), "adj_close", "equal_weighted")
    np.testing.assert_allclose(w["PORT"].sum(axis=1).to_numpy(), 1.0)
    assert np.allclose(w["PORT"].to_numpy(), 0.2)
    # a Benchmark has no prior-day market value on the first date -> no weight that day
    assert np.allclose(w["BENCH"].iloc[0].to_numpy(), 0.0)
    np.testing.assert_allclose(w["BENCH"].iloc[1:].sum(axis=1).to_numpy(), 1.0)


def test_benchmark_market_weights_anchor_on_prior_day_cap() -> None:
    frame = _frame()
    w = calculate_portfolio_constituent_weights(frame, "adj_close", "market_weighted")
    bench = frame[frame["portfolio_short_name"] == "BENCH"].pivot_table(index="as_of_date", columns="security_id", values="adj_close")
    shares = pd.Series({c: 1e6 * (i + 1) for i, c in enumerate(bench.columns)})
    cap_prev = bench.shift(1) * shares
    expected = cap_prev.div(cap_prev.sum(axis=1), axis=0).iloc[1:]
    np.testing.assert_allclose(w["BENCH"].iloc[1:].to_numpy(), expected.to_numpy(), atol=1e-12)
    assert np.allclose(w["BENCH"].iloc[0].to_numpy(), 0.0)  # no prior day on the first date


def test_max_weight_cap_renormalises() -> None:
    w = calculate_portfolio_constituent_weights(_frame(), "adj_close", "market_weighted", max_weight=0.25)
    live = w["BENCH"].iloc[1:]
    assert (live <= 0.25 + 1e-9).all().all()
    np.testing.assert_allclose(live.sum(axis=1).to_numpy(), 1.0)


def test_portfolio_specific_weighting_and_bad_method() -> None:
    frame = _frame()
    w = calculate_portfolio_constituent_weights(frame, "adj_close", "market_weighted", portfolio_specific_weights={"PORT": "equal_weighted"})
    assert np.allclose(w["PORT"].to_numpy(), 0.2)
    with pytest.raises(ValueError):
        calculate_portfolio_constituent_weights(frame, "adj_close", "nonsense")


def test_reconstruction_returns_pipeline() -> None:
    frame = _frame()
    rets = calculate_portfolio_constituent_returns(frame, "adj_close")["returns"]
    w = calculate_portfolio_constituent_weights(frame, "adj_close", "equal_weighted")
    port = portfolio_returns_from_constituents(rets, w)
    assert list(port.columns) == ["BENCH", "PORT"]
    # equal-weighted return == mean constituent return
    np.testing.assert_allclose(port["PORT"].to_numpy(), rets["PORT"].mean(axis=1).to_numpy(), atol=1e-12)
