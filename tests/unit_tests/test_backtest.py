"""Backtest engine and metrics."""

import numpy as np
import pandas as pd

from analytics.backtest import (
    BacktestResult,
    annualized_return,
    cumulative_returns,
    forward_returns,
    ic_summary,
    information_coefficient,
    max_drawdown,
    portfolio_returns_from_constituents,
    prices_to_returns,
    quantile_returns,
    run_backtest,
    sharpe_ratio,
    summary_table,
    turnover,
)
from analytics.factors import MomentumFactor
from analytics.portfolio import quantile_weights
from tests.unit_tests.synthetic import synthetic_prices


def test_run_backtest_shift_convention() -> None:
    idx = pd.bdate_range("2024-01-01", periods=4)
    ret = pd.DataFrame([[0.01, -0.01], [0.02, 0.0], [0.0, 0.03], [0.01, 0.01]], index=idx, columns=[1, 2])
    w = pd.DataFrame([[1.0, 0.0]] * 4, index=idx, columns=[1, 2])
    res0 = run_backtest(w, ret, shift_weights=0)
    res1 = run_backtest(w, ret, shift_weights=1)
    np.testing.assert_allclose(res0.returns["strategy"].to_numpy(), [0.01, 0.02, 0.0, 0.01])
    np.testing.assert_allclose(res1.returns["strategy"].to_numpy(), [0.0, 0.02, 0.0, 0.01])


def test_costs_charged_on_turnover() -> None:
    idx = pd.bdate_range("2024-01-01", periods=3)
    ret = pd.DataFrame(0.0, index=idx, columns=[1, 2])
    w = pd.DataFrame([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], index=idx, columns=[1, 2])
    res = run_backtest(w, ret, shift_weights=0, cost_bps=100)
    # one-way turnover = 0.5 * sum|dw|: day 1 enters from cash (0.5), day 2 is a
    # full switch (1.0), day 3 nothing -> costs 0.5%, 1%, 0 at 100 bps
    np.testing.assert_allclose(res.costs["strategy"].to_numpy(), [0.005, 0.01, 0.0])
    assert (res.returns["strategy"] <= res.gross_returns["strategy"]).all()


def test_reconstruction_path_matches_notebook_groupby_sum() -> None:
    idx = pd.bdate_range("2024-01-01", periods=5)
    cols = pd.MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 3)])
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0, 0.01, size=(5, 3)), index=idx, columns=cols)
    w = pd.DataFrame([[0.6, 0.4, 1.0]] * 5, index=idx, columns=cols)
    expected = (r * w).T.groupby(level=0).sum().T
    got = portfolio_returns_from_constituents(r, w)
    pd.testing.assert_frame_equal(got, expected)


def test_metrics_basic_values() -> None:
    r = pd.Series([0.1, -0.05, 0.02])
    assert np.isclose(cumulative_returns(r).iloc[-1], 1.1 * 0.95 * 1.02 - 1)
    assert np.isclose(max_drawdown(pd.Series([0.1, -0.5, 0.2])), -0.5)
    zero = pd.Series([0.0, 0.0, 0.0])
    assert np.isnan(sharpe_ratio(zero)) or np.isinf(sharpe_ratio(zero)) or sharpe_ratio(zero) == 0
    daily = pd.Series([0.001] * 252)
    assert np.isclose(annualized_return(daily), 1.001**252 - 1)


def test_summary_table_columns_and_benchmark_stats() -> None:
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2023-01-01", periods=300)
    rets = pd.DataFrame({"mkt": rng.normal(0.0004, 0.01, 300)}, index=idx)
    rets["strat"] = 0.5 * rets["mkt"] + rng.normal(0, 0.005, 300)
    table = summary_table(rets, benchmark="mkt")
    for col in ["total_return", "cagr", "volatility", "sharpe", "sortino", "max_drawdown", "calmar", "hit_rate", "periods", "beta", "correlation", "tracking_error"]:
        assert col in table.columns
    assert 0.3 < table.loc["strat", "beta"] < 0.7
    assert np.isnan(table.loc["mkt", "beta"])


def test_turnover_and_result_summary() -> None:
    idx = pd.bdate_range("2024-01-01", periods=3)
    w = pd.DataFrame([[1.0, 0.0], [0.5, 0.5], [0.5, 0.5]], index=idx, columns=[1, 2])
    to = turnover(w)
    np.testing.assert_allclose(to.to_numpy(), [0.5, 0.5, 0.0])
    res = BacktestResult(returns=pd.DataFrame({"s": [0.01, 0.0, -0.01]}, index=idx), weights={"s": w})
    assert "annual_turnover" in res.summary().columns
    assert res.growth_of_one().iloc[0, 0] == 1.01


def test_signal_metrics_on_a_predictive_signal() -> None:
    prices = synthetic_prices(n_securities=30, n_days=300, seed=5)
    fwd = forward_returns(prices, 21)
    # a signal equal to the future return has IC ~ 1 and monotone quantiles
    ic = information_coefficient(fwd, fwd)
    assert np.isclose(ic.dropna().mean(), 1.0)
    summ = ic_summary(ic)
    assert summ["pct_positive"] == 1.0
    q = quantile_returns(fwd, fwd, n_quantiles=5)
    assert (q.mean().diff().dropna() > 0).all()


def test_momentum_long_short_end_to_end_is_deterministic() -> None:
    prices = synthetic_prices(n_securities=40, n_days=600)
    sig = MomentumFactor(126, 21).compute(prices)
    w = quantile_weights(sig, 0.2)
    res = run_backtest(w, prices_to_returns(prices), shift_weights=1, cost_bps=5)
    assert len(res.returns) == len(prices)
    assert res.summary().loc["strategy", "periods"] == len(prices) - 0
    # dollar neutral: gross long == gross short once the signal exists
    live = w.abs().sum(axis=1) > 0
    assert np.allclose(w[live].clip(lower=0).sum(axis=1), 1.0)
    assert np.allclose(w[live].clip(upper=0).sum(axis=1), -1.0)
