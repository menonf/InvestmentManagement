"""Momentum, value, composite factors and transforms (pure pandas)."""

import numpy as np
import pandas as pd
import pytest

from analytics.factors import CompositeFactor, MomentumFactor, ValueFactor, combine, neutralize, rank_pct, sample_on, winsorize, zscore
from analytics.portfolio.schedule import rebalance_dates
from tests.unit_tests.synthetic import synthetic_fundamentals, synthetic_prices


def test_momentum_is_12_minus_1() -> None:
    prices = synthetic_prices(n_securities=3, n_days=400)
    mom = MomentumFactor(lookback=252, skip=21).compute(prices)
    t = prices.index[300]
    expected = prices.iloc[300 - 21] / prices.iloc[300 - 21 - 252] - 1.0
    pd.testing.assert_series_equal(mom.loc[t], expected, check_names=False)
    # first lookback+skip rows are undefined
    assert mom.iloc[: 252 + 21].isna().all().all()
    assert mom.iloc[252 + 21 :].notna().all().all()


def test_momentum_skip_excludes_recent_month() -> None:
    prices = synthetic_prices(n_securities=2, n_days=320)
    # a +50% jump in the last 5 days must not enter a skip=21 signal
    bumped = prices.copy()
    bumped.iloc[-5:, 0] *= 1.5
    a = MomentumFactor(252, 21).compute(prices).iloc[-1, 0]
    b = MomentumFactor(252, 21).compute(bumped).iloc[-1, 0]
    assert np.isclose(a, b)
    c = MomentumFactor(252, 0).compute(bumped).iloc[-1, 0]
    assert c > a


def test_momentum_vol_scaled_changes_ranking_but_not_shape() -> None:
    prices = synthetic_prices(n_securities=10, n_days=400)
    plain = MomentumFactor().compute(prices)
    scaled = MomentumFactor(vol_scale=True).compute(prices)
    assert plain.shape == scaled.shape
    assert scaled.iloc[-1].notna().all()


def test_momentum_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        MomentumFactor(lookback=0)


def test_value_factor_prefers_cheap_names() -> None:
    panel = synthetic_fundamentals(list(range(1, 41)))
    panel.loc[1, ["P/E", "P/B", "P/S", "EV/EBIT"]] = [5.0, 0.8, 0.5, 4.0]  # very cheap
    panel.loc[2, ["P/E", "P/B", "P/S", "EV/EBIT"]] = [80.0, 15.0, 12.0, 60.0]  # very expensive
    score = ValueFactor().score_panel(panel)
    assert score.loc[1] > score.drop([1, 2]).max()
    assert score.loc[2] < score.drop([1, 2]).min()


def test_value_factor_treats_negative_multiples_as_missing() -> None:
    panel = synthetic_fundamentals(list(range(1, 21)))
    panel["P/E"] = -10.0
    panel[["P/B", "P/S", "EV/EBIT"]] = np.nan
    score = ValueFactor(min_components=1).score_panel(panel)
    assert score.isna().all()


def test_panel_factor_broadcast_and_dynamic() -> None:
    prices = synthetic_prices(n_securities=20, n_days=120)
    panel = synthetic_fundamentals(list(prices.columns))
    static = ValueFactor(fundamentals=panel).compute(prices)
    assert static.shape == prices.shape
    assert static.iloc[0].equals(static.iloc[-1])

    calls = []

    def fn(dt):
        calls.append(dt)
        return synthetic_fundamentals(list(prices.columns), seed=dt.month)

    dates = rebalance_dates(prices.index, "M")
    dyn = ValueFactor().compute_dynamic(prices, fn, dates=dates)
    assert len(calls) == len(dates)
    assert dyn.loc[dates[1]].notna().any()
    # forward filled between rebalances
    day_after = prices.index[prices.index.get_loc(dates[1]) + 1]
    pd.testing.assert_series_equal(dyn.loc[day_after], dyn.loc[dates[1]], check_names=False)


def test_zscore_and_rank() -> None:
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(5, 50)))
    z = zscore(frame)
    assert np.allclose(z.mean(axis=1), 0, atol=1e-9)
    assert np.allclose(z.std(axis=1, ddof=0), 1, atol=1e-6)
    r = rank_pct(frame)
    assert (r.max(axis=1) == 1.0).all()


def test_winsorize_clips_tails() -> None:
    s = pd.Series(np.arange(101, dtype=float))
    w = winsorize(s, 0.05, 0.95)
    assert w.min() == 5.0 and w.max() == 95.0


def test_neutralize_demeans_within_groups() -> None:
    scores = pd.DataFrame([[1.0, 3.0, 10.0, 20.0]], columns=[1, 2, 3, 4])
    groups = pd.Series({1: "A", 2: "A", 3: "B", 4: "B"})
    out = neutralize(scores, groups)
    assert np.allclose(out.loc[0, [1, 2]], [-1.0, 1.0])
    assert np.allclose(out.loc[0, [3, 4]], [-5.0, 5.0])


def test_combine_matches_original_notebook_zscore_sum_up_to_scale() -> None:
    """Original notebook: composite = z(mom) + z(value) with NaN -> 0. combine() returns the
    weight-normalised version (divided by 2), so ranks are identical."""
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2024-01-01", periods=30)
    cols = list(range(10))
    mom = pd.DataFrame(rng.normal(size=(30, 10)), index=idx, columns=cols)
    val = pd.DataFrame(rng.normal(size=(30, 10)), index=idx, columns=cols)
    val.iloc[3, 2] = np.nan
    expected = pd.DataFrame(index=idx, columns=cols, dtype=float)
    for dt in idx:
        m = ((mom.loc[dt] - mom.loc[dt].mean()) / (mom.loc[dt].std(ddof=0) + 1e-9)).fillna(0)
        v = ((val.loc[dt] - val.loc[dt].mean()) / (val.loc[dt].std(ddof=0) + 1e-9)).fillna(0)
        expected.loc[dt] = m + v
    got = combine({"momentum": mom, "value": val})
    np.testing.assert_allclose(got.to_numpy(dtype=float) * 2, expected.to_numpy(dtype=float), atol=1e-9)
    pd.testing.assert_frame_equal(got.rank(axis=1), expected.rank(axis=1))


def test_combine_rank_method_and_weights() -> None:
    idx = pd.bdate_range("2024-01-01", periods=3)
    a = pd.DataFrame([[1, 2, 3]] * 3, index=idx, columns=[1, 2, 3], dtype=float)
    b = pd.DataFrame([[3, 2, 1]] * 3, index=idx, columns=[1, 2, 3], dtype=float)
    equal = combine({"a": a, "b": b}, method="rank")
    assert np.allclose(equal.to_numpy(), 0.0)
    tilted = combine({"a": a, "b": b}, weights={"a": 3, "b": 1}, method="rank")
    assert tilted.loc[idx[0], 3] > tilted.loc[idx[0], 1]
    with pytest.raises(ValueError):
        combine({})


def test_sample_on_forward_fills_between_rebalances() -> None:
    idx = pd.bdate_range("2024-01-01", periods=45)
    sig = pd.DataFrame(np.arange(45, dtype=float).reshape(-1, 1), index=idx, columns=[1])
    monthly = rebalance_dates(idx, "M")
    out = sample_on(sig, monthly)
    assert (out.loc[monthly, 1] == sig.loc[monthly, 1]).all()
    second = idx[idx.get_loc(monthly[0]) + 3]
    assert out.loc[second, 1] == sig.loc[monthly[0], 1]


def test_composite_factor_end_to_end() -> None:
    prices = synthetic_prices(n_securities=15, n_days=330)
    panel = synthetic_fundamentals(list(prices.columns))
    comp = CompositeFactor(
        {"momentum": MomentumFactor(126, 21), "value": ValueFactor(fundamentals=panel)},
        weights={"momentum": 0.5, "value": 0.5},
        rebalance_dates=rebalance_dates(prices.index, "M"),
    )
    sig = comp.compute(prices)
    assert sig.shape == prices.shape
    assert sig.iloc[-1].notna().all()
