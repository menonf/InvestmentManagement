"""Rebalance schedules and signal -> weight construction."""

import numpy as np
import pandas as pd
import pytest

from analytics.portfolio import cap_weights, month_starts, quantile_weights, quarter_starts, rebalance_dates


def test_month_and_quarter_starts() -> None:
    idx = pd.bdate_range("2024-01-01", "2024-06-30")
    ms = month_starts(idx)
    assert list(ms.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-06-03"]
    qs = quarter_starts(idx)
    assert list(qs.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-04-01"]
    assert len(rebalance_dates(idx, "D")) == len(idx)
    with pytest.raises(ValueError):
        rebalance_dates(idx, "X")


def test_quantile_weights_dollar_neutral_equal_weight() -> None:
    idx = pd.bdate_range("2024-01-01", periods=2)
    sig = pd.DataFrame([np.arange(20, dtype=float)] * 2, index=idx, columns=range(20))
    w = quantile_weights(sig, quantile=0.1)
    assert np.isclose(w.loc[idx[0]].clip(lower=0).sum(), 1.0)
    assert np.isclose(w.loc[idx[0]].clip(upper=0).sum(), -1.0)
    assert w.loc[idx[0], 19] == 0.5 and w.loc[idx[0], 18] == 0.5
    assert w.loc[idx[0], 0] == -0.5 and w.loc[idx[0], 1] == -0.5
    assert (w.loc[idx[0], 2:17] == 0).all()


def test_quantile_weights_matches_original_notebook_loop() -> None:
    """The original notebook took ceil(10% of names) per leg with equal weights and
    computed long_mean - short_mean; the vectorised weights must reproduce it."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2024-01-01", periods=40)
    cols = list(range(37))
    sig = pd.DataFrame(rng.normal(size=(40, 37)), index=idx, columns=cols)
    sig.iloc[5, :30] = np.nan  # a sparse day
    ret = pd.DataFrame(rng.normal(0, 0.01, size=(40, 37)), index=idx, columns=cols)
    expected = pd.Series(index=idx, dtype=float)
    for dt in idx:
        s = sig.loc[dt].dropna().sort_values(ascending=False)
        if len(s) < 4:
            expected.loc[dt] = 0.0
            continue
        n = max(1, int(np.ceil(0.1 * len(s))))
        expected.loc[dt] = ret.loc[dt, s.index[:n]].mean() - ret.loc[dt, s.index[-n:]].mean()
    w = quantile_weights(sig, quantile=0.1, min_names=4)
    got = (w * ret).sum(axis=1)
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), atol=1e-12)


def test_quantile_weights_respects_membership_and_long_only() -> None:
    idx = pd.bdate_range("2024-01-01", periods=1)
    sig = pd.DataFrame([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]], index=idx, columns=range(6))
    member = pd.DataFrame([[False, True, True, True, True, True]], index=idx, columns=range(6))
    w = quantile_weights(sig, quantile=0.2, membership=member, min_names=2)
    assert w.loc[idx[0], 0] == 0.0  # excluded despite top score
    assert w.loc[idx[0], 1] > 0
    lo = quantile_weights(sig, quantile=0.5, long_only=True, min_names=2)
    assert np.isclose(lo.sum(axis=1).iloc[0], 1.0) and (lo >= 0).all().all()


def test_cap_weights_redistributes() -> None:
    w = pd.DataFrame([[0.7, 0.3, -0.5, -0.5]], columns=range(4))
    capped = cap_weights(w, 0.5)
    assert capped.iloc[0, 0] <= 0.5 + 1e-9
    assert np.isclose(capped.iloc[0, :2].sum(), 1.0)
    assert np.isclose(capped.iloc[0, 2:].sum(), -1.0)
