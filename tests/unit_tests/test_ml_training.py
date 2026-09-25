"""Pure-pandas parts of the ML training pipeline (no scikit-learn, no DB)."""

import numpy as np
import pandas as pd

from analytics.factors.ml_training import forward_return_labels, time_based_split
from data_engineering.fundamentals import RATIO_COLUMNS
from tests.unit_tests.synthetic import synthetic_prices


def test_forward_return_labels_use_actual_trading_days() -> None:
    prices = synthetic_prices(n_securities=3, n_days=300)
    snaps = pd.date_range("2021-01-31", "2021-09-30", freq="ME")
    labels = forward_return_labels(prices, snaps, forward_months=3, min_history=30)
    assert set(labels.columns) == {"snapshot_date", "security_id", "y"}
    assert labels["snapshot_date"].isin(prices.index).all()
    first = labels[labels["security_id"] == 1001].iloc[0]
    t0 = first["snapshot_date"]
    t1 = prices.index[prices.index >= t0 + pd.DateOffset(months=3)][0]
    assert np.isclose(first["y"], prices.loc[t1, 1001] / prices.loc[t0, 1001] - 1.0)


def test_time_based_split_has_no_leakage() -> None:
    n_dates, n_sec = 24, 10
    dates = pd.date_range("2020-01-31", periods=n_dates, freq="ME")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"snapshot_date": np.repeat(dates, n_sec), "security_id": np.tile(range(n_sec), n_dates), "y": rng.normal(size=n_dates * n_sec)})
    for c in RATIO_COLUMNS:
        df[c] = rng.normal(size=len(df))
    df = df.sample(frac=1.0, random_state=1)  # shuffle rows: the split must not depend on row order
    X_train, X_test, y_train, y_test, split_date = time_based_split(df, test_size=0.25)
    train_dates = df.loc[X_train.index, "snapshot_date"]
    test_dates = df.loc[X_test.index, "snapshot_date"]
    assert train_dates.max() < test_dates.min()
    assert test_dates.min() == pd.Timestamp(split_date)
    assert len(X_train) + len(X_test) == len(df)
    assert list(X_train.columns) == RATIO_COLUMNS

    _, _, _, _, split2 = time_based_split(df, test_size=0.25, embargo_months=3)
    X_train_e = time_based_split(df, test_size=0.25, embargo_months=3)[0]
    assert df.loc[X_train_e.index, "snapshot_date"].max() < pd.Timestamp(split2) - pd.DateOffset(months=3)
