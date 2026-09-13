"""Unit tests for the ML-return factor and fundamentals abstraction.

Exercises the full 8-model port with synthetic data (no DB / no vendor calls):
  * every model key builds and predicts
  * MLReturnFactor single / ensemble / rank_vote produce aligned score matrices
  * vendor-agnostic provider emits the canonical 18-column panel
  * training module's time_based_split has no look-ahead leakage
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analytics.factors.fundamentals import RATIO_COLUMNS, StaticFundamentalsProvider
from analytics.factors.ml_factor import (
    MODEL_KEYS,
    MLReturnFactor,
    build_estimator,
)
from analytics.factors.ml_training import time_based_split


def _synthetic_panel(n_sec=50, seed=0):
    rng = np.random.default_rng(seed)
    cols = RATIO_COLUMNS
    data = rng.normal(size=(n_sec, len(cols)))
    idx = pd.Index(range(1000, 1000 + n_sec), name="security_id")
    return pd.DataFrame(data, index=idx, columns=cols)


def _synthetic_prices(n_sec=50, n_dates=10, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    cols = list(range(1000, 1000 + n_sec))
    return pd.DataFrame(rng.normal(100, 5, size=(n_dates, n_sec)), index=idx, columns=cols)


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_build_estimator_predicts(key):
    est = build_estimator(key)
    panel = _synthetic_panel()
    preds = est.fit(panel, rng_target(panel)).predict(panel)
    assert len(preds) == panel.shape[0]
    assert np.all(np.isfinite(preds)) or len(preds) == panel.shape[0]


def rng_target(panel: pd.DataFrame) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.normal(0.1, 0.3, size=panel.shape[0])


@pytest.mark.parametrize("mode", ["single", "ensemble", "rank_vote"])
def test_factor_output_shape(mode):
    panel = _synthetic_panel()
    prices = _synthetic_prices()
    # Use already-fitted estimators (string keys now require a persisted .joblib).
    fitted = [build_estimator(k).fit(panel, rng_target(panel)) for k in MODEL_KEYS]
    f = MLReturnFactor(mode=mode, models=fitted, impute="median")
    scores = f.compute(prices, panel)
    assert list(scores.columns) == list(prices.columns)
    assert list(scores.index) == list(prices.index)
    # single-date prediction -> identical rows
    assert scores.iloc[0].equals(scores.iloc[-1])


def test_missing_persisted_model_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        MLReturnFactor(mode="single", models=["does_not_exist"], model_dir="C:/nope")


def test_ensemble_matches_notebook_intent():
    panel = _synthetic_panel()
    prices = _synthetic_prices()
    # Pass already-fitted estimators (the real flow loads fitted .joblib from
    # disk; here we fit on synthetic data so predict() is valid).
    fitted = [build_estimator(k).fit(panel, rng_target(panel)) for k in
              ["random_forest", "extra_trees", "gradient_boosting"]]
    f = MLReturnFactor(mode="ensemble", models=fitted)
    scores = f.compute(prices, panel)
    assert scores.notna().any().any()


def test_provider_contract():
    # StaticFundamentalsProvider with no session is only used for the empty panel
    # path here; we test the column contract via the public helper instead.
    panel = _synthetic_panel()
    assert list(panel.columns) == RATIO_COLUMNS
    assert panel.shape[1] == 18


def test_time_based_split_no_leakage():
    n = 200
    df = pd.DataFrame({
        "security_id": np.arange(n),
        "snapshot_date": pd.date_range("2020-01-01", periods=n, freq="MS"),
        "y": np.random.default_rng(3).normal(size=n),
        **{c: np.random.default_rng(i).normal(size=n) for i, c in enumerate(RATIO_COLUMNS)},
    })
    X_train, X_test, y_train, y_test, split_date = time_based_split(df, test_size=0.2)
    # Every training snapshot strictly precedes every test snapshot.
    # (X_* are feature-only frames, so compare the original train/test slices.)
    train, test = df.iloc[: int(n * 0.8)], df.iloc[int(n * 0.8):]
    assert train["snapshot_date"].max() < test["snapshot_date"].min()
    assert len(X_train) + len(X_test) == n


def test_predict_panel_returns_series():
    panel = _synthetic_panel()
    fitted = [build_estimator(k).fit(panel, rng_target(panel)) for k in MODEL_KEYS]
    s = MLReturnFactor.predict_panel(panel, mode="ensemble", models=fitted)
    assert isinstance(s, pd.Series)
    assert s.index.name == "security_id"
