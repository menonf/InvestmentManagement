"""ML value factor and training helpers (scikit-learn required; skipped when unavailable)."""

import numpy as np
import pandas as pd
import pytest

from analytics.factors.ml_training import forward_return_labels, time_based_split
from data_engineering.fundamentals import RATIO_COLUMNS
from tests.unit_tests.synthetic import synthetic_fundamentals, synthetic_prices

sklearn = pytest.importorskip("sklearn")

from analytics.factors.ml_value import MODEL_KEYS, MLReturnFactor, build_estimator  # noqa: E402


def _target(panel: pd.DataFrame) -> np.ndarray:
    return np.random.default_rng(7).normal(0.1, 0.3, size=panel.shape[0])


@pytest.mark.parametrize("key", MODEL_KEYS)
def test_every_model_key_builds_and_predicts(key: str) -> None:
    panel = synthetic_fundamentals(list(range(1000, 1050))).fillna(0.0)
    preds = build_estimator(key).fit(panel.to_numpy(), _target(panel)).predict(panel.to_numpy())
    assert len(preds) == len(panel)


@pytest.mark.parametrize("mode", ["single", "ensemble", "rank_vote"])
def test_factor_output_shape(mode: str) -> None:
    panel = synthetic_fundamentals(list(range(1000, 1050)))
    prices = synthetic_prices(n_securities=50, n_days=10)
    prices.columns = list(range(1000, 1050))
    fitted = [build_estimator(k).fit(panel.fillna(0.0).to_numpy(), _target(panel)) for k in ["random_forest", "extra_trees"]]
    scores = MLReturnFactor(mode=mode, models=fitted).compute(prices, panel)
    assert list(scores.columns) == list(prices.columns)
    assert scores.iloc[0].equals(scores.iloc[-1])


def test_missing_persisted_model_raises() -> None:
    with pytest.raises(FileNotFoundError):
        MLReturnFactor(mode="single", models=["does_not_exist"], model_dir="/nope")


def test_predict_panel_returns_series() -> None:
    panel = synthetic_fundamentals(list(range(1000, 1030)))
    fitted = [build_estimator("extra_trees").fit(panel.fillna(0.0).to_numpy(), _target(panel))]
    s = MLReturnFactor.predict_panel(panel, models=fitted)
    assert isinstance(s, pd.Series) and s.index.name == "security_id" and len(s) == 30
