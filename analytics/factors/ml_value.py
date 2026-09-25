"""ML value factor: predict forward stock return from fundamental ratios.

Framework port of the "Build Your Own AI Investor" study. Eight scikit-learn
regressors are trained (see :mod:`.ml_training`) on the 18 ratios in
``RATIO_COLUMNS`` to predict a forward return; the tree ensembles proved the
most consistent, so the default is an average of RandomForest / ExtraTrees /
GradientBoosting predictions.

The factor is a :class:`~analytics.factors.base.PanelFactor`: it scores one
fundamentals panel at a time and the base class handles broadcasting or
point-in-time refreshing across dates. Fitted pipelines are persisted as
joblib files and loaded by model key, or passed in already fitted.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

import numpy as np
from pandas import DataFrame, Series

from data_engineering.fundamentals.ratios import RATIO_COLUMNS

from .base import PanelFactor

log = logging.getLogger(__name__)

#: Default location of persisted pipelines (written by ``ml_training``).
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

#: Canonical model keys in the order the study introduces them.
MODEL_KEYS = (
    "linear",
    "elastic_net",
    "knn",
    "svm",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
)

#: The three tree ensembles that gave the most consistent top/bottom separation.
DEFAULT_ENSEMBLE = ["random_forest", "extra_trees", "gradient_boosting"]


def build_estimator(key: str) -> Any:
    """Return a fresh, un-fitted scikit-learn estimator for ``key``.

    Pipelines mirror the study (scaler + model where the study scaled):
      linear / elastic_net / knn(40) / svm(rbf, C=100) inside a StandardScaler;
      decision_tree(depth 15); random_forest, extra_trees (depth 10, 100 trees);
      gradient_boosting (depth 10, 100 trees, lr 0.1).
    """
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    key = key.lower()
    scaled = {
        "linear": LinearRegression(),
        "elastic_net": ElasticNet(),
        "knn": KNeighborsRegressor(n_neighbors=40),
        "svm": SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1),
    }
    trees = {
        "decision_tree": lambda: DecisionTreeRegressor(random_state=42, max_depth=15),
        "random_forest": lambda: RandomForestRegressor(random_state=42, max_depth=10, n_estimators=100),
        "extra_trees": lambda: ExtraTreesRegressor(random_state=42, max_depth=10, n_estimators=100),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42, loss="squared_error"
        ),
    }
    if key in scaled:
        return Pipeline([("scaler", StandardScaler()), (key, scaled[key])])
    if key in trees:
        return trees[key]()
    raise ValueError(f"Unknown model key '{key}'. Available: {MODEL_KEYS}")


def load_estimator(key: str, model_dir: str = MODELS_DIR) -> Any:
    """Load a persisted ``<key>.joblib`` pipeline; raise if it has not been trained."""
    import joblib

    path = os.path.join(model_dir, f"{key}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No persisted model '{key}' at {path}. Train it with "
            f"analytics.factors.ml_training.train_models, pass a fitted estimator, or fix model_dir."
        )
    return joblib.load(path)


class MLReturnFactor(PanelFactor):
    """Predicted forward return from fundamental ratios as a cross-sectional signal.

    Args:
        mode: ``"single"`` (first model only), ``"ensemble"`` (average predicted
            return; default) or ``"rank_vote"`` (average of per-model percentile
            ranks - robust to calibration drift between models).
        models: model keys (loaded from ``model_dir``) and/or fitted estimators.
            A key with no persisted file is a hard error: an *unfitted* model
            would silently produce garbage scores.
        model_dir: directory holding ``<key>.joblib`` files.
        impute: ``"median"`` (per-column median of the panel), ``"zero"`` or
            ``None`` for missing ratios. Fully-missing columns are zero-filled
            so the 18-feature contract is always honoured.
        name: factor name used in outputs.
        fundamentals: optional static panel for single-argument ``compute``.
    """

    name = "ml_return"

    def __init__(
        self,
        mode: str = "ensemble",
        models: Optional[Sequence[Any]] = None,
        model_dir: str = MODELS_DIR,
        impute: Optional[str] = "median",
        score_missing: Optional[bool] = None,
        name: str = "ml_return",
        fundamentals: Optional[DataFrame] = None,
    ):
        """Resolve the model list into fitted estimators.

        ``score_missing`` is a convenience alias for ``impute``: pass ``False``
        to leave names whose every ratio is missing as NaN (do not impute them
        to the panel median / zero). When ``score_missing`` is given it takes
        precedence over ``impute`` (``False`` -> impute ``None``, ``True`` ->
        impute ``"median"``).
        """
        super().__init__(fundamentals)
        self.mode = mode.lower()
        if self.mode not in ("single", "ensemble", "rank_vote"):
            raise ValueError(f"unknown mode '{mode}'")
        self.model_dir = model_dir
        if score_missing is not None:
            impute = None if not score_missing else "median"
        self.impute = impute
        self.name = name
        if models is None:
            models = DEFAULT_ENSEMBLE if self.mode != "single" else ["gradient_boosting"]
        self._estimators: list[tuple[str, Any]] = [
            (m, load_estimator(m, model_dir)) if isinstance(m, str) else (type(m).__name__, m) for m in models
        ]
        if not self._estimators:
            raise ValueError("MLReturnFactor requires at least one model estimator.")
        if self.mode == "single":
            self._estimators = self._estimators[:1]

    @property
    def estimators(self) -> list[tuple[str, Any]]:
        """``(label, fitted_estimator)`` pairs in use."""
        return list(self._estimators)

    # -- scoring -------------------------------------------------------------

    def _impute(self, panel: DataFrame) -> DataFrame:
        if self.impute == "median":
            return panel.fillna(panel.median())
        if self.impute == "zero":
            return panel.fillna(0.0)
        if self.impute is None:
            return panel
        raise ValueError(f"Unknown impute method '{self.impute}'.")

    def score_panel(self, panel: DataFrame) -> Series:
        """Predict one cross-section of ratios -> ``security_id -> score``."""
        features = self._impute(panel.reindex(columns=RATIO_COLUMNS))
        if features.shape[0] == 0:
            return Series(dtype=float)
        if self.impute is None:
            # leave rows whose every ratio is missing as NaN rather than 0
            features = features.where(features.notna().any(axis=1), np.nan).fillna(0.0)
        else:
            features = features.fillna(0.0)
        X = features.to_numpy(dtype=float)
        preds = []
        for label, est in self._estimators:
            try:
                preds.append(np.asarray(est.predict(X), dtype=float))
            except Exception as exc:  # noqa: BLE001 - one bad model must not kill scoring
                log.warning("%s predict failed: %s", label, exc)
                preds.append(np.full(len(features), np.nan))
        stacked = np.vstack(preds)
        if self.mode == "rank_vote":
            ranks = [Series(row).rank(pct=True).to_numpy() for row in stacked]
            out = np.nanmean(np.vstack(ranks), axis=0)
        else:
            out = np.nanmean(stacked, axis=0)
        return Series(out, index=panel.index, dtype=float)

    # -- convenience -----------------------------------------------------------

    @classmethod
    def predict_panel(
        cls,
        fundamentals: DataFrame,
        mode: str = "ensemble",
        models: Optional[Sequence[Any]] = None,
        impute: Optional[str] = "median",
        model_dir: str = MODELS_DIR,
    ) -> Series:
        """One-shot prediction for a single panel: ``security_id -> predicted return``."""
        factor = cls(mode=mode, models=models, impute=impute, model_dir=model_dir)
        scores = factor.score_panel(fundamentals)
        scores.index.name = "security_id"
        return scores
