"""ML-return factor: predict annual stock return from fundamental ratios.

This is the framework-native port of the "Build Your Own AI Investor"
``Machine_Learning.ipynb`` study. The notebook trains 8 sklearn regressors to
predict a stock's annual return from 18 fundamental ratios and shows that the
*tree ensembles* (RandomForest / ExtraTrees / GradientBoosting) give the most
consistent top-/bottom-10 separation.

Design
------
* A :class:`MLReturnFactor` is a normal :class:`~analytics.factors.factors.Factor`:
  ``compute(prices, fundamentals)`` returns a cross-sectional score matrix
  (index = date, columns = security_id) of **predicted annual return**, ready to
  be ranked and stored by ``compute_and_store_factors``.
* The "model" is a *collection policy* that turns a fundamentals panel into
  scores. We faithfully support all 8 of the notebook's regressors and three
  combination modes:
    - ``single`` : one named model (e.g. ``gradient_boosting``)
    - ``ensemble`` : average of several fitted pipelines (default; matches the
      notebook's "try all, keep the promising" intent)
    - ``rank_vote`` : average of cross-sectional percentile ranks (robust to
      scale/calibration drift between models)
* Pipelines are fitted by :mod:`analytics.factors.ml_training` and persisted to
  ``analytics/factors/models/`` as joblib files, then loaded here.
* Pure-prediction mode (no persisted model) is supported via ``estimator`` /
  ``build_estimator`` for quick experimentation and tests, so the factor can run
  with an in-memory sklearn pipeline and no DB access.

Vendor-agnosticism: the factor never touches a vendor. It only consumes a wide
ratio panel (index = security_id, columns = RATIO_COLUMNS) produced by a
:class:`~analytics.factors.fundamentals.FundamentalsProvider`. Whatever vendor
loaded those ratios is irrelevant to scoring.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame

from .factors import Factor
from .fundamentals import RATIO_COLUMNS

# Persisted pipelines live here (written by ml_training.py).
_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# Canonical model keys in the order the notebook introduces them.
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

# Default ensemble = the three tree models the notebook found most consistent.
DEFAULT_ENSEMBLE = ["random_forest", "extra_trees", "gradient_boosting"]


# ---------------------------------------------------------------------------
# Estimator construction (faithful to the notebook's pipelines)
# ---------------------------------------------------------------------------


def build_estimator(key: str) -> Any:
    """Return a fresh, un-fitted sklearn estimator matching the notebook.

    Each pipeline mirrors the corresponding notebook cell:
      * linear        : PowerTransformer + LinearRegression
      * elastic_net   : PowerTransformer + ElasticNet (default alpha/l1_ratio)
      * knn           : PowerTransformer + KNeighborsRegressor(n_neighbors=40)
      * svm           : PowerTransformer + SVR (rbf, C=100, gamma=0.1, eps=0.1)
      * decision_tree : DecisionTreeRegressor(max_depth=15)
      * random_forest : RandomForestRegressor(max_depth=10, n_estimators=100)
      * extra_trees   : ExtraTreesRegressor(max_depth=10, n_estimators=100)
      * gradient_boosting : GradientBoostingRegressor(max_depth=10)
    """
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import ElasticNet, LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    key = key.lower()
    if key == "linear":
        return Pipeline([("scaler", StandardScaler()), ("linear", LinearRegression())])
    if key == "elastic_net":
        return Pipeline([("scaler", StandardScaler()), ("ElasticNet", ElasticNet())])
    if key == "knn":
        return Pipeline([("scaler", StandardScaler()), ("KNeighborsRegressor", KNeighborsRegressor(n_neighbors=40))])
    if key == "svm":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("SVR", SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)),
            ]
        )
    if key == "decision_tree":
        return DecisionTreeRegressor(random_state=42, max_depth=15)
    if key == "random_forest":
        return RandomForestRegressor(random_state=42, max_depth=10, n_estimators=100)
    if key == "extra_trees":
        return ExtraTreesRegressor(random_state=42, max_depth=10, n_estimators=100)
    if key == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42, loss="squared_error")
    raise ValueError(f"Unknown model key '{key}'. Available: {MODEL_KEYS}")


def _is_classifier(obj: Any) -> bool:
    from sklearn.base import is_classifier

    return bool(is_classifier(obj))


def _as_pipelines(items: list[Any]) -> list[Any]:
    """Normalise (key, estimator) pairs to estimators, building from str keys."""
    out = []
    for it in items:
        if isinstance(it, str):
            out.append(build_estimator(it))
        else:
            out.append(it)
    return out


# ---------------------------------------------------------------------------
# Factor
# ---------------------------------------------------------------------------


class MLReturnFactor(Factor):
    """Predict annual return from fundamental ratios -> cross-sectional signal.

    Args:
        mode: ``"single"``, ``"ensemble"`` or ``"rank_vote"``.
        models: model keys / fitted estimators to use. For ``single``, the first
            item is used. For ``ensemble``/``rank_vote`` all are combined.
        model_dir: directory holding persisted ``.joblib`` pipelines. If a model
            key has no in-memory estimator and a ``<key>.joblib`` file exists,
            it is loaded from here.
        impute: how to fill NaN ratios before prediction. ``"median"`` (default)
            imputes per-column median; ``"zero"`` fills 0 (only safe after a
            PowerTransformer); ``None`` leaves NaN (some models tolerate it,
            most do not).
        name: factor name stored in ``factor_scores`` (default ``"ml_return"``).
    """

    name = "ml_return"

    def __init__(
        self,
        mode: str = "ensemble",
        models: Optional[Sequence[Any]] = None,
        model_dir: str = _MODELS_DIR,
        impute: Optional[str] = "median",
        name: str = "ml_return",
        fundamentals: Optional[pd.DataFrame] = None,
    ):
        """Initialize the ML-return factor.

        Args:
            mode: Combination mode (``single``, ``ensemble`` or ``rank_vote``).
            models: Optional list of model keys to combine.
            model_dir: Directory holding the persisted joblib pipelines.
            impute: Strategy for imputing missing fundamentals (default ``median``).
            name: Factor name.
            fundamentals: Optional in-memory fundamentals frame for pure-prediction mode.
        """
        self.mode = mode.lower()
        self.model_dir = model_dir
        self.impute = impute
        self.name = name
        # Optional fundamentals panel (index=security_id, cols=RATIO_COLUMNS).
        # When set, compute(prices) can be called with a single argument (as the
        # generic compute_and_store_factors helper expects) and still score.
        self._fundamentals = fundamentals

        if models is None:
            models = DEFAULT_ENSEMBLE if self.mode != "single" else ["gradient_boosting"]

        self._estimators = self._resolve_estimators(list(models))
        if not self._estimators:
            raise ValueError("MLReturnFactor requires at least one model estimator.")
        if self.mode == "single" and len(self._estimators) > 1:
            self._estimators = self._estimators[:1]

    # -- construction helpers ----------------------------------------------

    def _resolve_estimators(self, models: list[Any]) -> list[tuple[str, Any]]:
        """Resolve a model spec into (label, fitted_estimator) pairs.

        Each item is either:
          * a fitted sklearn estimator (used as-is), or
          * a string key, which must resolve to a persisted ``.joblib`` in
            ``model_dir``. A string key with no persisted model is a hard error
            (we never silently fall back to an *unfitted* estimator, which would
            corrupt scores with NaN).
        """
        estimators = []
        for m in models:
            if isinstance(m, str):
                est = self._load_persisted(m)
            else:
                est = m
            estimators.append((m if isinstance(m, str) else getattr(m, "__class__").__name__, est))
        return estimators

    def _load_persisted(self, key: str) -> Any:
        import joblib

        path = os.path.join(self.model_dir, f"{key}.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No persisted model '{key}' at {path}. Train it first via "
                f"analytics.factors.ml_training.train_models, pass a fitted "
                f"estimator directly, or set model_dir correctly."
            )
        return joblib.load(path)

    # -- core --------------------------------------------------------------

    def _impute(self, panel: DataFrame) -> DataFrame:
        if self.impute == "median":
            return panel.fillna(panel.median())
        if self.impute == "zero":
            return panel.fillna(0.0)
        if self.impute is None:
            return panel
        raise ValueError(f"Unknown impute method '{self.impute}'.")

    def compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Return predicted-return scores (index=date, columns=security_id).

        ``fundamentals`` must be a wide ratio panel indexed by ``security_id``
        with columns = ``RATIO_COLUMNS``. It is broadcast across all price dates
        (the ML model predicts a *current* forward return from *current*
        fundamentals, so the score is stable across the holding period - which is
        exactly how the notebook uses a single fitted model to score new rows).

        For a time-varying backtest (e.g. quarterly fundamentals refreshed across
        the sample), call :meth:`compute_dynamic` instead.

        If ``fundamentals`` is None we fall back to any panel supplied at
        construction; if still None we have nothing to score and return an
        all-NaN frame aligned to ``prices``.
        """
        if fundamentals is None:
            fundamentals = self._fundamentals
        if fundamentals is None:
            return pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

        # Single static panel: score once, broadcast across all dates.
        scores = self._score_panel(fundamentals)
        out = pd.Series(scores, index=fundamentals.index)
        out_aligned = out.reindex(prices.columns)
        return pd.DataFrame(
            np.tile(out_aligned.to_numpy(dtype=float), (len(prices.index), 1)),
            index=prices.index,
            columns=prices.columns,
        )

    def compute_dynamic(
        self,
        prices: pd.DataFrame,
        fundamentals_fn: Callable[[Any], pd.DataFrame],
    ) -> pd.DataFrame:
        """Score with a *different* fundamentals panel at each date.

        Args:
            prices: date x security_id price panel.
            fundamentals_fn: callable ``f(date) -> DataFrame`` returning a wide
                ratio panel (index=security_id, cols=RATIO_COLUMNS) of fundamentals
                *available as of that date* (point-in-time). This is what makes the
                ML value signal refresh (e.g. quarterly) without look-ahead: the
                notebook passes a provider that only returns fundamentals with
                ``effective_date <= date``.

        Returns:
            date x security_id score frame where each row is scored from that
            date's own fundamentals.
        """
        scores_by_date = []
        for dt in prices.index:
            panel = fundamentals_fn(dt)
            if panel is None or len(panel) == 0:
                scores_by_date.append(pd.Series(np.nan, index=prices.columns, name=dt))
                continue
            sc = self._score_panel(panel)
            s = pd.Series(sc, index=panel.index).reindex(prices.columns)
            s.name = dt
            scores_by_date.append(s)
        return pd.concat(scores_by_date, axis=1).T

    # -- core scoring -------------------------------------------------------

    def _score_panel(self, panel: DataFrame) -> np.ndarray:
        """Predict a single cross-section of fundamentals -> 1-D array of scores.

        ``panel`` is a wide ratio panel indexed by security_id with columns =
        RATIO_COLUMNS. Returns an array aligned to ``panel.index``.
        """
        # Keep the exact 18-column contract the pipelines were trained on:
        # reindex to RATIO_COLUMNS, impute partial-NaN, then zero-fill any
        # *fully* NaN column so we never change the feature count (which would
        # break a fitted pipeline's predict). Predict on the array to avoid
        # sklearn's "fitted without feature names" warning.
        panel = self._impute(panel.reindex(columns=RATIO_COLUMNS))
        panel = panel.fillna(0.0)
        if panel.shape[1] == 0:
            return np.full(len(panel), np.nan)
        if panel.shape[0] == 0:
            return np.array([])

        X = panel.to_numpy(dtype=float)
        preds = []
        for _key, est in self._estimators:
            try:
                p = np.asarray(est.predict(X), dtype=float)
            except Exception as exc:  # noqa: BLE001 - one bad model must not kill scoring
                print(f"[ml_return] model predict failed: {exc}")
                p = np.full(len(panel), np.nan)
            preds.append(p)

        stacked = np.vstack(preds)  # (n_models, n_securities)

        if self.mode == "rank_vote":
            # Cross-sectional percentile rank per model, then average.
            out = np.zeros(stacked.shape[1])
            for row in stacked:
                out = out + pd.Series(row).rank(pct=True).to_numpy()
            out = out / len(stacked)
        else:  # single / ensemble -> average predicted returns
            out = np.nanmean(stacked, axis=0)
        return out

    # -- convenience API used by the demo notebook --------------------------

    @classmethod
    def predict_panel(
        cls,
        fundamentals: pd.DataFrame,
        mode: str = "ensemble",
        models: Optional[Sequence[Any]] = None,
        impute: Optional[str] = "median",
        model_dir: str = _MODELS_DIR,
    ) -> pd.Series:
        """One-shot prediction for a single fundamentals panel (no prices).

        Returns a Series indexed by security_id of predicted annual return
        (or averaged rank for ``rank_vote``).
        """
        # Build a throwaway factor and reuse compute() with a 1-row price index.
        fac = cls(mode=mode, models=models, impute=impute, model_dir=model_dir)
        dummy_prices = pd.DataFrame(index=[pd.Timestamp("2000-01-01")], columns=fundamentals.index)
        scores = fac.compute(dummy_prices, fundamentals)
        return scores.iloc[0]
