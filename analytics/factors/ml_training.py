"""Train and persist the ML value factor pipelines from database history.

Pipeline
--------
1. :func:`build_modelling_table` assembles ``(X, y)`` rows from the database:
   ``X`` = point-in-time ratio panel at each monthly snapshot date, ``y`` =
   forward ``forward_months`` total return from ``market_data``.
2. :func:`time_based_split` cuts on snapshot *dates* (never rows or randomly),
   so every training snapshot precedes every test snapshot - no look-ahead.
3. :func:`train_models` fits the requested estimators, reports train/test MSE
   and persists each pipeline as ``<model_dir>/<key>.joblib``.

Options worth knowing:
- ``demean_target=True`` subtracts the cross-sectional mean return at each
  snapshot so the model learns *relative* (stock-selection) return rather than
  the market's level, which is what a long/short factor actually trades.
- ``embargo_months`` in :func:`time_based_split` drops the snapshots whose
  forward window straddles the split date; with a 12-month label this stops
  the last training labels from overlapping the first test period.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from pandas import DataFrame

from data_engineering.fundamentals import RATIO_COLUMNS, FundamentalsProvider, get_fundamentals_provider

from .ml_value import DEFAULT_ENSEMBLE, MODEL_KEYS, MODELS_DIR, build_estimator

log = logging.getLogger(__name__)

_MODELS_DIR = MODELS_DIR  # backward-compatible name


def models_available(model_dir: str = MODELS_DIR, keys: Sequence[str] = DEFAULT_ENSEMBLE) -> bool:
    """Return True when every ``<key>.joblib`` in ``keys`` is persisted under ``model_dir``."""
    if not os.path.isdir(model_dir):
        return False
    return all(os.path.isfile(os.path.join(model_dir, f"{k}.joblib")) for k in keys)


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def forward_return_labels(
    prices: DataFrame, snapshot_dates: pd.DatetimeIndex, forward_months: int, min_history: int = 30
) -> DataFrame:
    """Forward total return per security from each snapshot date (pure pandas).

    For each snapshot the actual last trading day on/before it is the start,
    and the first trading day on/after ``start + forward_months`` is the end.
    Returns a long frame ``[snapshot_date, security_id, y]`` where
    ``snapshot_date`` is the *actual* trading day used (so it joins exactly
    with the fundamentals panel pulled for that day).
    """
    px = prices.sort_index()
    idx = pd.DatetimeIndex(px.index)
    rows = []
    for sd in snapshot_dates:
        prior = idx[idx <= sd]
        if len(prior) == 0:
            continue
        t0 = prior[-1]
        future = idx[idx >= t0 + pd.DateOffset(months=forward_months)]
        if len(future) == 0:
            continue
        window = px.loc[t0 : future[0]]
        if window.shape[0] < min_history:
            continue
        ret = (window.iloc[-1] / window.iloc[0] - 1.0).dropna()
        rows.append(DataFrame({"snapshot_date": t0, "security_id": ret.index, "y": ret.to_numpy()}))
    return pd.concat(rows, ignore_index=True) if rows else DataFrame(columns=["snapshot_date", "security_id", "y"])


def build_modelling_table(
    orm_session: Any,
    orm_engine: Any,
    start_date: str,
    end_date: str,
    forward_months: int = 12,
    min_history: int = 30,
    fundamentals_provider: Optional[FundamentalsProvider] = None,
    demean_target: bool = False,
    impute_median: bool = True,
) -> DataFrame:
    """Assemble the ``(X, y)`` modelling table from the database.

    Args:
        start_date / end_date: snapshot window ``YYYY-MM-DD`` (monthly snapshots,
            snapped to the last trading day of each month).
        forward_months: label horizon.
        min_history: minimum price observations in the forward window.
        fundamentals_provider: source of point-in-time ratios (default: the
            database provider over ``dbo.security_fundamentals``).
        demean_target: subtract the cross-sectional mean of ``y`` per snapshot.
        impute_median: fill missing ratios with the column median (else drop rows
            with any missing ratio).

    Returns:
        Long frame ``[security_id, snapshot_date, y, *RATIO_COLUMNS]``.
    """
    from data_engineering.database import database as db

    provider = fundamentals_provider or get_fundamentals_provider("static", orm_session=orm_session, orm_engine=orm_engine)

    market = db.read_market_data(orm_session, orm_engine, start_date, end_date)
    if market.empty:
        raise RuntimeError("No market_data in the database for the date window.")
    market["as_of_date"] = pd.to_datetime(market["as_of_date"])
    prices = market.pivot_table(index="as_of_date", columns="security_id", values="adj_close").sort_index()

    snapshots = pd.date_range(start_date, end_date, freq="ME")
    labels = forward_return_labels(prices, snapshots, forward_months, min_history)
    if labels.empty:
        raise RuntimeError("Could not compute any forward-return labels.")

    universe = db.read_security_master(orm_session, orm_engine)[["security_id", "name"]].rename(columns={"name": "symbol"})
    panels = []
    for t0 in sorted(labels["snapshot_date"].unique()):
        panel = provider.get_panel(universe, pd.Timestamp(t0).strftime("%Y-%m-%d"))
        if panel is None or panel.dropna(how="all").empty:
            continue
        panel = panel.dropna(how="all").reset_index()
        panel["snapshot_date"] = pd.Timestamp(t0)
        panels.append(panel)
    if not panels:
        raise RuntimeError("No fundamentals found in the database for the date window.")

    X = pd.concat(panels, ignore_index=True)
    X["snapshot_date"] = pd.to_datetime(X["snapshot_date"])
    labels["snapshot_date"] = pd.to_datetime(labels["snapshot_date"])
    table = X.merge(labels, on=["security_id", "snapshot_date"], how="inner").dropna(subset=["y"])

    if demean_target:
        table["y"] = table["y"] - table.groupby("snapshot_date")["y"].transform("mean")
    if impute_median:
        for col in RATIO_COLUMNS:
            if col in table.columns and table[col].isna().any():
                med = table[col].median()
                table[col] = table[col].fillna(med if pd.notna(med) else 0.0)
    else:
        table = table.dropna(subset=list(RATIO_COLUMNS))
    return table.sort_values(["snapshot_date", "security_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Splitting & training
# ---------------------------------------------------------------------------


def time_based_split(df: DataFrame, test_size: float = 0.2, embargo_months: int = 0) -> tuple[DataFrame, DataFrame, Any, Any, Any]:
    """Split on sorted unique ``snapshot_date`` so no date straddles the cut.

    All training snapshots are strictly before ``split_date``; test snapshots
    are on/after it. With ``embargo_months > 0`` the training snapshots within
    that many months before the split are dropped so their forward-return
    labels cannot overlap the test window.

    Returns ``(X_train, X_test, y_train, y_test, split_date)``.
    """
    df = df.sort_values("snapshot_date")
    dates = np.sort(df["snapshot_date"].unique())
    cut = int(len(dates) * (1 - test_size))
    cut = max(1, min(cut, len(dates) - 1))
    split_date = dates[cut]
    train = df[df["snapshot_date"] < split_date]
    if embargo_months:
        train = train[train["snapshot_date"] < pd.Timestamp(split_date) - pd.DateOffset(months=embargo_months)]
    test = df[df["snapshot_date"] >= split_date]
    feats = list(RATIO_COLUMNS)
    return train[feats], test[feats], train["y"], test["y"], split_date


def train_models(
    df: DataFrame,
    model_keys: Sequence[str] = MODEL_KEYS,
    test_size: float = 0.2,
    model_dir: str = MODELS_DIR,
    verbose: bool = True,
    embargo_months: int = 0,
) -> dict[str, Any]:
    """Fit each requested model on the time-based split and persist it to ``model_dir``.

    Returns ``{key: {"train_mse": ..., "test_mse": ...} | {"error": ...}}``.
    """
    import joblib
    from sklearn.metrics import mean_squared_error

    os.makedirs(model_dir, exist_ok=True)
    X_train, X_test, y_train, y_test, split_date = time_based_split(df, test_size, embargo_months)
    if verbose:
        log.info("training on %d rows, testing on %d rows (split %s)", len(X_train), len(X_test), pd.Timestamp(split_date).date())

    # Fit on arrays so persisted pipelines carry no feature-name expectation.
    Xtr, ytr = X_train.to_numpy(dtype=float), y_train.to_numpy(dtype=float)
    Xte, yte = X_test.to_numpy(dtype=float), y_test.to_numpy(dtype=float)
    metrics: dict[str, Any] = {}
    for key in model_keys:
        est = build_estimator(key)
        try:
            est.fit(Xtr, ytr)
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            log.warning("%s fit failed: %s", key, exc)
            metrics[key] = {"error": str(exc)}
            continue
        metrics[key] = {
            "train_mse": float(mean_squared_error(ytr, est.predict(Xtr))),
            "test_mse": float(mean_squared_error(yte, est.predict(Xte))) if len(yte) else float("nan"),
        }
        if verbose:
            log.info("  %-18s train_mse=%.4f  test_mse=%.4f", key, metrics[key]["train_mse"], metrics[key]["test_mse"])
        joblib.dump(est, os.path.join(model_dir, f"{key}.joblib"))
    return metrics


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def observe_prediction_ability(factor: Any, fundamentals: DataFrame, prices: DataFrame, n: int = 10) -> DataFrame:
    """Top-``n`` / bottom-``n`` predicted vs realised return on a held-out panel.

    ``prices`` must cover the realisation window: realised return is
    ``last / first - 1`` per security. Returns a one-row summary frame.
    """
    predicted = factor.score_panel(fundamentals).dropna().sort_values(ascending=False)
    realised = (prices.iloc[-1] / prices.iloc[0] - 1.0).reindex(predicted.index)
    top, bottom = predicted.index[:n], predicted.index[-n:]
    return DataFrame(
        [
            {
                "top_n_predicted": float(predicted.loc[top].mean()),
                "top_n_realised": float(realised.loc[top].mean()),
                "bottom_n_predicted": float(predicted.loc[bottom].mean()),
                "bottom_n_realised": float(realised.loc[bottom].mean()),
                "spread_realised": float(realised.loc[top].mean() - realised.loc[bottom].mean()),
                "n": int(n),
            }
        ]
    )


def build_modelling_table_from_panel(
    prices: DataFrame,
    provider: Any,
    all_symbols: DataFrame,
    start_date: str,
    end_date: str,
    forward_months: int = 12,
    min_history: int = 30,
    demean_target: bool = False,
    impute_median: bool = True,
) -> DataFrame:
    """Panel-only twin of :func:`build_modelling_table` (no database session).

    Walks monthly snapshots between ``start_date`` and ``end_date``, pulls the
    point-in-time ratio panel from ``provider.get_panel`` for each, attaches the
    forward return (via :func:`forward_return_labels`) and stacks the rows into
    the same ``[security_id, snapshot_date, y, *RATIO_COLUMNS]`` frame.

    ``provider`` may be any fundamentals provider exposing ``get_panel(symbols,
    as_of_date)`` returning a ``security_id``-indexed ratio frame.
    """
    px = prices.sort_index()
    snapshots = pd.date_range(start_date, end_date, freq="ME")
    labels = forward_return_labels(px, snapshots, forward_months, min_history)
    if labels.empty:
        raise RuntimeError("Could not compute any forward-return labels.")

    panels = []
    for t0 in sorted(labels["snapshot_date"].unique()):
        panel = provider.get_panel(all_symbols, pd.Timestamp(t0).strftime("%Y-%m-%d"))
        if panel is None or panel.dropna(how="all").empty:
            continue
        panel = panel.dropna(how="all").reset_index()
        if "security_id" not in panel.columns:
            continue
        panel["snapshot_date"] = pd.Timestamp(t0)
        panels.append(panel)
    if not panels:
        raise RuntimeError("No fundamentals found for the date window.")

    X = pd.concat(panels, ignore_index=True)
    X["snapshot_date"] = pd.to_datetime(X["snapshot_date"])
    labels["snapshot_date"] = pd.to_datetime(labels["snapshot_date"])
    table = X.merge(labels, on=["security_id", "snapshot_date"], how="inner").dropna(subset=["y"])

    if demean_target:
        table["y"] = table["y"] - table.groupby("snapshot_date")["y"].transform("mean")
    if impute_median:
        for col in RATIO_COLUMNS:
            if col in table.columns and table[col].isna().any():
                med = table[col].median()
                table[col] = table[col].fillna(med if pd.notna(med) else 0.0)
    else:
        table = table.dropna(subset=list(RATIO_COLUMNS))
    return table.sort_values(["snapshot_date", "security_id"]).reset_index(drop=True)


def save_training_metadata(model_dir: str, metadata: dict[str, Any]) -> None:
    """Persist training run metadata as JSON beside the ``.joblib`` models."""
    import json

    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "training_metadata.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)


def load_training_metadata(model_dir: str) -> Optional[dict[str, Any]]:
    """Load metadata written by :func:`save_training_metadata`, or ``None``."""
    import json

    path = os.path.join(model_dir, "training_metadata.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data
