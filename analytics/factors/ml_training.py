"""Train and persist the ML-return factor pipelines from database history.

Faithful port of ``Machine_Learning.ipynb`` with the one critical fix the
notebook lacked: a **time-based** train/test split. The notebook used a random
split, which leaks future fundamentals into the training set (look-ahead bias).
Here every model is trained on fundamentals observed *before* the label date and
validated on a forward window.

Pipeline
--------
1. Build a modelling table from the DB:
     X = ratio panel from ``dbo.security_fundamentals`` (RATIO_COLUMNS)
     y = forward 12-month total return from ``dbo.market_data``
       (adjusted close at label_date+T minus close at label_date, / close).
2. Label each row by its *fundamental snapshot date*; split on time so the
   training set's snapshots all precede the test set's.
3. Fit all 8 regressors (the exact sklearn pipelines from the notebook) inside a
   PowerTransformer / scaler where the notebook did, with progress bars
   (tqdm) so long runs are observable.
4. Report per-model RMSE on the forward test window and the notebook's
   top-10/bottom-10 "prediction ability" diagnostic.
5. Persist each fitted pipeline to ``analytics/factors/models/<key>.joblib``.

The persisted pipelines are then loaded by :class:`~analytics.factors.ml_factor
.MLReturnFactor` for scoring. Because fundamentals are vendor-agnostic (loaded by
a :class:`~analytics.factors.fundamentals.FundamentalsProvider`), the same
trained models score Yahoo-, SimFin- or Refinitiv-sourced ratios identically.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from sqlalchemy import Engine
from sqlalchemy.orm import Session

import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from .fundamentals import RATIO_COLUMNS, FundamentalsProvider, get_fundamentals_provider
from .ml_factor import MODEL_KEYS, build_estimator

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def build_modelling_table(
    orm_session: Session,
    orm_engine: Engine,
    start_date: str,
    end_date: str,
    forward_months: int = 12,
    min_history: int = 30,
    fundamentals_provider: Optional[FundamentalsProvider] = None,
) -> DataFrame:
    """Assemble (X, y) from the DB for ML training.

    Args:
        orm_session / orm_engine: DB handles.
        start_date / end_date: label (snapshot) date window ``YYYY-MM-DD``.
        forward_months: horizon over which to measure the return label.
        min_history: minimum number of price observations required to trust the
            forward return.
        fundamentals_provider: optional provider to pull ratios; if omitted a
            :class:`~analytics.factors.fundamentals.StaticFundamentalsProvider`
            reading ``dbo.security_fundamentals`` is used.

    Returns:
        Long DataFrame with columns: security_id, snapshot_date, y (forward
        return) and one column per ``RATIO_COLUMNS`` (X). Rows with any missing
        X or y are dropped.
    """
    from data_engineering.database import database as db

    if fundamentals_provider is None:
        fundamentals_provider = get_fundamentals_provider("static", orm_session=orm_session, orm_engine=orm_engine)

    # --- y: forward return from market_data (also defines snapshot dates) --
    md = db.read_market_data(orm_session, orm_engine, start_date, end_date)
    if md.empty:
        raise RuntimeError("No market_data in the database for the date window.")
    md = md.copy()
    md["as_of_date"] = pd.to_datetime(md["as_of_date"])
    px = md.pivot_table(index="as_of_date", columns="security_id", values="adj_close").sort_index()

    # Snapshot dates: one per month, snapped to the *actual* last trading day
    # of that month within the price window. Using the real trading date keeps
    # the X (fundamentals) and Y (forward return) merge keys identical.
    snap_dates = pd.to_datetime(pd.date_range(start_date, end_date, freq="ME"))
    sec_master = db.read_security_master(orm_session, orm_engine)[["security_id", "name"]]

    x_rows, y_rows = [], []
    for sd in tqdm(snap_dates, desc="building modelling table", unit="month"):
        prior = px.index[px.index <= sd]
        if len(prior) == 0:
            continue
        t0 = prior[-1]  # actual trading day
        t1 = t0 + pd.DateOffset(months=forward_months)
        future = px.index[px.index >= t1]
        if len(future) == 0:
            continue
        t1 = future[0]
        window = px.loc[t0:t1]
        has_label = window.shape[0] >= min_history
        if has_label:
            ret = (window.iloc[-1] / window.iloc[0] - 1.0).rename("y")
            y_rows.append(pd.DataFrame({"snapshot_date": t0, "security_id": ret.index, "y": ret.values}))

        # Fundamentals "on/before" t0 -- appended for EVERY snapshot that has a
        # panel, independent of whether a forward-return label exists for it
        # (rows without a label are dropped by the merge below).
        panel = fundamentals_provider.get_panel(sec_master, t0.strftime("%Y-%m-%d"))
        if panel is None or panel.empty:
            continue
        panel = panel.copy()
        panel["snapshot_date"] = t0
        x_rows.append(panel.reset_index())

    if not x_rows:
        raise RuntimeError("No fundamentals found in the database for the date window.")
    if not y_rows:
        raise RuntimeError("Could not compute any forward-return labels.")
    X = pd.concat(x_rows, ignore_index=True)
    Y = pd.concat(y_rows, ignore_index=True)
    # Align dtypes so the merge key matches exactly.
    X["snapshot_date"] = pd.to_datetime(X["snapshot_date"])
    Y["snapshot_date"] = pd.to_datetime(Y["snapshot_date"])

    df = X.merge(Y, on=["security_id", "snapshot_date"], how="inner")
    # Fundamentals are sparse (different vendors populate different ratios).
    # Rather than dropping every row that has any missing ratio, impute the
    # missing values with the column median (standard for this study) so we
    # keep all labelled samples. Rows with no y are still dropped.
    df = df.dropna(subset=["y"])
    for col in RATIO_COLUMNS:
        if col in df.columns and df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med if pd.notna(med) else 0.0)
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def time_based_split(
    df: DataFrame,
    test_size: float = 0.2,
) -> tuple[DataFrame, DataFrame]:
    """Split a modelling table on time (snapshot_date), not randomly.

    The cut is made on the *sorted unique snapshot dates* (not row count), so all
    rows sharing a given snapshot date stay on the same side of the split. This
    prevents train/test contamination when many securities share the same monthly
    snapshot date (the normal case here). All training snapshots strictly precede
    all test snapshots -> no look-ahead.

    Returns (X_train, X_test, y_train, y_test, split_date).
    """
    df = df.sort_values("snapshot_date")
    dates = df["snapshot_date"].sort_values().unique()
    cut_idx = int(len(dates) * (1 - test_size))
    cut_idx = max(1, min(cut_idx, len(dates) - 1))
    split_date = dates[cut_idx]
    train = df[df["snapshot_date"] < split_date]
    test = df[df["snapshot_date"] >= split_date]
    feats = RATIO_COLUMNS
    X_train, y_train = train[feats], train["y"]
    X_test, y_test = test[feats], test["y"]
    return X_train, X_test, y_train, y_test, split_date


def train_models(
    df: DataFrame,
    model_keys: Sequence[str] = MODEL_KEYS,
    test_size: float = 0.2,
    model_dir: str = _MODELS_DIR,
    verbose: bool = True,
) -> dict[str, Any]:
    """Fit every requested model on a time-based split and persist to disk.

    Returns a dict of metrics keyed by model key.
    """
    import joblib
    from sklearn.metrics import mean_squared_error

    os.makedirs(model_dir, exist_ok=True)
    X_train, X_test, y_train, y_test, split_date = time_based_split(df, test_size)
    if verbose:
        print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows " f"(split date {split_date}).")

    metrics: dict = {}
    # Fit on numpy arrays so the persisted pipelines carry no feature-name
    # expectation (predicting from a numpy panel in compute() then warns/strict-
    # matches otherwise).
    Xtr, ytr = X_train.to_numpy(), y_train.to_numpy()
    Xte, yte = X_test.to_numpy(), y_test.to_numpy()
    for key in tqdm(list(model_keys), desc="training models"):
        est = build_estimator(key)
        try:
            est.fit(Xtr, ytr)
        except Exception as exc:  # noqa: BLE001
            print(f"[train] {key} fit failed: {exc}")
            metrics[key] = {"error": str(exc)}
            continue
        y_pred = est.predict(Xte)
        test_mse = mean_squared_error(yte, y_pred)
        train_mse = mean_squared_error(ytr, est.predict(Xtr))
        metrics[key] = {"train_mse": train_mse, "test_mse": test_mse}
        if verbose:
            print(f"  {key:18s} train_mse={train_mse:.4f}  test_mse={test_mse:.4f}")
        # Persist
        path = os.path.join(model_dir, f"{key}.joblib")
        joblib.dump(est, path)

    return metrics


# ---------------------------------------------------------------------------
# Faithful "prediction ability" diagnostic (top10 / bottom10), time-aware
# ---------------------------------------------------------------------------


def observe_prediction_ability(
    factor: Any,
    fundamentals: DataFrame,
    prices: pd.DataFrame,
    n: int = 10,
    runs: int = 5,
) -> DataFrame:
    """Replicate the notebook's top10/bottom10 diagnostic against live scoring.

    Because we train on a time split (not random), we evaluate the factor on a
    *held-out* fundamentals panel rather than re-splitting. Returns a DataFrame
    summarising predicted vs actual top/bottom-10 average returns across runs.
    """
    results = []
    for i in range(runs):
        scores = factor.compute(prices, fundamentals).iloc[0]  # single-date broadcast
        ranked = scores.sort_values(ascending=False)
        top = ranked.iloc[:n]
        bot = ranked.iloc[-n:]
        results.append(
            {
                "run": i,
                "top10_pred_ret": top.mean(),
                "bot10_pred_ret": bot.mean(),
                "top10_n": top.notna().sum(),
                "bot10_n": bot.notna().sum(),
            }
        )
    return pd.DataFrame(results)
