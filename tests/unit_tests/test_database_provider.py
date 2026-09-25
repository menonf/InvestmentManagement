"""Point-in-time behaviour of the database-backed fundamentals provider (fake DB read)."""

import numpy as np
import pandas as pd
import pytest

from data_engineering.database import database as db
from data_engineering.fundamentals import RATIO_COLUMNS, StaticFundamentalsProvider


def _fake_rows() -> pd.DataFrame:
    rows = []
    for sec in (1, 2):
        for eff, pe in (("2024-03-31", 10.0 + sec), ("2024-06-30", 20.0 + sec), ("2024-09-30", 30.0 + sec)):
            rows.append({"security_id": sec, "metric_type": "P/E", "metric_value": pe, "source_vendor": "refinitiv", "effective_date": pd.Timestamp(eff).date(), "end_date": None})
    rows.append({"security_id": 1, "metric_type": "shares_outstanding", "metric_value": 5e6, "source_vendor": "refinitiv", "effective_date": pd.Timestamp("2024-01-01").date(), "end_date": None})
    rows.append({"security_id": 3, "metric_type": "P/E", "metric_value": 99.0, "source_vendor": "yahoo", "effective_date": pd.Timestamp("2024-01-01").date(), "end_date": None})
    return pd.DataFrame(rows)


def _make(monkeypatch_fn, lag=0, vendor="refinitiv"):
    monkeypatch_fn(db, "read_security_fundamentals", lambda *a, **k: _fake_rows())
    return StaticFundamentalsProvider(object(), object(), source_vendor=vendor, availability_lag_days=lag)


def _monkeypatch():
    saved = {}

    def setattr_(obj, name, value):
        saved[(obj, name)] = getattr(obj, name)
        setattr(obj, name, value)

    def undo():
        for (obj, name), val in saved.items():
            setattr(obj, name, val)

    return setattr_, undo


def test_only_rows_on_or_before_as_of_date_are_visible() -> None:
    patch, undo = _monkeypatch()
    try:
        p = _make(patch)
        symbols = pd.DataFrame({"security_id": [1, 2, 3], "symbol": ["a", "b", "c"]})
        panel = p.get_panel(symbols, "2024-07-15")
        assert list(panel.columns) == RATIO_COLUMNS
        assert panel.loc[1, "P/E"] == 21.0 and panel.loc[2, "P/E"] == 22.0  # June quarter, not September
        assert np.isnan(panel.loc[3, "P/E"])  # other vendor filtered out
        early = p.get_panel(symbols, "2024-01-15")
        assert early["P/E"].isna().all()  # nothing known yet -> NaN, never the latest value
    finally:
        undo()


def test_availability_lag_delays_visibility() -> None:
    patch, undo = _monkeypatch()
    try:
        p = _make(patch, lag=45)
        symbols = pd.DataFrame({"security_id": [1], "symbol": ["a"]})
        assert p.get_panel(symbols, "2024-07-15")["P/E"].iloc[0] == 11.0  # June filing not yet published
        assert p.get_panel(symbols, "2024-08-15")["P/E"].iloc[0] == 21.0
    finally:
        undo()


def test_history_and_cache_refresh() -> None:
    patch, undo = _monkeypatch()
    try:
        p = _make(patch)
        symbols = pd.DataFrame({"security_id": [1, 2], "symbol": ["a", "b"]})
        panel, history = p.get_panel_history(symbols, "2024-12-31", start_date="2024-06-01")
        assert history is not None
        assert set(pd.to_datetime(history["effective_date"]).dt.strftime("%Y-%m-%d")) == {"2024-06-30", "2024-09-30"}
        assert panel.loc[1, "P/E"] == 31.0
        calls = {"n": 0}

        def counting(*a, **k):
            calls["n"] += 1
            return _fake_rows()

        patch(db, "read_security_fundamentals", counting)
        p.get_panel(symbols, "2024-12-31")
        assert calls["n"] == 0  # cached
        p.refresh()
        p.get_panel(symbols, "2024-12-31")
        assert calls["n"] == 1
    finally:
        undo()


def test_in_memory_provider_is_point_in_time() -> None:
    from data_engineering.fundamentals import InMemoryFundamentalsProvider
    from tests.unit_tests.synthetic import synthetic_fundamentals_history

    history = synthetic_fundamentals_history([1, 2, 3], ["2024-03-31", "2024-06-30", "2024-09-30"])
    provider = InMemoryFundamentalsProvider(history, availability_lag_days=45)
    symbols = pd.DataFrame({"security_id": [1, 2, 3]})
    assert provider.get_panel(symbols, "2024-04-30")["P/E"].isna().all()  # March quarter not public until mid-May
    may = provider.get_panel(symbols, "2024-05-20")
    march_rows = history[history["effective_date"] == "2024-03-31"].set_index("security_id")
    assert np.allclose(may["P/E"].to_numpy(), march_rows["P/E"].to_numpy())
    _, hist = provider.get_panel_history(symbols, "2024-12-31", start_date="2024-06-01")
    assert set(hist["effective_date"].dt.strftime("%Y-%m-%d")) == {"2024-06-30", "2024-09-30"}
