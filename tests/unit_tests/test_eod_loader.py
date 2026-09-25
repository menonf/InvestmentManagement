"""Batch / retry orchestration of the EOD loader with a fake vendor (no network, no DB)."""

import pandas as pd

from data_engineering.loaders.eod_loader import EodLoadSummary, fetch_in_batches


def _universe(n: int) -> pd.DataFrame:
    return pd.DataFrame({"security_id": range(1, n + 1), "symbol": [f"S{i}" for i in range(1, n + 1)], "ric": [f"S{i}.O" for i in range(1, n + 1)]})


def _rows(sec_ids, n_days: int = 3) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    return pd.DataFrame([{"security_id": s, "as_of_date": d, "adj_close": 100.0 + s} for s in sec_ids for d in dates])


def test_retries_only_missing_securities() -> None:
    calls = []

    def flaky(symbol_df, start, end):
        ids = list(symbol_df["security_id"])
        calls.append(ids)
        # first call of every batch drops the last security; the retry returns it
        if len(calls) % 2 == 1:
            ids = ids[:-1]
        return _rows(ids)

    frame, missing = fetch_in_batches(_universe(6), flaky, "2024-01-01", "2024-01-05", batch_size=3, max_retries=3, retry_backoff_seconds=0, timeout_seconds=None)
    assert missing == []
    assert sorted(frame["security_id"].unique()) == [1, 2, 3, 4, 5, 6]
    assert calls[1] == [3] and calls[3] == [6]  # retries requested only the missing id
    assert not frame.duplicated(["security_id", "as_of_date"]).any()


def test_reports_missing_after_exhausting_retries() -> None:
    def never_returns_two(symbol_df, start, end):
        return _rows([s for s in symbol_df["security_id"] if s != 2])

    frame, missing = fetch_in_batches(_universe(3), never_returns_two, "2024-01-01", "2024-01-05", batch_size=3, max_retries=2, retry_backoff_seconds=0, timeout_seconds=None)
    assert missing == [2]
    assert set(frame["security_id"]) == {1, 3}


def test_vendor_exception_is_survived() -> None:
    def boom(symbol_df, start, end):
        raise RuntimeError("gateway timeout")

    frame, missing = fetch_in_batches(_universe(2), boom, "2024-01-01", "2024-01-05", batch_size=2, max_retries=2, retry_backoff_seconds=0, timeout_seconds=5)
    assert frame.empty and missing == [1, 2]


def test_on_batch_callback_and_summary_coverage() -> None:
    written = []
    fetch_in_batches(_universe(4), lambda df, s, e: _rows(list(df["security_id"])), "2024-01-01", "2024-01-05", batch_size=2, timeout_seconds=None, on_batch=written.append)
    assert len(written) == 2
    summary = EodLoadSummary(rows_written=10, securities_requested=4, securities_loaded=3)
    assert summary.coverage == 0.75
