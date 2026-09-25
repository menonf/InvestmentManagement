"""Batched, retried end-of-day price loading with data-quality guards.

The pull logic that used to live in the demo notebook (batch the universe,
retry only the securities a batch failed to return, guard against implausible
prints, write per batch, then mop up and prune holiday rows) is now a handful
of composable functions:

    universe = build_backtest_universe(...)
    summary  = load_eod_prices(universe, session, start, end)          # main pull
    summary2 = mop_up_missing_prices(universe, session, engine, start, end)  # gap recovery
    prune_low_coverage_holdings(engine, ["SP500", "SPY", "GSPC"], start, end)
    coverage_report(engine, "SP500")

``fetch_in_batches`` is vendor-agnostic and takes any callable of the shape
``fetch(symbol_df, start, end) -> DataFrame`` so it can be unit-tested with a
fake vendor.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import pandas as pd
from pandas import DataFrame

from data_engineering.database import database as db

log = logging.getLogger(__name__)

FetchFn = Callable[[DataFrame, str, str], DataFrame]


@dataclass
class EodLoadSummary:
    """Outcome of an EOD load."""

    rows_written: int = 0
    securities_requested: int = 0
    securities_loaded: int = 0
    dropped_implausible_rows: int = 0
    missing_security_ids: list[int] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of requested securities that returned at least one row."""
        return self.securities_loaded / self.securities_requested if self.securities_requested else 0.0


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def drop_implausible_moves(prices: DataFrame, threshold: float = 0.25, column: str = "adj_close") -> tuple[DataFrame, int]:
    """Remove rows whose day-over-day move in ``column`` exceeds ``threshold``.

    Split / reverse-split glitches, zero prices and bad ticks show up as a
    single impossible jump that would otherwise detonate a cap-weighted
    reconstruction. Only the offending row is dropped; the price level itself
    is never altered. Returns ``(clean_frame, n_dropped)``.
    """
    if prices.empty:
        return prices, 0
    ordered = prices.sort_values(["security_id", "as_of_date"])
    change = ordered.groupby("security_id")[column].pct_change().abs()
    bad = change > threshold
    return ordered[~bad].copy(), int(bad.sum())


def _run_with_timeout(fn: Callable[[], DataFrame], timeout: Optional[float]) -> tuple[Optional[DataFrame], str]:
    """Run ``fn`` in a worker thread so a hung vendor call cannot stall the load."""
    if timeout is None:
        return fn(), "ok"
    out: dict[str, DataFrame] = {}
    err: dict[str, BaseException] = {}

    def _target() -> None:
        try:
            out["df"] = fn()
        except BaseException as exc:  # noqa: BLE001 - surfaced to caller
            err["exc"] = exc

    worker = threading.Thread(target=_target, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        return None, "TIMEOUT"
    if "exc" in err:
        return None, f"ERR {type(err['exc']).__name__}: {str(err['exc'])[:80]}"
    return out.get("df"), "ok"


def fetch_in_batches(
    universe: DataFrame,
    fetch: FetchFn,
    start_date: str,
    end_date: str,
    *,
    batch_size: int = 20,
    max_retries: int = 5,
    retry_backoff_seconds: float = 20.0,
    timeout_seconds: Optional[float] = 900,
    on_batch: Optional[Callable[[DataFrame], None]] = None,
) -> tuple[DataFrame, list[int]]:
    """Fetch ``universe`` in batches, retrying only the securities each batch missed.

    Args:
        universe: ``security_id``, ``symbol`` (and optionally ``ric``) per row.
        fetch: ``fetch(symbol_df, start, end) -> standardized price frame``.
        batch_size: securities per vendor call (small batches come back complete).
        max_retries: attempts per batch; each retry re-requests only missing ids.
        retry_backoff_seconds: pause between retries of the same batch.
        timeout_seconds: per-call wall-clock guard (``None`` disables the thread).
        on_batch: optional callback receiving each batch's de-duplicated rows
            (used by :func:`load_eod_prices` to write incrementally).

    Returns:
        ``(all_rows, missing_security_ids)``.
    """
    rows = [r for _, r in universe.iterrows()]
    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
    log.info("EOD pull: %d securities in %d batches of %d", len(rows), len(batches), batch_size)

    frames: list[DataFrame] = []
    missing: list[int] = []
    for b, batch in enumerate(batches, start=1):
        by_id = {int(r["security_id"]): r for r in batch}
        pending = set(by_id)
        got_frames: list[DataFrame] = []
        status = "ok"
        for attempt in range(1, max_retries + 1):
            request = DataFrame([by_id[i] for i in sorted(pending)])
            df, status = _run_with_timeout(lambda: fetch(request, start_date, end_date), timeout_seconds)
            if df is not None and not df.empty:
                got_frames.append(df)
                pending -= set(int(x) for x in df["security_id"].dropna())
            if not pending:
                break
            if attempt < max_retries:
                log.warning(
                    "batch %d/%d attempt %d: %d/%d returned [%s] -- retrying %d missing in %.0fs",
                    b,
                    len(batches),
                    attempt,
                    len(by_id) - len(pending),
                    len(by_id),
                    status,
                    len(pending),
                    retry_backoff_seconds,
                )
                time.sleep(retry_backoff_seconds)
        if got_frames:
            batch_df = pd.concat(got_frames, ignore_index=True).drop_duplicates(subset=["security_id", "as_of_date"], keep="last")
            frames.append(batch_df)
            if on_batch is not None:
                on_batch(batch_df)
        missing.extend(sorted(pending))
        log.info("batch %d/%d done: %d/%d securities covered", b, len(batches), len(by_id) - len(pending), len(by_id))

    all_rows = pd.concat(frames, ignore_index=True) if frames else DataFrame()
    return all_rows, missing


# ---------------------------------------------------------------------------
# Database-backed pipeline steps
# ---------------------------------------------------------------------------


def _vendor_fetch(vendor: str) -> FetchFn:
    from data_engineering.eod_data import get_vendor

    v = get_vendor(vendor)

    def _fetch(symbol_df: DataFrame, start: str, end: str) -> DataFrame:
        data, _missing = v.fetch(symbol_df, start, end)
        return data

    return _fetch


def _stamp_vendor(df: DataFrame, vendor: str, currency: str) -> DataFrame:
    out = df.copy()
    out["source_vendor"] = vendor.capitalize() if vendor.islower() else vendor
    if "price_currency" not in out.columns:
        out["price_currency"] = currency
    return out


def load_eod_prices(
    universe: DataFrame,
    session: Any,
    start_date: str,
    end_date: str,
    *,
    vendor: str = "refinitiv",
    fetch: Optional[FetchFn] = None,
    batch_size: int = 20,
    max_retries: int = 5,
    retry_backoff_seconds: float = 20.0,
    timeout_seconds: Optional[float] = 900,
    implausible_move_threshold: Optional[float] = 0.25,
    price_currency: str = "USD",
) -> EodLoadSummary:
    """Pull EOD prices for ``universe`` from ``vendor`` and upsert them batch by batch.

    Each batch is de-duplicated on ``(security_id, as_of_date)``, screened with
    :func:`drop_implausible_moves` and written immediately, so a killed run keeps
    everything loaded so far. Returns an :class:`EodLoadSummary`.
    """
    fetch_fn = fetch or _vendor_fetch(vendor)
    summary = EodLoadSummary(securities_requested=int(universe["security_id"].nunique()))
    loaded: set[int] = set()

    def _write(batch_df: DataFrame) -> None:
        if implausible_move_threshold is not None:
            batch_df, dropped = drop_implausible_moves(batch_df, implausible_move_threshold)
            summary.dropped_implausible_rows += dropped
        if batch_df.empty:
            return
        db.write_market_data(_stamp_vendor(batch_df, vendor, price_currency), session)
        summary.rows_written += len(batch_df)
        loaded.update(int(x) for x in batch_df["security_id"].unique())

    _, missing = fetch_in_batches(
        universe,
        fetch_fn,
        start_date,
        end_date,
        batch_size=batch_size,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        timeout_seconds=timeout_seconds,
        on_batch=_write,
    )
    summary.securities_loaded = len(loaded)
    summary.missing_security_ids = sorted(set(missing) - loaded)
    log.info(
        "EOD pull complete: %d rows, %d/%d securities, %d implausible rows dropped, %d missing",
        summary.rows_written,
        summary.securities_loaded,
        summary.securities_requested,
        summary.dropped_implausible_rows,
        len(summary.missing_security_ids),
    )
    return summary


def find_broken_securities(engine: Any, security_ids: Iterable[int], start_date: str, end_date: str, min_days: int = 50) -> list[int]:
    """Securities with fewer than ``min_days`` priced days in the window (or none at all).

    A healthy constituent has hundreds of trading days; fewer than ~50 means the
    original pull failed for it. Names with valid-but-short history because they
    joined the index late are left alone by the threshold.
    """
    import sqlalchemy as sa

    window = pd.read_sql_query(
        sa.text("SELECT security_id, as_of_date FROM dbo.market_data WHERE as_of_date BETWEEN :s AND :e"),
        engine,
        params={"s": start_date, "e": end_date},
    )
    days = window.assign(security_id=window["security_id"].astype(int)).groupby("security_id")["as_of_date"].nunique()
    return sorted(int(s) for s in security_ids if int(days.get(int(s), 0)) < min_days)


def mop_up_missing_prices(
    universe: DataFrame,
    session: Any,
    engine: Any,
    start_date: str,
    end_date: str,
    *,
    vendor: str = "refinitiv",
    fetch: Optional[FetchFn] = None,
    min_days: int = 50,
    batch_size: int = 20,
    **load_kwargs: Any,
) -> EodLoadSummary:
    """Re-fetch only the securities that still have no usable prices after the main load."""
    broken = find_broken_securities(engine, universe["security_id"].tolist(), start_date, end_date, min_days=min_days)
    log.info("mop-up: %d/%d securities broken/empty in %s..%s", len(broken), len(universe), start_date, end_date)
    if not broken:
        return EodLoadSummary(securities_requested=0)
    subset = universe[universe["security_id"].isin(broken)].reset_index(drop=True)
    summary = load_eod_prices(subset, session, start_date, end_date, vendor=vendor, fetch=fetch, batch_size=batch_size, **load_kwargs)
    if summary.missing_security_ids:
        names = subset.set_index("security_id")["symbol"].to_dict()
        log.warning(
            "mop-up: still missing (typically delisted/renamed): %s", [(s, names.get(s)) for s in summary.missing_security_ids][:40]
        )
    return summary


def coverage_report(engine: Any, portfolio_short_name: str) -> DataFrame:
    """Per-date share of a portfolio's holdings that have a market_data row."""
    import sqlalchemy as sa

    q = sa.text(
        "SELECT ph.as_of_date, COUNT(DISTINCT ph.security_id) AS holdings, "
        "COUNT(DISTINCT CASE WHEN md.security_id IS NOT NULL THEN ph.security_id END) AS priced "
        "FROM dbo.portfolio_holdings ph "
        "JOIN dbo.portfolio p ON p.port_id = ph.port_id AND p.portfolio_short_name = :name "
        "LEFT JOIN dbo.market_data md ON md.security_id = ph.security_id AND md.as_of_date = ph.as_of_date "
        "GROUP BY ph.as_of_date ORDER BY ph.as_of_date"
    )
    cov = pd.read_sql_query(q, engine, params={"name": portfolio_short_name})
    cov["coverage_pct"] = (cov["priced"] / cov["holdings"] * 100).round(1)
    return cov


def prune_low_coverage_holdings(
    engine: Any, portfolio_short_names: list[str], start_date: str, end_date: str, min_coverage: float = 0.5
) -> list[str]:
    """Delete holdings on dates where fewer than ``min_coverage`` of names have a price.

    Holdings are generated for every weekday, but exchanges are closed on
    holidays; a date with 0-3 prices would otherwise renormalise the whole
    index onto those few names. Returns the pruned dates.
    """
    import sqlalchemy as sa

    names = tuple(portfolio_short_names)
    q = sa.text(
        "SELECT ph.as_of_date FROM dbo.portfolio_holdings ph "
        "JOIN dbo.portfolio p ON p.port_id = ph.port_id AND p.portfolio_short_name IN :names "
        "LEFT JOIN dbo.market_data md ON md.security_id = ph.security_id AND md.as_of_date = ph.as_of_date "
        "WHERE ph.as_of_date >= :s AND ph.as_of_date <= :e GROUP BY ph.as_of_date "
        "HAVING COUNT(DISTINCT CASE WHEN md.security_id IS NOT NULL THEN ph.security_id END) "
        "* 1.0 / COUNT(DISTINCT ph.security_id) < :cov"
    ).bindparams(sa.bindparam("names", expanding=True))
    dates = pd.read_sql_query(q, engine, params={"names": list(names), "s": start_date, "e": end_date, "cov": min_coverage})[
        "as_of_date"
    ].tolist()
    if dates:
        delete = sa.text(
            "DELETE FROM dbo.portfolio_holdings WHERE as_of_date = :d "
            "AND port_id IN (SELECT port_id FROM dbo.portfolio WHERE portfolio_short_name IN :names)"
        ).bindparams(sa.bindparam("names", expanding=True))
        with engine.begin() as conn:
            for d in dates:
                conn.execute(delete, {"d": d, "names": list(names)})
    log.info("pruned holdings on %d low-coverage dates", len(dates))
    return [str(pd.to_datetime(d).date()) for d in dates]
