"""Fundamentals loading with per-security retry and a point-in-time audit."""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
from pandas import DataFrame

from data_engineering.database import database as db
from data_engineering.fundamentals import RATIO_COLUMNS, FundamentalsProvider, collect_and_store_fundamentals

log = logging.getLogger(__name__)


def fundamentals_coverage(
    session: Any, engine: Any, security_ids: list[int], from_date: str, to_date: str, source_vendor: str = "refinitiv"
) -> DataFrame:
    """Stored ratio rows for ``security_ids`` with ``effective_date`` inside the window."""
    rows = db.read_security_fundamentals(session, engine, metric_type=None)
    rows = rows[
        (rows["source_vendor"] == source_vendor) & rows["security_id"].isin(security_ids) & rows["metric_type"].isin(RATIO_COLUMNS)
    ].copy()
    rows["effective_date"] = pd.to_datetime(rows["effective_date"])
    return rows[(rows["effective_date"] >= pd.Timestamp(from_date)) & (rows["effective_date"] <= pd.Timestamp(to_date))]


def load_fundamentals_with_retry(
    universe: DataFrame,
    session: Any,
    engine: Any,
    from_date: str,
    to_date: str,
    *,
    provider_name: str = "refinitiv",
    frequency: str = "Q",
    source_vendor: str = "refinitiv",
    retry_individually: bool = True,
) -> dict[str, Any]:
    """Load ratios for ``universe`` then retry, one security at a time, any name left without rows.

    Smaller requests are far less likely to hit the gateway timeout that makes
    a whole chunk disappear, which is why the retry is per security.

    Returns a summary dict: rows written, covered / missing security ids.
    """
    kwargs: dict[str, Any] = {"frequency": frequency}
    if frequency == "Q":
        kwargs["start_date"] = from_date
    symbols = universe[["security_id", "symbol"]]
    written = collect_and_store_fundamentals(
        provider_name, symbols, to_date, session, engine, source_vendor=source_vendor, provider_kwargs=kwargs
    )

    all_ids = set(int(s) for s in symbols["security_id"])
    covered = set(fundamentals_coverage(session, engine, list(all_ids), from_date, to_date, source_vendor)["security_id"].astype(int))
    missed = sorted(all_ids - covered)
    log.info("coverage after initial pull: %d/%d securities", len(covered), len(all_ids))

    recovered = 0
    if retry_individually and missed:
        for i, sec_id in enumerate(missed, start=1):
            one = symbols[symbols["security_id"] == sec_id]
            try:
                n = collect_and_store_fundamentals(
                    provider_name, one, to_date, session, engine, source_vendor=source_vendor, provider_kwargs=kwargs
                )
            except Exception as exc:  # noqa: BLE001 - keep going through the list
                log.warning("[%d/%d] security %s failed: %s", i, len(missed), sec_id, exc)
                continue
            written += n
            if n:
                recovered += 1
        covered = set(
            fundamentals_coverage(session, engine, list(all_ids), from_date, to_date, source_vendor)["security_id"].astype(int)
        )
        log.info("coverage after retry: %d/%d (%d recovered of %d)", len(covered), len(all_ids), recovered, len(missed))
    return {"rows_written": int(written), "covered": sorted(covered), "missing": sorted(all_ids - covered)}


def check_point_in_time(
    provider: FundamentalsProvider,
    universe: DataFrame,
    from_date: str,
    to_date: str,
    stored_rows: DataFrame,
    freq: str = "QE",
) -> int:
    """Assert the provider never exposes a ratio dated after the as-of date it was asked for.

    For every period end in ``[from_date, to_date]`` the panel returned by
    ``provider.get_panel`` is compared with the latest ``effective_date`` per
    (security, metric) in ``stored_rows`` at or before that date. Returns the
    worst-case number of leaked values (must be 0) and raises ``AssertionError``
    on the first violation.
    """
    rows = stored_rows[stored_rows["metric_type"].isin(RATIO_COLUMNS)].copy()
    rows["effective_date"] = pd.to_datetime(rows["effective_date"])
    worst = 0
    for period_end in pd.date_range(from_date, to_date, freq=freq):
        panel = provider.get_panel(universe, period_end.strftime("%Y-%m-%d"))
        visible = panel.stack(future_stack=True).dropna()
        if visible.empty:
            continue
        latest = rows[rows["effective_date"] <= period_end].groupby(["security_id", "metric_type"])["effective_date"].max()
        leaked = [(s, m) for (s, m) in visible.index if (s, m) not in latest.index or latest.loc[(s, m)] > period_end]
        worst = max(worst, len(leaked))
        assert not leaked, f"forward-looking fundamentals at {period_end.date()}: {leaked[:3]}"
    return worst
