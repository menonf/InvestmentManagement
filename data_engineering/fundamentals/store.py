"""Collect fundamentals from a provider and persist them to the database."""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
from pandas import DataFrame

from .ratios import RATIO_COLUMNS

log = logging.getLogger(__name__)

FUNDAMENTALS_COLUMNS = ["security_id", "metric_type", "metric_value", "source_vendor", "effective_date", "end_date"]


def panel_to_long(panel: DataFrame, effective_date: Any, source_vendor: str) -> DataFrame:
    """Melt a wide ratio panel (index=security_id) into ``security_fundamentals`` rows."""
    long_df = panel[RATIO_COLUMNS].stack(future_stack=True).dropna().rename("metric_value").reset_index()
    long_df.columns = ["security_id", "metric_type", "metric_value"]
    long_df["security_id"] = long_df["security_id"].astype(int)
    long_df["metric_value"] = long_df["metric_value"].astype(float)
    long_df["source_vendor"] = source_vendor
    long_df["effective_date"] = pd.to_datetime(effective_date).date()
    long_df["end_date"] = None
    return long_df[FUNDAMENTALS_COLUMNS]


def history_to_long(history: DataFrame, source_vendor: str) -> DataFrame:
    """Melt a long per-period history ``[security_id, effective_date, *ratios]`` into DB rows."""
    melted = history.melt(
        id_vars=["security_id", "effective_date"], value_vars=RATIO_COLUMNS, var_name="metric_type", value_name="metric_value"
    )
    melted = melted.dropna(subset=["metric_value"])
    melted["security_id"] = melted["security_id"].astype(int)
    melted["metric_value"] = melted["metric_value"].astype(float)
    melted["effective_date"] = pd.to_datetime(melted["effective_date"]).dt.date
    melted["source_vendor"] = source_vendor
    melted["end_date"] = None
    return melted[FUNDAMENTALS_COLUMNS]


def collect_and_store_fundamentals(
    provider_name: str,
    symbols: DataFrame,
    as_of_date: str,
    orm_session: Any,
    orm_engine: Any,
    source_vendor: Optional[str] = None,
    provider_kwargs: Optional[dict[str, Any]] = None,
) -> int:
    """Collect ratios via a named provider and upsert them into ``dbo.security_fundamentals``.

    With ``provider_kwargs={"frequency": "Q", "start_date": ...}`` the provider's
    quarterly history is stored with one row per (security, metric, period end);
    otherwise a single snapshot dated ``as_of_date`` is stored.

    Returns:
        Number of (security, metric, date) rows written.
    """
    from data_engineering.database import database as db

    from . import get_fundamentals_provider

    kwargs = dict(provider_kwargs or {})
    frequency = kwargs.get("frequency", "FY")
    start_date = kwargs.get("start_date")
    provider = get_fundamentals_provider(provider_name, orm_session=orm_session, orm_engine=orm_engine, **kwargs)
    snapshot, history = provider.get_panel_history(symbols, as_of_date, frequency=frequency, start_date=start_date)
    vendor = source_vendor or provider.name

    if history is not None and not history.empty:
        rows = history_to_long(history, vendor)
    elif snapshot is not None and not snapshot.empty:
        rows = panel_to_long(snapshot, as_of_date, vendor)
    else:
        log.warning("provider '%s' returned no data", provider_name)
        return 0

    if rows.empty:
        log.warning("no non-null ratios to store")
        return 0
    n = db.write_security_fundamentals(rows, orm_session)
    log.info("wrote %s fundamentals rows from '%s'", n, vendor)
    return int(n)
