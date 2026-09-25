"""Point-in-time fundamentals from the unified ``dbo.security_fundamentals`` table."""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
from pandas import DataFrame

from .base import FundamentalsProvider
from .ratios import RATIO_COLUMNS, empty_panel

log = logging.getLogger(__name__)


class StaticFundamentalsProvider(FundamentalsProvider):
    """Read ratios that a vendor loader already persisted to the database.

    This is the recommended scoring path: collection (slow, vendor-specific)
    is decoupled from scoring (fast, vendor-agnostic). Whichever vendor wrote
    the rows, they all sit in one table keyed by ``metric_type``.

    Point-in-time guarantee
    -----------------------
    ``get_panel(symbols, as_of_date)`` only uses rows whose availability date is
    on or before ``as_of_date``. There is deliberately *no* fallback to a
    globally-latest value, because that would leak future fundamentals into an
    earlier rebalance.

    ``effective_date`` in the table is the fiscal **period end**. Statements are
    published weeks after that, so scoring on ``effective_date`` alone is mildly
    forward-looking. ``availability_lag_days`` shifts every row's availability
    forward by that many calendar days (45 is a common choice for quarterly US
    filings; 0 keeps the historical behaviour).

    The long table is read once and cached; call :meth:`refresh` after writing
    new rows.
    """

    name = "static"

    def __init__(
        self,
        orm_session: Any,
        orm_engine: Any,
        source_vendor: Optional[str] = None,
        availability_lag_days: int = 0,
    ):
        """Bind to a database session.

        Args:
            orm_session: active SQLAlchemy ORM session.
            orm_engine: SQLAlchemy engine.
            source_vendor: optional ``source_vendor`` filter (e.g. ``"refinitiv"``).
            availability_lag_days: publication lag added to ``effective_date``.
        """
        self._session = orm_session
        self._engine = orm_engine
        self._source_vendor = source_vendor
        self._lag = pd.Timedelta(days=int(availability_lag_days))
        self._long: Optional[DataFrame] = None

    # -- data access ---------------------------------------------------------

    def refresh(self) -> None:
        """Drop the cached table so the next call re-reads the database."""
        self._long = None

    def _load(self) -> DataFrame:
        if self._long is None:
            from data_engineering.database import database as db

            long_df = db.read_security_fundamentals(self._session, self._engine, metric_type=None)
            if long_df.empty:
                self._long = long_df
                return long_df
            long_df = long_df.copy()
            # Plain str metric names: a pandas "string" dtype would fail the
            # reindex against the plain-str RATIO_COLUMNS and drop every column.
            long_df["metric_type"] = long_df["metric_type"].astype(str)
            long_df = long_df[long_df["metric_type"].isin(RATIO_COLUMNS)]
            if self._source_vendor:
                long_df = long_df[long_df["source_vendor"] == self._source_vendor]
            long_df["available_date"] = pd.to_datetime(long_df["effective_date"]) + self._lag
            long_df["security_id"] = long_df["security_id"].astype(int)
            self._long = long_df.sort_values("available_date").reset_index(drop=True)
        return self._long

    # -- provider API --------------------------------------------------------

    def get_panel(self, symbols: DataFrame, as_of_date: str) -> DataFrame:
        """Latest ratio per (security, metric) available on/before ``as_of_date``."""
        sec_ids = [int(s) for s in symbols["security_id"].tolist()]
        long_df = self._load()
        if long_df.empty:
            return empty_panel(sec_ids)

        cutoff = pd.Timestamp(as_of_date)
        visible = long_df[(long_df["available_date"] <= cutoff) & (long_df["security_id"].isin(sec_ids))]
        if visible.empty:
            return empty_panel(sec_ids)

        latest = visible.groupby(["security_id", "metric_type"], as_index=False).tail(1)
        wide = latest.pivot_table(index="security_id", columns="metric_type", values="metric_value")
        wide = wide.reindex(index=pd.Index(sec_ids, name="security_id"), columns=RATIO_COLUMNS)
        return wide[RATIO_COLUMNS].astype(float)

    def get_panel_history(
        self,
        symbols: DataFrame,
        as_of_date: str,
        frequency: str = "FY",
        start_date: Optional[str] = None,
    ) -> tuple[DataFrame, Optional[DataFrame]]:
        """Return ``(latest_panel, history)`` where history is the stored long frame.

        The history frame carries every stored fiscal period for the requested
        securities (``effective_date`` = period end), which is what a
        point-in-time training table needs.
        """
        sec_ids = [int(s) for s in symbols["security_id"].tolist()]
        long_df = self._load()
        panel = self.get_panel(symbols, as_of_date)
        if long_df.empty:
            return panel, None
        sub = long_df[long_df["security_id"].isin(sec_ids)]
        if start_date:
            sub = sub[pd.to_datetime(sub["effective_date"]) >= pd.Timestamp(start_date)]
        sub = sub[pd.to_datetime(sub["effective_date"]) <= pd.Timestamp(as_of_date)]
        if sub.empty:
            return panel, None
        history = sub.pivot_table(index=["security_id", "effective_date"], columns="metric_type", values="metric_value")
        history = history.reindex(columns=RATIO_COLUMNS).reset_index()
        return panel, history


#: Clearer alias for new code; ``StaticFundamentalsProvider`` is kept for callers.
DatabaseFundamentalsProvider = StaticFundamentalsProvider
