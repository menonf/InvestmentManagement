"""In-memory point-in-time fundamentals provider (tests, demos, CSV-backed research)."""

from __future__ import annotations

from typing import Optional

import pandas as pd
from pandas import DataFrame

from .base import FundamentalsProvider
from .ratios import RATIO_COLUMNS, empty_panel


class InMemoryFundamentalsProvider(FundamentalsProvider):
    """Serve point-in-time panels from a long history frame already in memory.

    Args:
        history: ``[security_id, effective_date, *RATIO_COLUMNS]`` with one row
            per security per fiscal period (the same shape the Refinitiv
            provider's quarterly history and the database provider's
            ``get_panel_history`` return).
        availability_lag_days: publication lag added to ``effective_date``
            before a row becomes visible (see ``StaticFundamentalsProvider``).
    """

    name = "memory"

    def __init__(self, history: DataFrame, availability_lag_days: int = 0):
        """Index the history by availability date."""
        hist = history.copy()
        hist["security_id"] = hist["security_id"].astype(int)
        hist["effective_date"] = pd.to_datetime(hist["effective_date"])
        hist["available_date"] = hist["effective_date"] + pd.Timedelta(days=int(availability_lag_days))
        self._history = hist.sort_values(["available_date", "security_id"]).reset_index(drop=True)

    def get_panel(self, symbols: DataFrame, as_of_date: str) -> DataFrame:
        """Latest row per security available on/before ``as_of_date``."""
        sec_ids = [int(s) for s in symbols["security_id"]]
        visible = self._history[
            (self._history["available_date"] <= pd.Timestamp(as_of_date)) & self._history["security_id"].isin(sec_ids)
        ]
        if visible.empty:
            return empty_panel(sec_ids)
        latest = visible.groupby("security_id").tail(1).set_index("security_id")
        return latest.reindex(index=pd.Index(sec_ids, name="security_id"), columns=RATIO_COLUMNS).astype(float)

    def get_panel_history(
        self, symbols: DataFrame, as_of_date: str, frequency: str = "FY", start_date: Optional[str] = None
    ) -> tuple[DataFrame, Optional[DataFrame]]:
        """``(latest_panel, history rows with effective_date in the window)``."""
        sec_ids = [int(s) for s in symbols["security_id"]]
        hist = self._history[
            self._history["security_id"].isin(sec_ids) & (self._history["effective_date"] <= pd.Timestamp(as_of_date))
        ]
        if start_date:
            hist = hist[hist["effective_date"] >= pd.Timestamp(start_date)]
        return self.get_panel(symbols, as_of_date), (
            hist[["security_id", "effective_date"] + RATIO_COLUMNS].reset_index(drop=True) if not hist.empty else None
        )
