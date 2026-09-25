"""Provider contract shared by every fundamentals source."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from pandas import DataFrame, Series


class FundamentalsProvider(ABC):
    """Abstract source of a cross-sectional fundamental-ratio panel.

    A provider turns a security universe + as-of date into a wide ratio panel
    (index = ``security_id``, columns = ``RATIO_COLUMNS``). Factors only ever
    see that panel, so swapping vendors never touches the analytics layer.
    """

    #: Unique, lowercase provider key (``"refinitiv"``, ``"yahoo"``, ...).
    name: str = "base"

    @abstractmethod
    def get_panel(self, symbols: DataFrame, as_of_date: str) -> DataFrame:
        """Return the ratio panel for ``symbols`` as known on ``as_of_date``.

        Args:
            symbols: DataFrame with at least ``security_id`` and ``symbol``
                (vendor-native ticker) columns.
            as_of_date: inclusive snapshot date ``YYYY-MM-DD``. Providers return
                the most recent fundamentals on or before this date.

        Returns:
            DataFrame indexed by ``security_id`` with columns = ``RATIO_COLUMNS``.
            Missing ratios stay NaN - imputation is the factor's decision.
        """
        raise NotImplementedError

    def get_panel_history(
        self,
        symbols: DataFrame,
        as_of_date: str,
        frequency: str = "FY",
        start_date: Optional[str] = None,
    ) -> tuple[DataFrame, Optional[DataFrame]]:
        """Return ``(latest_panel, history)``.

        ``history`` is a long frame ``[security_id, effective_date, *RATIO_COLUMNS]``
        with one row per fiscal period, or ``None`` for providers that only know
        a single snapshot. The default wraps :meth:`get_panel`.
        """
        return self.get_panel(symbols, as_of_date), None

    def latest_effective_dates(self, symbols: DataFrame, as_of_date: str) -> Series:
        """``security_id -> latest fiscal period end`` visible on/before ``as_of_date``.

        Derived from :meth:`get_panel_history` where the provider exposes period
        history; providers with direct access should override for efficiency.
        Names with no visible fundamentals come back as ``NaT``. Returns a
        ``Series`` indexed by ``security_id``.
        """
        import pandas as pd
        from pandas import Series

        sec_ids = [int(s) for s in symbols["security_id"]]
        idx = pd.Index(sec_ids, name="security_id")
        _panel, history = self.get_panel_history(symbols, as_of_date)
        if history is None or history.empty:
            return Series(pd.NaT, index=idx)
        hist = history.copy()
        hist["effective_date"] = pd.to_datetime(hist["effective_date"])
        latest = hist.groupby("security_id")["effective_date"].max()
        return latest.reindex(idx)
