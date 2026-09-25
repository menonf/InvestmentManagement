"""LSEG / Refinitiv fundamentals provider.

Pulls raw accounting items per RIC (annual snapshot via ``get_data`` or
quarterly history via ``get_history``) and derives the 18 ratios with
:func:`~data_engineering.fundamentals.ratios.compute_ratios`. All RIC and
session plumbing comes from :mod:`data_engineering.refinitiv`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
from pandas import DataFrame

from data_engineering.refinitiv import ensure_session, get_data_chunked, get_history_chunked, qualify_tickers

from .base import FundamentalsProvider
from .ratios import RATIO_COLUMNS, compute_ratios, empty_panel

log = logging.getLogger(__name__)

#: ``TR.*`` codes requested from LSEG (the request must use code form).
REFINITIV_REQUEST_FIELDS = [
    "TR.PriceClose",
    "TR.MarketCapitalization",
    "TR.EBIT",
    "TR.TotalRevenue",
    "TR.NetIncome",
    "TR.TotalDebt",
    "TR.TotalAssets",
    "TR.TotalCurrentAssets",
    "TR.CurrentLiabilities",
    "TR.TotalLiabilities",
    "TR.GrossProfit",
    "TR.RetainedEarnings",
    "TR.InterestExpense",
    "TR.OperatingIncome",
    "TR.SharesOutstanding",
]

#: Display names LSEG returns -> internal raw item names (see ``ratios.RAW_ITEMS``).
REFINITIV_RAW_FIELDS = {
    "Price Close": "price",
    "Market Capitalization": "mkt_cap",
    "EBIT": "ebit",
    "Total Revenue": "revenue",
    "Net Income Incl Extra Before Distributions": "net_income",
    "Total Debt": "total_debt",
    "Total Assets": "total_assets",
    "Total Current Assets": "current_assets",
    "Current Liabilities": "current_liab",
    "Total Liabilities": "total_liab",
    "Gross Profit": "gross_profit",
    "Retained Earnings (Accumulated Deficit)": "retained_earnings",
    "Interest Expense": "interest_exp",
    "Operating Income": "operating_income",
    "Outstanding Shares": "shares",
}

#: Items this LSEG entitlement does not return. ``compute_ratios`` derives
#: book equity (TA - TL), market cap (price x shares) and EV (mkt cap + debt);
#: Cash Ratio and Op.In./(NWC+FA) stay NaN.
REFINITIV_UNAVAILABLE = {
    "Enterprise Value": "ent_val",
    "Total Stockholders Equity": "book_equity",
    "Cash and Short Term Investments": "cash",
    "Net Property Plant and Equipment": "fixed_assets",
}

ANNUAL_CHUNK = 100
QUARTERLY_CHUNK = 15


def compute_refinitiv_ratios(raw: DataFrame) -> DataFrame:
    """Rename LSEG display columns to raw items and compute the 18 ratios."""
    return compute_ratios(raw.rename(columns=REFINITIV_RAW_FIELDS))


class RefinitivFundamentalsProvider(FundamentalsProvider):
    """Collect the 18 ratios from LSEG via ``lseg.data``.

    Requires LSEG Workspace running locally (the session is opened lazily).

    Args:
        orm_session / orm_engine: optional DB handles used to resolve RICs from
            ``security_vendor_xref`` (preferred over the ``symbol`` column).
        frequency: ``"FY"`` = latest fiscal-year snapshot; ``"Q"`` = quarterly
            history (used by :meth:`get_panel_history`).
        start_date: history start for the quarterly mode.
    """

    name = "refinitiv"

    def __init__(
        self,
        session: Optional[Any] = None,
        orm_session: Optional[Any] = None,
        orm_engine: Optional[Any] = None,
        frequency: str = "FY",
        start_date: Optional[str] = None,
    ):
        """Store DB handles and pull-mode defaults; the LSEG session opens lazily."""
        self._session = session
        self._orm_session = orm_session
        self._orm_engine = orm_engine
        self._frequency = frequency
        self._start_date = start_date

    # -- RIC resolution ------------------------------------------------------

    def _resolve_rics(self, symbols: DataFrame) -> DataFrame:
        """Add an exchange-qualified ``ric`` column to ``symbols``.

        The fundamentals endpoint only returns data for qualified RICs
        (``AAPL.O``), so every base identifier - from the vendor xref when a DB
        session is available, else the ``symbol`` column - is qualified via
        LSEG ``symbol_conversion``.
        """
        out = symbols.copy()
        out["ric"] = pd.NA

        if self._orm_session is not None and self._orm_engine is not None:
            from data_engineering.database import database as db

            xref = db.read_security_vendor_xref(self._orm_session, self._orm_engine, vendor="Refinitiv")
            if not xref.empty:
                xref = xref[xref["security_id"].isin(symbols["security_id"].tolist())]
                xref = xref.sort_values("is_primary", ascending=False).drop_duplicates("security_id")
                out["ric"] = out["security_id"].map(xref.set_index("security_id")["vendor_ticker"])

        missing = out["ric"].isna()
        if missing.any():
            out.loc[missing, "ric"] = out.loc[missing, "symbol"]

        qmap = qualify_tickers(out["ric"].dropna().unique().tolist())
        out["ric"] = out["ric"].map(lambda r: qmap.get(r, r))
        return out

    # -- provider API --------------------------------------------------------

    def get_panel(self, symbols: DataFrame, as_of_date: str) -> DataFrame:
        """Latest fiscal-year ratio panel for ``symbols``."""
        return self.get_panel_history(symbols, as_of_date, frequency="FY")[0]

    def get_panel_history(
        self,
        symbols: DataFrame,
        as_of_date: str,
        frequency: str = "FY",
        start_date: Optional[str] = None,
    ) -> tuple[DataFrame, Optional[DataFrame]]:
        """Return ``(snapshot_panel, history)``.

        * ``frequency="FY"``: latest fiscal-year snapshot; history is ``None``.
        * ``frequency="Q"``: quarterly history from ``start_date`` (or
          ``as_of_date``) to today. The snapshot is the most recent quarter per
          security; history is a long frame ``[security_id, effective_date,
          *RATIO_COLUMNS]`` with one row per fiscal quarter. Each RIC reports
          on its *own* fiscal calendar, so period ends are not aligned across
          names and are never snapped to a common grid.
        """
        sec_ids = symbols["security_id"].tolist()
        try:
            ensure_session()
        except Exception as exc:  # noqa: BLE001 - Workspace must be running locally
            log.error("could not open LSEG session: %s", exc)
            return empty_panel(sec_ids), None

        resolved = self._resolve_rics(symbols)
        rics = resolved["ric"].dropna().unique().tolist()
        if not rics:
            return empty_panel(sec_ids), None

        if frequency == "Q":
            return self._quarterly_history(resolved, rics, start_date or as_of_date)

        raw = get_data_chunked(
            rics,
            REFINITIV_REQUEST_FIELDS,
            {"Frq": "FY"},
            chunk_size=ANNUAL_CHUNK,
            max_retries=3,
            inter_chunk_sleep=1.0,
        )
        if raw.empty:
            return empty_panel(sec_ids), None
        raw = raw.rename(columns={"Instrument": "ric"}).set_index("ric")
        return self._to_security_panel(compute_refinitiv_ratios(raw), resolved, sec_ids), None

    # -- internals -----------------------------------------------------------

    def _quarterly_history(self, resolved: DataFrame, rics: list[str], start_date: str) -> tuple[DataFrame, Optional[DataFrame]]:
        sec_map = resolved.dropna(subset=["ric"]).set_index("ric")["security_id"].astype(int).to_dict()
        params = {
            "Frq": "Q",
            "SDate": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
            "EDate": pd.Timestamp.today().strftime("%Y-%m-%d"),
        }
        records: list[DataFrame] = []
        for chunk, hist in get_history_chunked(rics, REFINITIV_REQUEST_FIELDS, params, chunk_size=QUARTERLY_CHUNK):
            hist = hist.copy()
            hist.index = pd.to_datetime(hist.index)
            if not isinstance(hist.columns, pd.MultiIndex):  # single RIC -> flat columns
                hist.columns = pd.MultiIndex.from_product([chunk[:1], hist.columns])
            for ric in hist.columns.get_level_values(0).unique():
                sec_id = sec_map.get(ric)
                if sec_id is None:
                    continue
                sub = hist[ric].dropna(how="all")  # only this RIC's real fiscal dates
                if sub.empty:
                    continue
                ratios = compute_refinitiv_ratios(sub)
                ratios.index.name = "effective_date"
                ratios = ratios.reset_index()
                ratios.insert(0, "security_id", int(sec_id))
                records.append(ratios[["security_id", "effective_date"] + RATIO_COLUMNS])

        sec_ids = resolved["security_id"].tolist()
        if not records:
            return empty_panel(sec_ids), None
        history = pd.concat(records, ignore_index=True)
        snapshot = history.sort_values("effective_date").groupby("security_id").tail(1).set_index("security_id")[RATIO_COLUMNS]
        panel = empty_panel(sec_ids)
        panel.loc[snapshot.index.intersection(panel.index)] = snapshot
        return panel, history

    @staticmethod
    def _to_security_panel(ratios_by_ric: DataFrame, resolved: DataFrame, sec_ids: list[int]) -> DataFrame:
        sec_map = resolved.dropna(subset=["ric"]).set_index("ric")["security_id"].astype(int)
        joined = ratios_by_ric.join(sec_map.rename("security_id"), how="left").dropna(subset=["security_id"])
        joined["security_id"] = joined["security_id"].astype(int)
        joined = joined.set_index("security_id")[RATIO_COLUMNS]
        panel = empty_panel(sec_ids)
        panel.loc[joined.index.intersection(panel.index)] = joined
        return panel[RATIO_COLUMNS]
