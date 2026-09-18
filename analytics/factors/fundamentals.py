"""Vendor-agnostic fundamentals (ratio) interface for ML-driven factors.

The ML factor in :mod:`analytics.factors.ml_factor` needs a *cross-sectional*
panel of fundamental ratios (one row per security) at a point in time. This
module defines:

    * ``RATIO_COLUMNS`` - the canonical 18 fundamental ratios the "Build Your
      Own AI Investor" models are trained on. Keeping these names stable is
      important: a pipeline fitted on ``RATIO_COLUMNS`` will break if the
      feature order/names change.
    * ``FundamentalsProvider`` - the abstract contract every vendor implements.
      A provider turns a security universe + as_of_date into a *wide* ratio
      panel (index = security_id, columns = ``RATIO_COLUMNS``). The ML factor
      never sees the vendor; it only sees this panel.
    * Concrete providers:
        - ``StaticFundamentalsProvider`` reads the unified
          ``dbo.security_fundamentals`` table (whatever vendor loaded it). This
          is the recommended production path: collection is decoupled from
          scoring.
        - ``YahooFundamentalsProvider`` collects live ratios via yfinance.
        - ``SimFinFundamentalsProvider`` collects via the SimFin API.
        - ``RefinitivFundamentalsProvider`` collects via LSEG/Refinitiv
          (lseg.data) - requires the local Workspace session.

Because every provider emits the *same* wide panel, the framework can swap
Yahoo / SimFin / Refinitiv without touching the factor or the models.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from sqlalchemy import Engine
from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Canonical ratio contract
# ---------------------------------------------------------------------------

#: The 18 fundamental ratios the ML models consume, in the exact order/naming
#: used when the pipelines were (re)trained. DO NOT reorder or rename without
#: retraining, or ``feature_names_in_`` mismatches will raise at predict time.
RATIO_COLUMNS = [
    "EV/EBIT",
    "Op. In./(NWC+FA)",
    "P/E",
    "P/B",
    "P/S",
    "Op. In./Interest Expense",
    "Working Capital Ratio",
    "RoE",
    "ROCE",
    "Debt/Equity",
    "Debt Ratio",
    "Cash Ratio",
    "Asset Turnover",
    "Gross Profit Margin",
    "(CA-CL)/TA",
    "RE/TA",
    "EBIT/TA",
    "Book Equity/TL",
]


def _empty_panel(security_ids: Iterable[int]) -> DataFrame:
    """Wide ratio panel (security_id index) filled with NaN."""
    idx = list(security_ids)
    return DataFrame(index=pd.Index(idx, name="security_id"), columns=RATIO_COLUMNS, dtype=float)


def _safe_ratio(*operands: Optional[float], numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Divide ``numerator`` by ``denominator`` only when all values are present.

    Returns ``None`` if any operand is ``None``/``NaN`` or the denominator is 0,
    so missing fundamentals degrade gracefully instead of raising.
    """
    if numerator is None or denominator in (None, 0):
        return None
    for op in operands:
        if op is None:
            return None
    return numerator / denominator


# ---------------------------------------------------------------------------
# Provider contract
# ---------------------------------------------------------------------------


class FundamentalsProvider(ABC):
    """Abstract source of a cross-sectional fundamental-ratio panel.

    Implementations turn a security universe into a *wide* ratio panel so the
    ML factor stays vendor-agnostic.
    """

    #: Unique, lowercase provider key (e.g. ``"yahoo"``).
    name: str = "base"

    @abstractmethod
    def get_panel(
        self,
        symbols: DataFrame,
        as_of_date: str,
    ) -> DataFrame:
        """Return a wide ratio panel for the given universe.

        Args:
            symbols: DataFrame with at least ``security_id`` and ``symbol``
                (vendor-native ticker) columns.
            as_of_date: Inclusive snapshot date ``YYYY-MM-DD``. Providers return
                the most recent fundamentals on or before this date.

        Returns:
            DataFrame indexed by ``security_id`` with columns = ``RATIO_COLUMNS``.
            Missing ratios are left as NaN (never silently zero-filled here; the
            factor decides how to impute).
        """
        raise NotImplementedError

    def get_panel_history(
        self,
        symbols: DataFrame,
        as_of_date: str,
        frequency: str = "FY",
        start_date: Optional[str] = None,
    ) -> tuple[DataFrame, Optional[DataFrame]]:
        """Return ``(latest_panel, None)`` by default.

        The default implementation wraps :meth:`get_panel` (single snapshot, no
        history frame). Providers that support per-period history override this.
        """
        return self.get_panel(symbols, as_of_date), None


# ---------------------------------------------------------------------------
# DB-backed (vendor-neutral) provider
# ---------------------------------------------------------------------------


class StaticFundamentalsProvider(FundamentalsProvider):
    """Read ratios from the unified ``dbo.security_fundamentals`` table.

    This is the recommended production provider: whichever vendor collected the
    ratios (Yahoo/SimFin/Refinitiv), they all land in the same table keyed by
    ``metric_type``, so scoring is 100% vendor-agnostic. The ML factor is
    constructed with one of these by default.
    """

    name = "static"

    def __init__(self, orm_session: Session, orm_engine: Engine, source_vendor: Optional[str] = None):
        """Initialize the static provider from the DB-backed fundamentals table.

        Args:
            orm_session: Active SQLAlchemy ORM session.
            orm_engine: SQLAlchemy engine.
            source_vendor: Optional vendor filter applied when reading fundamentals.
        """
        # Local import avoids a hard dependency on the DB layer for pure tests.
        from data_engineering.database import database as db

        self._db = db
        self._session = orm_session
        self._engine = orm_engine
        self._source_vendor = source_vendor

    def get_panel(self, symbols: DataFrame, as_of_date: str) -> DataFrame:
        """Return the fundamentals panel for the given symbols.

        Args:
            symbols: DataFrame with a ``security_id`` column.
            as_of_date: Snapshot date (kept for interface compatibility).

        Returns:
            Wide fundamentals panel indexed by ``security_id``.
        """
        sec_ids = symbols["security_id"].tolist()
        long_df = self._db.read_security_fundamentals(self._session, self._engine, metric_type=None)
        if long_df.empty:
            return _empty_panel(sec_ids)

        long_df = long_df.copy()
        long_df["as_of_date"] = pd.to_datetime(long_df["effective_date"])
        # Coerce metric_type to plain str so the pivoted column index matches
        # the plain-str RATIO_COLUMNS contract (pandas "string" dtype would
        # otherwise fail the reindex and drop every column -> all-NaN panel).
        long_df["metric_type"] = long_df["metric_type"].astype(str)
        cutoff = pd.Timestamp(as_of_date)
        if self._source_vendor:
            long_df = long_df[long_df["source_vendor"] == self._source_vendor]

        # CRITICAL (no forward-looking bias): only fundamentals observed ON or
        # BEFORE the snapshot may be used. We deliberately do NOT fall back to a
        # globally-latest value, because that would leak future fundamentals into
        # an earlier rebalance. Names/metrics with no observation at-or-before
        # the cutoff simply stay NaN and are imputed downstream.
        latest_at_or_before = (
            long_df[long_df["as_of_date"] <= cutoff]
            .sort_values("as_of_date")
            .groupby(["security_id", "metric_type"], as_index=False)
            .tail(1)
        )
        if latest_at_or_before.empty:
            return _empty_panel(sec_ids)

        wide = latest_at_or_before.pivot_table(index="security_id", columns="metric_type", values="metric_value")
        # Restrict/order to the canonical contract; missing -> NaN.
        wide = wide.reindex(columns=RATIO_COLUMNS)
        wide = wide.reindex(index=pd.Index(sec_ids, name="security_id"))
        return wide[RATIO_COLUMNS]

    def get_panel_history(
        self,
        symbols: DataFrame,
        as_of_date: str,
        frequency: str = "FY",
        start_date: Optional[str] = None,
    ) -> tuple[DataFrame, Optional[DataFrame]]:
        """Return ``(latest_panel, None)`` for the DB-backed static provider.

        The static provider reads a single point-in-time snapshot; there is no
        per-period history, so the history frame is always ``None``.
        """
        return self.get_panel(symbols, as_of_date), None


# ---------------------------------------------------------------------------
# Yahoo (live) provider
# ---------------------------------------------------------------------------


class YahooFundamentalsProvider(FundamentalsProvider):
    """Collect the 18 ratios live from Yahoo Finance via yfinance.

    Ratio definitions follow the "Build Your Own AI Investor" feature set as
    closely as Yahoo's reported fields allow. Where a precise field is
    unavailable, the ratio is left NaN rather than guessed. This provider is
    useful for bootstrapping the ``security_fundamentals`` table before
    switching scoring to :class:`StaticFundamentalsProvider`.
    """

    name = "yahoo"

    def get_panel(self, symbols: DataFrame, as_of_date: str) -> DataFrame:
        """Return the Yahoo-sourced fundamentals panel for the given symbols.

        Args:
            symbols: DataFrame with ``security_id`` and ``symbol`` columns.
            as_of_date: Snapshot date (kept for interface compatibility).

        Returns:
            Wide fundamentals panel indexed by ``security_id`` (NaN where missing).
        """
        panel = _empty_panel(symbols["security_id"].tolist())
        for _, row in symbols.iterrows():
            sec_id = row["security_id"]
            try:
                ratios = self._ratios_for_symbol(row["symbol"])
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not fail all
                print(f"[yahoo fundamentals] skipped {row['symbol']}: {exc}")
                continue
            for col, val in ratios.items():
                if col in panel.columns and val is not None and pd.notna(val):
                    panel.loc[sec_id, col] = float(val)
        return panel

    # -- ratio extraction --------------------------------------------------

    @staticmethod
    def _ratios_for_symbol(symbol: str) -> dict[str, Any]:
        import yfinance as yf

        tk = yf.Ticker(symbol)
        info = tk.info or {}

        price = info.get("currentPrice") or info.get("regularMarketPrice")
        ent_val = info.get("enterpriseValue")
        ebit = info.get("ebit")
        revenue = info.get("totalRevenue")
        net_income = info.get("netIncomeToCommon") or info.get("netIncome")
        book_value = info.get("bookValue")
        total_debt = info.get("totalDebt")
        total_equity = info.get("stockholdersEquity") or info.get("totalStockholderEquity")
        cash = info.get("totalCash")
        total_assets = info.get("totalAssets")
        current_assets = info.get("totalCurrentAssets")
        current_liab = info.get("totalCurrentLiabilities")
        working_capital = (current_assets or 0) - (current_liab or 0)
        fixed_assets = info.get("propertyPlantEquipment") or 0
        gross_profit = info.get("grossProfits")
        retained_earnings = info.get("retainedEarnings")
        interest_exp = info.get("interestExpense")
        operating_income = info.get("operatingIncome") or info.get("ebit")
        shares = info.get("sharesOutstanding") or 1

        ratios: dict[str, Any] = {}

        # Valuation multiples
        ratios["P/E"] = _safe_ratio(net_income, price, numerator=price, denominator=(net_income / shares)) if net_income else None
        ratios["P/B"] = _safe_ratio(book_value, price, numerator=price, denominator=book_value)
        ratios["P/S"] = _safe_ratio(revenue, price, numerator=price, denominator=(revenue / shares)) if revenue else None
        ratios["EV/EBIT"] = _safe_ratio(ebit, ent_val, numerator=ent_val, denominator=ebit)

        # Profitability / returns
        ratios["RoE"] = _safe_ratio(net_income, total_equity, numerator=net_income, denominator=total_equity)
        ratios["ROCE"] = (
            _safe_ratio(ebit, total_equity, total_debt, numerator=ebit, denominator=(total_equity + total_debt))
            if (total_equity is not None and total_debt is not None)
            else None
        )
        ratios["Gross Profit Margin"] = _safe_ratio(revenue, gross_profit, numerator=gross_profit, denominator=revenue)
        ratios["EBIT/TA"] = _safe_ratio(ebit, total_assets, numerator=ebit, denominator=total_assets)
        ratios["RE/TA"] = _safe_ratio(retained_earnings, total_assets, numerator=retained_earnings, denominator=total_assets)
        ratios["Asset Turnover"] = _safe_ratio(revenue, total_assets, numerator=revenue, denominator=total_assets)

        # Leverage / liquidity
        ratios["Debt/Equity"] = _safe_ratio(total_debt, total_equity, numerator=total_debt, denominator=total_equity)
        ratios["Debt Ratio"] = _safe_ratio(total_debt, total_assets, numerator=total_debt, denominator=total_assets)
        ratios["Cash Ratio"] = _safe_ratio(current_liab, cash, numerator=cash, denominator=current_liab)
        ratios["Working Capital Ratio"] = (
            _safe_ratio(current_liab, current_assets, numerator=current_assets, denominator=current_liab) if current_liab else None
        )
        ratios["(CA-CL)/TA"] = (
            _safe_ratio(
                total_assets, current_assets, current_liab, numerator=(current_assets - current_liab), denominator=total_assets
            )
            if (total_assets and current_assets is not None and current_liab is not None)
            else None
        )
        ratios["Book Equity/TL"] = (
            _safe_ratio(total_debt, total_equity, numerator=total_equity, denominator=(total_debt + total_equity))
            if (total_debt is not None and total_equity is not None)
            else None
        )

        # Coverage / operating efficiency
        ratios["Op. In./Interest Expense"] = (
            _safe_ratio(interest_exp, operating_income, numerator=operating_income, denominator=abs(interest_exp))
            if (interest_exp and operating_income and interest_exp != 0)
            else None
        )
        ratios["Op. In./(NWC+FA)"] = (
            _safe_ratio(
                operating_income,
                working_capital,
                fixed_assets,
                numerator=operating_income,
                denominator=(working_capital + fixed_assets),
            )
            if (operating_income and (working_capital + fixed_assets))
            else None
        )

        return ratios


# ---------------------------------------------------------------------------
# Refinitiv / LSEG fundamental field mapping + ratio computation
# ---------------------------------------------------------------------------

# Request the TR.* field codes (LSEG requires the code form in `fields=`).
# The returned DataFrame uses *display* column names (e.g. "Price Close"),
# which we map to short internal names below. Some fields are unavailable via
# this LSEG setup (EnterpriseValue, TotalStockholdersEquity, CashAndShortTermInvestments,
# NetPropertyPlantEquipment) - their ratios degrade gracefully to NaN or a
# derived proxy (see _compute_refinitiv_ratios).
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

# Display-name (returned) -> internal name used by _compute_refinitiv_ratios.
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
# Fields not returned by this LSEG setup; left as NaN (ratios referencing them
# degrade gracefully). Kept here for documentation / future vendor coverage.
REFINITIV_UNAVAILABLE = {
    "Enterprise Value": "ent_val",
    "Total Stockholders Equity": "book_equity",
    "Cash and Short Term Investments": "cash",
    "Net Property Plant and Equipment": "fixed_assets",
}


def _compute_refinitiv_ratios(raw: DataFrame) -> DataFrame:
    """Compute the 18 ``RATIO_COLUMNS`` from raw Refinitiv data items.

    Args:
        raw: DataFrame indexed by RIC (or with an ``Instrument``/``ric`` column)
            whose columns are the ``TR.*`` fields in ``REFINITIV_RAW_FIELDS``.

    Returns:
        DataFrame with the same index as ``raw`` and columns = ``RATIO_COLUMNS``.
        Ratios whose components are missing are left NaN (never zero-filled).
    """
    d = raw.rename(columns=REFINITIV_RAW_FIELDS)

    def col(name: str) -> pd.Series:
        if name not in d.columns:
            return pd.Series(np.nan, index=d.index)
        # LSEG can return text (e.g. "NM", "-") for some fundamentals; coerce.
        # Force plain float64 (not nullable Int64/Float64): the ratio math below
        # mixes columns (e.g. mkt_cap.where(..., price*shares)) and pandas 3.x
        # raises when assigning a float into a nullable-Int64 array.
        return pd.to_numeric(d[name], errors="coerce").astype(float)

    price = col("price")
    mkt_cap = col("mkt_cap")
    shares = col("shares")
    # This LSEG entitlement returns TR.MarketCapitalization as null but does
    # return PriceClose and SharesOutstanding, so derive market cap from those.
    # Falls back to the direct field when present.
    derived_mkt_cap = price * shares
    mkt_cap = mkt_cap.where(mkt_cap.notna(), derived_mkt_cap)
    ebit = col("ebit")
    revenue = col("revenue")
    net_income = col("net_income")
    total_debt = col("total_debt")
    total_assets = col("total_assets")
    current_assets = col("current_assets")
    current_liab = col("current_liab")
    total_liab = col("total_liab")
    gross_profit = col("gross_profit")
    retained_earnings = col("retained_earnings")
    interest_exp = col("interest_exp")
    operating_income = col("operating_income")
    shares = col("shares")

    def safe(s: "pd.Series") -> "pd.Series":
        # div-by-zero -> NaN, not inf
        return s.replace(0, np.nan)

    nwc = current_assets - current_liab
    # Derive book equity from the accounting identity (Total Assets - Total Liab)
    # since LSEG does not return a direct StockholdersEquity field here. This
    # unlocks the equity-based ratios without an extra round-trip.
    book_equity_derived = total_assets - total_liab

    out = pd.DataFrame(index=d.index, columns=RATIO_COLUMNS, dtype=float)
    # EV/EBIT: enterprise value is unavailable via this LSEG entitlement, so proxy
    # EV with market cap + total debt (a standard "equity value + net debt" proxy).
    # mkt_cap itself is derived (= price * shares) because TR.MarketCapitalization
    # returns null here.
    ev_proxy = mkt_cap + total_debt
    out["EV/EBIT"] = ev_proxy / safe(ebit)
    # OpInc/(NWC+FA): fixed_assets (Net PPE) is unavailable for this LSEG setup, so
    # we cannot form the NWC+FA denominator. Leave as NaN rather than guess.
    out["Op. In./(NWC+FA)"] = np.nan
    # Market-cap based multiples (price & derived mkt_cap available).
    out["P/E"] = mkt_cap / safe(net_income)
    out["P/B"] = mkt_cap / safe(book_equity_derived)
    out["P/S"] = mkt_cap / safe(revenue)
    out["Op. In./Interest Expense"] = operating_income / safe(interest_exp.abs())
    out["Working Capital Ratio"] = current_assets / safe(current_liab)
    out["RoE"] = net_income / safe(book_equity_derived)
    out["ROCE"] = ebit / safe(book_equity_derived + total_debt)
    out["Debt/Equity"] = total_debt / safe(book_equity_derived)
    out["Debt Ratio"] = total_debt / safe(total_assets)
    # Cash Ratio: cash unavailable -> NaN.
    out["Cash Ratio"] = np.nan
    out["Asset Turnover"] = revenue / safe(total_assets)
    out["Gross Profit Margin"] = gross_profit / safe(revenue)
    out["(CA-CL)/TA"] = nwc / safe(total_assets)
    out["RE/TA"] = retained_earnings / safe(total_assets)
    out["EBIT/TA"] = ebit / safe(total_assets)
    out["Book Equity/TL"] = book_equity_derived / safe(total_liab)
    # Coerce to plain float64 (pd.to_numeric may yield nullable Float64 when a
    # string was coerced); the factor/DB layer expects float64.
    return out.astype("float64")


# ---------------------------------------------------------------------------
# SimFin provider
# ---------------------------------------------------------------------------


class SimFinFundamentalsProvider(FundamentalsProvider):
    """Collect ratios via the SimFin API.

    Extends the existing :mod:`data_engineering.eod_data.simfin` stub (which
    today only fetches shares outstanding) to the full ratio set. SimFin's
    bulk/sharepoint endpoints expose the fundamentals needed for most ratios;
    fields not supplied by the free tier are left NaN.
    """

    name = "simfin"

    def __init__(self, token: Optional[str] = None):
        """Initialize the SimFin provider.

        Args:
            token: Optional SimFin API token (falls back to keyring/env).
        """
        self._token = token

    def get_panel(self, symbols: DataFrame, as_of_date: str) -> DataFrame:
        """Return the SimFin fundamentals panel for the given symbols.

        Args:
            symbols: DataFrame with a ``security_id`` column.
            as_of_date: Snapshot date (kept for interface compatibility).

        Returns:
            Wide fundamentals panel indexed by ``security_id`` (stub returns NaN).
        """
        # Placeholder integration: the SimFin share/outstanding helper already
        # exists; ratio extraction mirrors YahooFundamentalsProvider but sourced
        # from SimFin's fundamentals endpoint. Left as a structured stub because
        # it requires a SimFin API token and bulk download wiring.
        panel = _empty_panel(symbols["security_id"].tolist())
        print("[simfin fundamentals] stub: implement ratio extraction from " "SimFin fundamentals endpoint (token + bulk download).")
        return panel


# ---------------------------------------------------------------------------
# Refinitiv / LSEG provider
# ---------------------------------------------------------------------------


class RefinitivFundamentalsProvider(FundamentalsProvider):
    """Collect the 18 ratios from LSEG/Refinitiv via ``lseg.data``.

    Pulls annual fundamental data items per RIC with ``ld.get_data`` and derives
    the 18 notebook ratios via :func:`_compute_refinitiv_ratios`. Reuses the
    existing Refinitiv vendor's ticker->RIC resolution and session helper, so it
    needs the local Workspace (desktop or SSO) session open - just like the
    price vendor.

    The latest fiscal-year snapshot is returned (Refinitiv point-in-time pulls
    require extra period parameters); ``as_of_date`` is accepted for interface
    compatibility but the most recent FY is used.
    """

    name = "refinitiv"

    def __init__(
        self,
        session: Optional[Any] = None,
        orm_session: Optional[Session] = None,
        orm_engine: Optional[Engine] = None,
        frequency: str = "FY",
        start_date: Optional[str] = None,
    ):
        """Initialize the Refinitiv fundamentals provider.

        Args:
            session: Optional pre-opened LSEG session (opened lazily if omitted).
            orm_session: Optional ORM session used to resolve RICs from xref.
            orm_engine: Optional ORM engine used to resolve RICs from xref.
            frequency: Pull mode (``FY`` = single snapshot, ``Q`` = quarterly).
            start_date: Optional start date for the historical/pull mode.
        """
        # ``session`` is accepted for symmetry with other providers; the live
        # LSEG session is opened lazily inside get_panel via the vendor helper.
        # ``orm_session``/``orm_engine`` let us resolve RICs from
        # security_vendor_xref (preferred) instead of name conversion.
        # ``frequency``/``start_date`` select the historical pull mode used by
        # collect_and_store_fundamentals (FY = single snapshot, Q = quarterly).
        self._session = session
        self._orm_session = orm_session
        self._orm_engine = orm_engine
        self._frequency = frequency
        self._start_date = start_date

    def _resolve_rics(self, symbols: DataFrame) -> DataFrame:
        """Resolve *exchange-qualified* Refinitiv RICs for the symbols.

        The fundamentals endpoint (``ld.get_data`` with ``TR.*`` fields) only
        returns data for exchange-qualified RICs (e.g. ``AAPL.O``). Bare tickers
        (``AAPL``) resolve for only a handful of names, which is why earlier
        loads stored almost no ratios. So we obtain the base identifier from
        ``security_vendor_xref`` (preferred) or the symbol column, then qualify
        *every* one via LSEG ``symbol_conversion`` (which appends the correct
        exchange suffix, e.g. ``.O`` for NASDAQ). The qualified RIC is kept
        intact here -- we deliberately do NOT strip the qualifier as the price
        path does, because the fundamentals endpoint needs it.
        """
        from lseg.data.content import symbol_conversion

        out = symbols.copy()
        out["ric"] = pd.NA

        # 1) base identifier per security from xref (vendor='Refinitiv'), which
        #    stores the base ticker (e.g. AAPL); else fall back to the symbol col.
        if self._orm_session is not None and self._orm_engine is not None:
            from data_engineering.database import database as db

            sec_ids = symbols["security_id"].tolist()
            xref = db.read_security_vendor_xref(self._orm_session, self._orm_engine, vendor="Refinitiv")
            if not xref.empty:
                xref = xref[xref["security_id"].isin(sec_ids)]
                xref = xref.sort_values("is_primary", ascending=False).drop_duplicates("security_id")
                m = xref.set_index("security_id")["vendor_ticker"]
                out["ric"] = out["security_id"].map(m)

        # Fallback: use the symbol column for any still-missing rows.
        missing_mask = out["ric"].isna()
        if missing_mask.any():
            out.loc[missing_mask, "ric"] = out.loc[missing_mask, "symbol"]

        # 2) Qualify every base identifier to an exchange RIC via symbol_conversion.
        bare = out["ric"].dropna().unique().tolist()
        if bare:
            try:
                sc = symbol_conversion.Definition(
                    symbols=bare,
                    from_symbol_type=symbol_conversion.SymbolTypes.TICKER_SYMBOL,
                    to_symbol_types=[symbol_conversion.SymbolTypes.RIC],
                ).get_data()
                scdf = sc.data.df.reset_index()
                if "RIC" in scdf.columns and "index" in scdf.columns:
                    qmap = scdf.set_index("index")["RIC"].to_dict()
                    out["ric"] = out["ric"].map(lambda r: qmap.get(r, r))
                else:
                    print(f"[refinitiv fundamentals] symbol_conversion returned unexpected columns: {list(scdf.columns)}")
            except Exception as exc:  # noqa: BLE001
                print(f"[refinitiv fundamentals] symbol_conversion qualification failed: {exc}")
        return out

    def get_panel(self, symbols: DataFrame, as_of_date: str) -> DataFrame:
        """Return the latest fiscal-year fundamentals panel for the given symbols.

        Args:
            symbols: DataFrame with a ``security_id`` column.
            as_of_date: Snapshot date (kept for interface compatibility).

        Returns:
            Wide fundamentals panel indexed by ``security_id``.
        """
        return self.get_panel_history(symbols, as_of_date, frequency="FY")[0]

    def get_panel_history(
        self,
        symbols: DataFrame,
        as_of_date: str,
        frequency: str = "FY",
        start_date: Optional[str] = None,
    ) -> tuple[DataFrame, Optional[DataFrame]]:
        """Return (panel_snapshot, history_df) for the given frequency.

        * ``frequency="FY"`` -> the latest fiscal-year snapshot (original behaviour),
          returned as a wide panel indexed by security_id (history_df is None).
        * ``frequency="Q"``  -> quarterly history via ``ld.get_history`` with a real
          DatetimeIndex of fiscal period-ends. Returns:
            - panel_snapshot: the most-recent-quarter panel (security_id index), and
            - history_df: a long DataFrame with columns
              [security_id, effective_date, <RATIO_COLUMNS>] carrying every
              quarter's ratios (used for per-quarter DB storage / scoring).

        LSEG aligns quarters to each instrument's OWN fiscal year, so the period
        index is NOT shared across names; we therefore keep each observation's own
        period-end as its effective_date and never assume a common calendar grid.
        """
        import lseg.data as ld  # noqa: F401  (used via rdv session)

        from data_engineering.eod_data import refinitiv as rdv

        try:
            rdv.ensure_refinitiv_session()
        except Exception as exc:  # noqa: BLE001 - session must be open locally
            print(f"[refinitiv fundamentals] could not open LSEG session: {exc}")
            return _empty_panel(symbols["security_id"].tolist()), None

        resolved = self._resolve_rics(symbols)
        rics = resolved["ric"].dropna().unique().tolist()
        if not rics:
            return _empty_panel(symbols["security_id"].tolist()), None

        if frequency == "Q":
            return self._get_quarterly_history(resolved, rics, start_date or as_of_date)

        # ---- Annual (FY) path: original single-snapshot logic ----
        CHUNK = 100
        raw_frames = []
        try:
            for i in range(0, len(rics), CHUNK):
                chunk = rics[i : i + CHUNK]
                try:
                    raw = ld.get_data(
                        universe=chunk,
                        fields=REFINITIV_REQUEST_FIELDS,
                        parameters={"Frq": "FY"},
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[refinitiv fundamentals] get_data failed for chunk {i}: {exc}")
                    continue
                if raw is not None and not raw.empty:
                    raw_frames.append(raw)
        except Exception as exc:  # noqa: BLE001
            print(f"[refinitiv fundamentals] get_data failed: {exc}")
            return _empty_panel(symbols["security_id"].tolist()), None

        if not raw_frames:
            return _empty_panel(symbols["security_id"].tolist()), None

        raw = pd.concat(raw_frames, ignore_index=True)
        raw = raw.rename(columns={"Instrument": "ric"}).set_index("ric")
        computed = _compute_refinitiv_ratios(raw)
        panel = self._finalize_panel(computed, resolved, symbols)
        return panel, None

    def _get_quarterly_history(self, resolved: DataFrame, rics: list[str], start_date: str) -> tuple[DataFrame, Optional[DataFrame]]:
        """Pull quarterly history via ld.get_history and compute ratios per quarter."""
        import lseg.data as ld

        # Determine the history window from start_date -> today.
        sd = pd.Timestamp(start_date).strftime("%Y-%m-%d")
        ed = pd.Timestamp.today().strftime("%Y-%m-%d")
        # get_history hits the local UDF API (localhost:9005) and times out on
        # large requests; keep chunks small and retry on transient timeouts
        # instead of silently dropping the whole chunk's securities.
        CHUNK = 15
        MAX_RETRIES = 3
        RETRY_SLEEP = 5  # seconds
        long_records = []
        n_chunks = max(1, (len(rics) + CHUNK - 1) // CHUNK)
        sec_map = resolved.dropna(subset=["ric"]).set_index("ric")["security_id"].astype(int).to_dict()
        for n, i in enumerate(range(0, len(rics), CHUNK), start=1):
            chunk = rics[i : i + CHUNK]
            hist = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    hist = ld.get_history(
                        universe=chunk,
                        fields=REFINITIV_REQUEST_FIELDS,
                        parameters={"Frq": "Q", "SDate": sd, "EDate": ed},
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    if attempt < MAX_RETRIES:
                        print(
                            f"[refinitiv fundamentals] get_history chunk {i} "
                            f"(RICs {n}/{n_chunks}) attempt {attempt}/{MAX_RETRIES} "
                            f"failed ({type(exc).__name__}); retrying in {RETRY_SLEEP}s"
                        )
                        time.sleep(RETRY_SLEEP)
                    else:
                        print(
                            f"[refinitiv fundamentals] get_history chunk {i} "
                            f"(RICs {n}/{n_chunks}) FAILED after {MAX_RETRIES} attempts: {exc}"
                        )
            if hist is None or not isinstance(hist, pd.DataFrame) or hist.empty:
                print(f"[refinitiv fundamentals] chunk {i} (RICs {n}/{n_chunks}): " f"no data returned ({len(chunk)} RICs)")
                continue
            print(f"[refinitiv fundamentals] chunk {i} (RICs {n}/{n_chunks}): " f"OK, {len(hist)} rows x {hist.shape[1]} cols")
            # hist: DatetimeIndex (period-end) x MultiIndex columns (RIC, Metric).
            # Each RIC reports on its OWN fiscal-period dates (sparse), so we must
            # select per-RIC and keep only that RIC's real observation dates.
            h = hist.copy()
            h.index = pd.to_datetime(h.index)
            if not isinstance(h.columns, pd.MultiIndex):
                # Defensive: wrap a flat metric index (single RIC) as (ric, metric).
                h.columns = pd.MultiIndex.from_product([chunk[:1], h.columns])
            # level0 = RIC, level1 = metric
            for ric in h.columns.get_level_values(0).unique():
                sec_id = sec_map.get(ric)
                if sec_id is None:
                    continue
                sub = h[ric]  # Date x Metric (metric = LSEG display names)
                sub = sub.dropna(how="all")  # keep only this RIC's real fiscal dates
                if sub.empty:
                    continue
                sub = sub.copy()
                sub.index.name = "effective_date"
                comp = _compute_refinitiv_ratios(sub)
                comp["security_id"] = int(sec_id)
                comp = comp.reset_index().rename(columns={"index": "effective_date"})
                if "effective_date" not in comp.columns:
                    comp = comp.rename(columns={comp.columns[0]: "effective_date"})
                long_records.append(comp[["security_id", "effective_date"] + RATIO_COLUMNS])

        if not long_records:
            return _empty_panel(resolved["security_id"].tolist()), None
        history = pd.concat(long_records, ignore_index=True)
        # Snapshot = most recent quarter per security.
        snap = (
            history.sort_values("effective_date")
            .groupby("security_id", as_index=False)
            .tail(1)
            .set_index("security_id")[RATIO_COLUMNS]
        )
        final = _empty_panel(resolved["security_id"].tolist())
        final.loc[snap.index.intersection(final.index)] = snap
        return final, history

    @staticmethod
    def _finalize_panel(computed: DataFrame, resolved: DataFrame, symbols: DataFrame) -> DataFrame:
        sec_map = resolved.dropna(subset=["ric"]).set_index("ric")["security_id"].astype(int)
        computed = computed.join(sec_map.rename("security_id"), how="left").dropna(subset=["security_id"])
        computed["security_id"] = computed["security_id"].astype(int)
        computed = computed.set_index("security_id")[RATIO_COLUMNS]
        final = _empty_panel(symbols["security_id"].tolist())
        final.loc[computed.index.intersection(final.index)] = computed
        return final[RATIO_COLUMNS]


def get_fundamentals_provider(name: str, **kwargs: Any) -> FundamentalsProvider:
    """Instantiate a registered fundamentals provider by name."""
    registry: dict[str, type[FundamentalsProvider]] = {
        "static": StaticFundamentalsProvider,
        "yahoo": YahooFundamentalsProvider,
        "simfin": SimFinFundamentalsProvider,
        "refinitiv": RefinitivFundamentalsProvider,
    }
    key = name.lower()
    if key not in registry:
        raise KeyError(f"Unknown fundamentals provider '{name}'. Available: {sorted(registry)}")
    return registry[key](**kwargs)


# ---------------------------------------------------------------------------
# Collection -> persistence helper (production loading path)
# ---------------------------------------------------------------------------


def collect_and_store_fundamentals(
    provider_name: str,
    symbols: DataFrame,
    as_of_date: str,
    orm_session: Session,
    orm_engine: Engine,
    source_vendor: Optional[str] = None,
    provider_kwargs: Optional[dict[str, Any]] = None,
) -> int:
    """Collect ratios via a provider and persist them to dbo.security_fundamentals.

    This is the bridge that lets the DB-backed ``StaticFundamentalsProvider`` (and
    therefore ``MLReturnFactor``) score ratios from *any* vendor without the
    vendor being live at scoring time. Whatever produced the rows is recorded in
    ``source_vendor`` so they stay distinguishable.

    Args:
        provider_name: one of the registered provider keys.
        symbols: DataFrame with ``security_id`` and ``symbol`` columns.
        as_of_date: snapshot date (``YYYY-MM-DD``); recorded as effective_date.
        orm_session / orm_engine: DB handles for the write.
        source_vendor: vendor label stored on each row (defaults to provider_name).
        provider_kwargs: extra kwargs passed to the provider constructor.

    Returns:
        Number of (security, metric) rows written.
    """
    from data_engineering.database import database as db

    provider = get_fundamentals_provider(
        provider_name,
        orm_session=orm_session,
        orm_engine=orm_engine,
        **(provider_kwargs or {}),
    )
    snapshot, history = provider.get_panel_history(
        symbols,
        as_of_date,
        frequency=provider_kwargs.get("frequency", "FY") if provider_kwargs else "FY",
        start_date=provider_kwargs.get("start_date") if provider_kwargs else None,
    )
    vendor = source_vendor or provider.name
    long_rows = []

    # Quarterly history path: write one row per (security, metric, period-end).
    if history is not None and not history.empty:
        for _, rec in history.iterrows():
            sec_id = int(rec["security_id"])
            eff = pd.to_datetime(rec["effective_date"]).date()
            for metric in RATIO_COLUMNS:
                val = rec[metric]
                if pd.isna(val):
                    continue
                long_rows.append(
                    {
                        "security_id": sec_id,
                        "metric_type": metric,
                        "metric_value": float(val),
                        "source_vendor": vendor,
                        "effective_date": eff,
                        "end_date": None,
                    }
                )
        if not long_rows:
            print("[fundamentals] no non-null quarterly ratios to store.")
            return 0
        df_long = pd.DataFrame(long_rows)
        n = db.write_security_fundamentals(df_long, orm_session)
        return n

    # Annual single-snapshot path (original logic).
    if snapshot is None or snapshot.empty:
        print(f"[fundamentals] provider '{provider_name}' returned no data.")
        return 0
    panel = snapshot
    effective = pd.to_datetime(as_of_date).date()
    for sec_id, row in panel.iterrows():
        for metric in RATIO_COLUMNS:
            val = row[metric]
            if pd.isna(val):
                continue
            long_rows.append(
                {
                    "security_id": int(sec_id),
                    "metric_type": metric,
                    "metric_value": float(val),
                    "source_vendor": vendor,
                    "effective_date": effective,
                    "end_date": None,
                }
            )

    if not long_rows:
        print("[fundamentals] no non-null ratios to store.")
        return 0

    df_long = pd.DataFrame(long_rows)
    db.write_security_fundamentals(df_long, orm_session)
    print(f"[fundamentals] wrote {len(df_long)} rows from '{vendor}' (as_of {effective}).")
    return len(df_long)
