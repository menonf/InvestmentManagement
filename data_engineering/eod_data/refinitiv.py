"""Refinitiv (LSEG) end-of-day price vendor.

Implements the :class:`~data_engineering.eod_data.base.PriceVendor` template:
``_fetch_raw`` pulls ``TR.*`` price fields per RIC, ``_map_columns`` renames
LSEG's display names, and the base class standardises the result.

Callers may pass a ``ric`` column in ``symbol_df`` (e.g. ``vendor_ticker`` from
``security_vendor_xref``). When absent, tickers are resolved to RICs via LSEG
``symbol_conversion``.

Backward-compatible facade:
    get_stock_price(symbol_df, start_date, end_date, interval="1d") -> DataFrame
"""

from __future__ import annotations

import logging

import pandas as pd
from pandas import DataFrame

from data_engineering.refinitiv import ensure_session, qualify_tickers, strip_exchange_qualifier

from .base import STANDARD_COLUMNS, PriceVendor

log = logging.getLogger(__name__)

#: interval -> Refinitiv frequency code (no 5-day bars; fall back to daily).
INTERVAL_TO_FRQ = {"1d": "D", "1wk": "W", "1mo": "M", "5d": "D"}

REFINITIV_FIELDS = [
    "TR.OPENPRICE.Date",
    "TR.OPENPRICE",
    "TR.HIGHPRICE",
    "TR.LOWPRICE",
    "TR.CLOSEPRICE(Adjusted=0)",
    "TR.CLOSEPRICE(Adjusted=1)",
    "TR.ACCUMULATEDVOLUME",
    "TR.DivUnadjustedNet",
    "TR.AdjmtFactorAdjustmentFactor",
]

_COLUMN_MAPPING = {
    "Date": "as_of_date",
    "Open Price": "open",
    "High Price": "high",
    "Low Price": "low",
    "Accumulated Volume": "volume",
}

#: Universe size per ``get_data`` call; LSEG rejects very large universes.
REQUEST_CHUNK = 500


def ensure_refinitiv_session() -> None:
    """Open the LSEG session (kept for callers of the old name)."""
    ensure_session()


def resolve_tickers_to_rics(symbol_df: DataFrame) -> DataFrame:
    """Add a ``ric`` column by qualifying ``symbol`` tickers via LSEG.

    The exchange-qualified form (``AAPL.O``) is kept because ``ld.get_data``
    returns sparse or empty rows for bare US tickers. Tickers the conversion
    service cannot resolve are passed through unchanged (and logged).
    """
    qmap = qualify_tickers(symbol_df["symbol"].drop_duplicates().tolist())
    out = symbol_df.copy()
    out["ric"] = out["symbol"].map(lambda t: qmap.get(t, t))
    return out


def _get_frequency(interval: str) -> str:
    if interval not in INTERVAL_TO_FRQ:
        log.warning("interval '%s' not supported by Refinitiv EOD; using daily", interval)
    return INTERVAL_TO_FRQ.get(interval, "D")


class RefinitivVendor(PriceVendor):
    """LSEG/Refinitiv EOD prices via ``ld.get_data``. Requires LSEG Workspace."""

    name = "refinitiv"

    def _fetch_raw(self, symbol_df: DataFrame, start_date: str, end_date: str, interval: str) -> tuple[DataFrame, list[str]]:
        ensure_session()
        import lseg.data as ld

        if "ric" not in symbol_df.columns or symbol_df["ric"].isna().all():
            try:
                symbol_df = resolve_tickers_to_rics(symbol_df)
            except Exception as exc:  # noqa: BLE001 - resolver is fragile; report and stop
                log.error("resolving tickers to RICs failed: %s", exc)
                return pd.DataFrame(), symbol_df["symbol"].tolist()

        no_ric = symbol_df[symbol_df["ric"].isna()]
        for sym in no_ric["symbol"]:
            log.warning("no RIC found for symbol %s", sym)
        valid = symbol_df[symbol_df["ric"].notna()].copy()
        if valid.empty:
            return pd.DataFrame(), symbol_df["symbol"].tolist()

        rics = valid["ric"].drop_duplicates().tolist()
        frames = []
        for i in range(0, len(rics), REQUEST_CHUNK):
            chunk = rics[i : i + REQUEST_CHUNK]
            try:
                frames.append(
                    ld.get_data(
                        universe=chunk,
                        fields=REFINITIV_FIELDS,
                        parameters={"SDate": start_date, "EDate": end_date, "Frq": _get_frequency(interval)},
                    )
                )
            except Exception as exc:  # noqa: BLE001
                log.error("Refinitiv get_data failed for chunk starting at %d: %s", i, exc)
        raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if raw.empty:
            return raw, no_ric["symbol"].tolist() + valid["symbol"].tolist()

        raw = self._attach_security_ids(raw, valid)
        raw = raw.dropna(subset=["security_id", "as_of_date"])
        raw["security_id"] = raw["security_id"].astype(int)
        returned = set(raw["security_id"])
        missing = no_ric["symbol"].tolist() + [s for s, sid in zip(valid["symbol"], valid["security_id"]) if int(sid) not in returned]
        if missing:
            log.info("symbols with no data: %s", missing)
        return raw, missing

    @staticmethod
    def _attach_security_ids(raw: DataFrame, valid: DataFrame) -> DataFrame:
        """Rename LSEG columns and join ``security_id`` on the *bare* RIC.

        LSEG echoes the qualified RIC as ``Instrument`` (``AAPL.O``) while the
        caller's ``ric`` may be qualified or bare; both sides are reduced to the
        bare ticker before joining so no listing is dropped.
        """
        columns = raw.columns.tolist()
        close_idx = [i for i, c in enumerate(columns) if c == "Close Price"]
        if len(close_idx) == 2:  # unadjusted first, adjusted second
            columns[close_idx[0]] = "close"
            columns[close_idx[1]] = "adj_close"
            raw.columns = columns
        raw = raw.rename(columns=_COLUMN_MAPPING)
        raw["_key"] = raw["Instrument"].map(strip_exchange_qualifier)
        keys = valid[["security_id", "ric"]].copy()
        keys["_key"] = keys["ric"].map(strip_exchange_qualifier)
        keys = keys.drop_duplicates("_key")
        raw = raw.merge(keys[["security_id", "_key"]], on="_key", how="left").drop(columns=["_key"])
        raw["dividends"] = 0
        raw["stock_splits"] = 0
        return raw

    def _map_columns(self, raw_df: DataFrame) -> DataFrame:
        return raw_df[[c for c in STANDARD_COLUMNS if c in raw_df.columns]]


def get_stock_price(symbol_df: DataFrame, start_date: str, end_date: str, interval: str = "1d") -> DataFrame:
    """Fetch EOD prices from Refinitiv (single-frame return kept for existing callers)."""
    data, _no_data = RefinitivVendor().fetch(symbol_df, start_date, end_date, interval)
    return data
