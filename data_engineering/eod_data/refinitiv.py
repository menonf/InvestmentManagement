"""Refinitiv (LSEG) EOD price vendor.

Public API (unchanged for backward compatibility):
    get_stock_price(symbol_df, start_date, end_date, interval="1d") -> DataFrame
        Single standardized DataFrame (no separate no_data frame).

Session handling: requires an open LSEG desktop session. The notebook opens it
before calling this function; :func:`ensure_refinitiv_session` is idempotent.
"""

from __future__ import annotations

import datetime

import lseg.data as ld
import pandas as pd
from lseg.data.content import symbol_conversion
from pandas import DataFrame

from .base import STANDARD_COLUMNS, PriceVendor

# Map interval -> Refinitiv frequency code. Refinitiv has no 5d; fall back to D.
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

# Refinitiv returns both adjusted & unadjusted close as "Close Price".
_COLUMN_MAPPING = {
    "Date": "as_of_date",
    "Open Price": "open",
    "High Price": "high",
    "Low Price": "low",
    "Accumulated Volume": "volume",
    "TR.DivUnadjustedNet": "dividends",
    "TR.AdjmtFactorAdjustmentFactor": "stock_splits",
}

# Data-driven RIC overrides (ticker -> RIC) for symbols Refinitiv mis-resolves.
RIC_OVERRIDES = {"ANSS": "ANSS.OQ^G25"}


def ensure_refinitiv_session() -> None:
    """Open the LSEG (ld) desktop session; optionally the legacy rd session."""
    ld.open_session()
    try:
        import refinitiv.data as rd  # optional legacy lib

        rd.open_session()
    except Exception:
        pass


def _normalize_ric(ric: str) -> str:
    """Strip a trailing exchange qualifier so a resolved RIC matches the xref.

    Strip a trailing exchange qualifier so a resolved RIC matches the
    ``vendor_ticker`` stored in ``security_vendor_xref``.

    Refinitiv's ticker->RIC conversion returns the *exchange-qualified* form
    (e.g. ``MSFT.O`` or ``MSFT.OQ`` for NASDAQ, ``IBM.N`` for NYSE). Our
    ``security_vendor_xref`` rows store the qualified form, but the qualifier
    length varies (``.O`` vs ``.OQ``), so a naive 2-char strip misses the 3-char
    ``.OQ`` names. This strips any trailing ``.<exchange>`` suffix (all trailing
    alpha chars after a ``.``) on both sides of the join, so the notebook's
    ``ric_to_secid`` map connects them regardless of which qualifier Refinitiv
    emitted. Index-level / special RICs (e.g. ``'.SPX'``) are left intact.
    """
    if not isinstance(ric, str) or not ric:
        return ric
    # Leave index-level / special RICs (e.g. '.SPX') intact.
    if ric.startswith("."):
        return ric
    head, sep, tail = ric.rpartition(".")
    if sep and head and tail.isalpha():
        return head
    return ric


def resolve_tickers_to_rics(symbol_df: DataFrame) -> DataFrame:
    """Resolve Yahoo-style tickers to Refinitiv RICs (adds a ``ric`` column)."""
    ensure_refinitiv_session()
    tickers = symbol_df["symbol"].drop_duplicates().tolist()
    response = symbol_conversion.Definition(
        symbols=tickers,
        from_symbol_type=symbol_conversion.SymbolTypes.TICKER_SYMBOL,
        to_symbol_types=[symbol_conversion.SymbolTypes.RIC],
    ).get_data()
    ric_df = response.data.df.reset_index()
    if ric_df.empty:
        raise RuntimeError("RIC resolution failed for all tickers")
    ric_df = ric_df.rename(columns={"index": "symbol", "RIC": "ric"})

    # Apply data-driven overrides.
    ric_df["ric"] = ric_df.apply(lambda r: RIC_OVERRIDES.get(r["symbol"], r["ric"]), axis=1)
    # Normalise the resolved RIC to its base (qualifier-stripped) form so it
    # aligns with the keys built from security_vendor_xref downstream.
    ric_df["ric"] = ric_df["ric"].apply(_normalize_ric)
    return symbol_df.merge(ric_df[["symbol", "ric"]], on="symbol", how="left")


class RefinitivVendor(PriceVendor):
    """Refinitiv (LSEG) end-of-day price vendor.

    Requires an open LSEG desktop/SSO session. Returns a single standardized
    DataFrame (no separate no-data frame).
    """

    name = "refinitiv"

    def fetch(self, symbol_df: DataFrame, start_date: str, end_date: str, interval: str = "1d") -> tuple[DataFrame, DataFrame]:
        """Fetch EOD prices for the given symbols.

        Args:
            symbol_df: DataFrame with ``symbol`` and ``security_id`` columns.
            start_date: ISO start date.
            end_date: ISO end date.
            interval: Bar interval (default ``1d``).

        Returns:
            Standardized EOD DataFrame.
        """
        # Refinitiv keeps its original single-DataFrame contract (no no_data).
        _validate_inputs(symbol_df, start_date, end_date)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        frq = _get_refinitiv_frequency(interval)

        # Skip RIC resolution when the caller already supplied a valid 'ric'
        # column (e.g. vendor_ticker pulled straight from security_vendor_xref,
        # which is already a Refinitiv RIC). The external resolver is fragile
        # and breaks the whole fetch when it errors.
        if "ric" not in symbol_df.columns or symbol_df["ric"].isna().all():
            try:
                symbol_df = resolve_tickers_to_rics(symbol_df)
            except Exception as exc:  # noqa: BLE001
                print(f"Error resolving tickers to RICs: {exc}")
                return pd.DataFrame(columns=STANDARD_COLUMNS), pd.DataFrame(columns=["symbol", "security_id"])

        no_ric = symbol_df[symbol_df["ric"].isna()]["symbol"].tolist()
        for s in no_ric:
            print(f"No RIC found for symbol: {s}")
        valid = symbol_df[symbol_df["ric"].notna()].copy()
        if valid.empty:
            print("Warning: No valid data retrieved for any symbol")
            return pd.DataFrame(columns=STANDARD_COLUMNS), pd.DataFrame(columns=["symbol", "security_id"])

        rics = valid["ric"].drop_duplicates().tolist()
        # Refinitiv rejects very large universes in a single get_data call, so
        # batch the RIC list into chunks.
        raw_frames = []
        for i in range(0, len(rics), 500):
            chunk = rics[i : i + 500]
            raw_frames.append(_fetch_refinitiv_data(chunk, start_date, end_date, frq))
        raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
        if raw.empty:
            print("Warning: No valid data retrieved for any symbol")
            return pd.DataFrame(columns=STANDARD_COLUMNS), pd.DataFrame(columns=["symbol", "security_id"])

        df = _standardize_dataframe(raw, valid, now, interval)
        returned = set(df["security_id"])
        missing = [s for s in valid["symbol"] if valid.loc[valid["symbol"] == s, "security_id"].iloc[0] not in returned]
        if missing:
            print(f"symbols with no data: {missing}")
        return df.round(4), pd.DataFrame(columns=["symbol", "security_id"])

    def _fetch_raw(
        self, symbol_df: DataFrame, start_date: str, end_date: str, interval: str
    ) -> tuple[DataFrame, DataFrame]:  # pragma: no cover
        raise NotImplementedError("Use RefinitivVendor.fetch directly.")


_VENDOR = RefinitivVendor()


def get_stock_price(symbol_df: DataFrame, start_date: str, end_date: str, interval: str = "1d") -> DataFrame:
    """Fetch EOD prices from Refinitiv (backward-compatible single-frame return)."""
    vendor = RefinitivVendor()
    data, _no_data = vendor.fetch(symbol_df, start_date, end_date, interval)
    return data


# --- internal helpers retained from prior implementation -----------------


def _validate_inputs(symbol_df: DataFrame, start_date: str, end_date: str) -> None:
    missing = [c for c in ("symbol", "security_id") if c not in symbol_df.columns]
    if missing:
        raise ValueError(f"DataFrame must contain columns: {missing}")
    try:
        s = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        e = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        if s >= e:
            raise ValueError("start_date must be before end_date")
    except ValueError as exc:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {exc}")


def _get_refinitiv_frequency(interval: str) -> str:
    if interval not in INTERVAL_TO_FRQ:
        print(f"Warning: Interval '{interval}' not supported by Refinitiv EOD. Using daily ('D').")
    return INTERVAL_TO_FRQ.get(interval, "D")


def _fetch_refinitiv_data(rics: list[str], start_date: str, end_date: str, frq: str) -> DataFrame:
    try:
        return ld.get_data(
            universe=rics,
            fields=REFINITIV_FIELDS,
            parameters={"SDate": start_date, "EDate": end_date, "Frq": frq},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error retrieving data from Refinitiv: {exc}")
        return pd.DataFrame()


def _standardize_dataframe(df: DataFrame, symbol_df_valid: DataFrame, current_time: str, interval: str) -> DataFrame:
    columns = df.columns.tolist()
    close_idx = [i for i, c in enumerate(columns) if c == "Close Price"]
    if len(close_idx) == 2:
        columns[close_idx[0]] = "close"
        columns[close_idx[1]] = "adj_close"
        df.columns = columns
    df = df.rename(columns=_COLUMN_MAPPING)
    # Refinitiv echoes the QUALIFIED RIC as `Instrument` (e.g. `AAPL.O`, `JPM.N`),
    # while the caller's `ric` may be either the qualified form or a base/bare
    # ticker (e.g. `AAPL`, `JPM`). Normalize BOTH sides to their bare-ticker root
    # before the join so they always align -- the original code merged the raw
    # strings and silently dropped every NASDAQ(.O/.OQ)/NYSE(.N) constituent
    # (and SPY/.SPX) whose echoed Instrument didn't byte-match `ric`.
    df["_inst_norm"] = df["Instrument"].map(_normalize_ric)
    _valid = symbol_df_valid[["security_id", "ric"]].copy()
    _valid["_ric_norm"] = _valid["ric"].map(_normalize_ric)
    df = df.merge(_valid, left_on="_inst_norm", right_on="_ric_norm", how="left")
    df = df.drop(columns=["_inst_norm", "_ric_norm"])
    df["dividends"] = 0
    df["stock_splits"] = 0
    df["dataload_date"] = current_time
    df["interval"] = interval
    return df[
        [
            "as_of_date",
            "security_id",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "dividends",
            "stock_splits",
            "dataload_date",
            "interval",
        ]
    ]
