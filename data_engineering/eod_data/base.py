"""Shared base class and helpers for end-of-day (EOD) price vendors.

Goal: every vendor (Yahoo, Refinitiv, Tiingo, Marketstack, SimFin) returns the
*exact same* standardized DataFrame so the rest of the engine is vendor-agnostic.

Standardized schema (columns, in order):
    as_of_date, security_id, open, high, low, close, adj_close,
    volume, dividends, stock_splits, dataload_date, interval

Public entry point for callers that do not care about vendor specifics:

    from data_engineering.eod_data import get_vendor
    vendor = get_vendor("yahoo")          # or "refinitiv", "tiingo", ...
    df, no_data = vendor.fetch(symbols, start_date, end_date, interval="1d")

Each vendor subclass implements only:
    - ``name`` (class attribute)
    - ``_fetch_raw(symbols, start, end, interval) -> (raw_df, missing_symbols)``
    - optionally ``_map_columns(raw_df) -> df`` if its field names differ.

The base class handles validation, column standardization, rounding,
dataload_date stamping and missing-symbol accounting.
"""

from __future__ import annotations

import abc
import datetime

import pandas as pd
from pandas import DataFrame

# The canonical output schema every vendor must produce.
STANDARD_COLUMNS = [
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

NUMERIC_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]

REQUIRED_INPUT_COLUMNS = ["symbol", "security_id"]


def _validate_symbols(symbol_df: DataFrame) -> None:
    """Raise ``ValueError`` if the input frame lacks required columns."""
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in symbol_df.columns]
    if missing:
        raise ValueError(f"DataFrame must contain columns: {missing}")


def _validate_dates(start_date: str, end_date: str) -> None:
    """Raise ``ValueError`` on malformed dates or inverted range."""
    try:
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {exc}")
    if start >= end:
        raise ValueError("start_date must be before end_date")


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class PriceVendor(abc.ABC):
    """Abstract base for EOD price vendors.

    Subclasses implement the vendor-specific fetch and (optionally) column
    mapping. The base class guarantees a uniform output contract.
    """

    #: Unique, lowercase vendor key (e.g. ``"yahoo"``).
    name: str = "base"

    def fetch(
        self,
        symbol_df: DataFrame,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> tuple[DataFrame, DataFrame]:
        """Fetch EOD data for every symbol in ``symbol_df``.

        Args:
            symbol_df: DataFrame with ``symbol`` and ``security_id`` columns.
            start_date: Inclusive start date ``YYYY-MM-DD``.
            end_date: Inclusive end date ``YYYY-MM-DD``.
            interval: Bar interval (default ``"1d"``).

        Returns:
            ``(data, no_data)`` where ``data`` follows :data:`STANDARD_COLUMNS`
            and ``no_data`` lists symbols for which nothing was returned.
        """
        _validate_symbols(symbol_df)
        _validate_dates(start_date, end_date)

        requested = set(symbol_df["symbol"])
        raw_df, raw_missing = self._fetch_raw(symbol_df, start_date, end_date, interval)

        if raw_df is None or raw_df.empty:
            no_data = self._build_no_data(symbol_df, requested)
            return pd.DataFrame(columns=STANDARD_COLUMNS), no_data

        df = self._standardize(raw_df, symbol_df, interval)
        no_data = self._build_no_data(symbol_df, set(df["symbol"]) if "symbol" in df else set())
        return df, no_data

    # -- vendor hooks -----------------------------------------------------

    @abc.abstractmethod
    def _fetch_raw(
        self,
        symbol_df: DataFrame,
        start_date: str,
        end_date: str,
        interval: str,
    ) -> tuple[DataFrame, list[str]]:
        """Return ``(raw_df, missing_symbol_list)``.

        ``raw_df`` is vendor-shaped; it will be standardized afterwards.
        """
        raise NotImplementedError

    def _map_columns(self, raw_df: DataFrame) -> DataFrame:
        """Override to rename vendor fields to the canonical schema.

        Default assumes the raw frame already uses canonical column names
        (aside from ``symbol`` / ``security_id`` keying).
        """
        return raw_df

    # -- internal standardization ----------------------------------------

    def _standardize(self, raw_df: DataFrame, symbol_df: DataFrame, interval: str) -> DataFrame:
        df = self._map_columns(raw_df.copy())

        # Ensure security_id is present (vendor may have supplied only symbol).
        if "security_id" not in df.columns and "symbol" in df.columns:
            sec_map = dict(zip(symbol_df["symbol"], symbol_df["security_id"]))
            df["security_id"] = df["symbol"].map(sec_map)

        # Stamp metadata.
        now = _now()
        df["dataload_date"] = now
        df["interval"] = interval

        # Default dividend / split columns if the vendor does not provide them.
        for col, default in (("dividends", 0), ("stock_splits", 0)):
            if col not in df.columns:
                df[col] = default

        # Round numerics.
        numeric = [c for c in NUMERIC_COLUMNS if c in df.columns]
        if numeric:
            df[numeric] = df[numeric].round(4)

        # Order / subset to the canonical schema.
        ordered = [c for c in STANDARD_COLUMNS if c in df.columns]
        df = df[ordered]
        return df

    @staticmethod
    def _build_no_data(symbol_df: DataFrame, returned_symbols: set[str]) -> DataFrame:
        missing = [s for s in symbol_df["symbol"] if s not in returned_symbols]
        if not missing:
            return pd.DataFrame(columns=["symbol", "security_id"])
        sub = symbol_df[symbol_df["symbol"].isin(missing)][["symbol", "security_id"]]
        return sub.drop_duplicates().reset_index(drop=True)
