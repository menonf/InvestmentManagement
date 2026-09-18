"""Yahoo Finance EOD price vendor.

Public API (unchanged for backward compatibility):
    get_stock_price(symbol_df, start_date, end_date, interval="1d")
        -> (DataFrame[standard schema], DataFrame[no_data])

New unified entry point:
    from data_engineering.eod_data import get_vendor
    get_vendor("yahoo").fetch(symbol_df, start_date, end_date)
"""

from __future__ import annotations

import datetime

import pandas as pd
import yfinance as yf
from pandas import DataFrame

from .base import NUMERIC_COLUMNS, STANDARD_COLUMNS, PriceVendor

_YAHOO_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "stock_splits",
}


class YahooVendor(PriceVendor):
    """Yahoo Finance end-of-day price vendor.

    Fetches EOD bars via ``yfinance`` and returns the standardized schema.
    """

    name = "yahoo"

    def _fetch_raw(self, symbol_df: DataFrame, start_date: str, end_date: str, interval: str) -> tuple[DataFrame, DataFrame]:
        dataframes = []
        missing = []
        for _, row in symbol_df.iterrows():
            symbol = row["symbol"]
            sec = row["security_id"]
            try:
                hist = yf.Ticker(symbol).history(start=start_date, end=end_date, interval=interval, auto_adjust=False)
                if hist.empty:
                    missing.append(symbol)
                    print(f"No data found for symbol: {symbol}")
                    continue
                frame = hist.copy()
                frame.insert(0, "as_of_date", frame.index.date)
                frame.insert(1, "security_id", sec)
                dataframes.append(frame)
            except Exception as exc:  # noqa: BLE001 - surface per-symbol failures
                missing.append(symbol)
                msg = str(exc)
                if "404 Client Error" in msg or "symbol may be delisted" in msg:
                    print(f"Symbol {symbol} may be invalid or delisted: {msg}")
                else:
                    print(f"Error retrieving data for {symbol}: {msg}")

        if not dataframes:
            print("Warning: No valid data retrieved for any symbol")
            return pd.DataFrame(), missing

        combined = pd.concat(dataframes, ignore_index=True)
        combined = combined.rename(columns=_YAHOO_COLUMN_MAP)
        combined["dataload_date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        combined["interval"] = interval
        combined[NUMERIC_COLUMNS] = combined[NUMERIC_COLUMNS].round(4)
        return combined, missing


_VENDOR = YahooVendor()


def get_stock_price(symbol_df: DataFrame, start_date: str, end_date: str, interval: str = "1d") -> tuple[DataFrame, DataFrame]:
    """Fetch EOD prices from Yahoo (backward-compatible tuple return)."""
    data, no_data = _VENDOR.fetch(symbol_df, start_date, end_date, interval)
    if data.empty:
        data = pd.DataFrame(columns=STANDARD_COLUMNS)
    return data, no_data
