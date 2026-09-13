"""Tiingo EOD price vendor.

Public API (unchanged):
    get_stock_price(symbol_df, token, start_date, end_date) -> (df, no_data)
"""

from __future__ import annotations

import pandas as pd
import requests
from pandas import DataFrame

from .base import STANDARD_COLUMNS, PriceVendor

_TIINGO_COLUMN_MAP = {
    "date": "as_of_date",
    "adjClose": "adj_close",
    "divCash": "dividends",
    "splitFactor": "stock_splits",
}


class TiingoVendor(PriceVendor):
    """Tiingo end-of-day price vendor.

    Fetches EOD bars from the Tiingo REST API using an API token.
    """

    name = "tiingo"
    base_url = "https://api.tiingo.com/tiingo/daily"

    def __init__(self, token: str):
        """Initialize the vendor with a Tiingo API token."""
        self.token = token

    @classmethod
    def from_env(cls, env_var: str = "TIINGO_API_TOKEN") -> TiingoVendor:
        """Build a vendor instance from an environment variable.

        Args:
            env_var: Name of the environment variable holding the API token.

        Returns:
            A configured :class:`TiingoVendor`.
        """
        import os

        tok = os.environ.get(env_var)
        if not tok:
            raise RuntimeError(f"{env_var} not set")
        return cls(tok)

    def _fetch_raw(self, symbol_df: DataFrame, start_date: str, end_date: str, interval: str) -> tuple[DataFrame, DataFrame]:
        headers = {"Content-Type": "application/json"}
        frames, missing = [], []
        for _, row in symbol_df.iterrows():
            symbol, sec = row["symbol"], row["security_id"]
            url = f"{self.base_url}/{symbol}/prices" f"?startDate={start_date}&endDate={end_date}&token={self.token}"
            try:
                resp = requests.get(url, headers=headers, timeout=30)
                resp.raise_for_status()
                data = pd.DataFrame(resp.json())
                if data.empty:
                    missing.append(symbol)
                    print(f"No data found for symbol: {symbol}")
                    continue
                data = data.copy()
                data.insert(0, "security_id", sec)
                frames.append(data)
            except Exception as exc:  # noqa: BLE001
                missing.append(symbol)
                print(f"Error retrieving data for {symbol}: {exc}")
        combined = pd.concat(frames, ignore_index=True).round(4) if frames else pd.DataFrame()
        return combined, missing

    def _map_columns(self, raw_df: DataFrame) -> DataFrame:
        df = raw_df.rename(columns=_TIINGO_COLUMN_MAP)
        df["interval"] = "1d"
        out = [c for c in STANDARD_COLUMNS if c in df.columns]
        return df[out]


def get_stock_price(symbol_df: DataFrame, token: str, start_date: str, end_date: str) -> DataFrame:
    """Fetch EOD prices from Tiingo (backward-compatible signature/return)."""
    vendor = TiingoVendor(token)
    data, no_data = vendor.fetch(symbol_df, start_date, end_date, interval="1d")
    if data.empty and no_data.empty:
        data = pd.DataFrame(columns=STANDARD_COLUMNS)
    return data, no_data
