"""Marketstack EOD price vendor.

Public API (unchanged):
    get_stock_price_marketstack(symbol_df, start_date, end_date, api_key, interval="1d")
        -> (df, no_data)
"""

from __future__ import annotations

import pandas as pd
import requests

from .base import STANDARD_COLUMNS, PriceVendor

_BASE_URL = "https://api.marketstack.com/v1/eod"


class MarketstackVendor(PriceVendor):
    """Marketstack end-of-day price vendor.

    Fetches EOD bars from the Marketstack REST API using an API key.
    """

    name = "marketstack"
    base_url = _BASE_URL

    def __init__(self, api_key: str):
        """Initialize the vendor with a Marketstack API key."""
        self.api_key = api_key

    @classmethod
    def from_env(cls, env_var: str = "MARKETSTACK_API_KEY") -> MarketstackVendor:
        """Build a vendor instance from an environment variable.

        Args:
            env_var: Name of the environment variable holding the API key.

        Returns:
            A configured :class:`MarketstackVendor`.
        """
        import os

        key = os.environ.get(env_var)
        if not key:
            raise RuntimeError(f"{env_var} not set")
        return cls(key)

    def _fetch_raw(self, symbol_df: pd.DataFrame, start_date: str, end_date: str, interval: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        symbols = symbol_df["symbol"].tolist()
        symbol_map = dict(zip(symbol_df["symbol"], symbol_df["security_id"]))
        limit, offset = 1000, 0
        all_rows, missing = [], []

        while True:
            params = {
                "access_key": self.api_key,
                "symbols": ",".join(symbols),
                "date_from": start_date,
                "date_to": end_date,
                "limit": limit,
                "offset": offset,
            }
            try:
                resp = requests.get(self.base_url, params=params, timeout=30)
                resp.raise_for_status()
            except requests.exceptions.RequestException as exc:
                raise ConnectionError(f"Request to Marketstack failed: {exc}")

            payload = resp.json()
            if "error" in payload:
                info = payload["error"]
                raise ValueError(f"Marketstack API error: {info.get('message', info)}")
            if "data" not in payload or not payload["data"]:
                print("No data returned by API for the given parameters.")
                break

            for item in payload["data"]:
                sym = item.get("symbol")
                if sym is None:
                    continue
                all_rows.append(
                    {
                        "as_of_date": item["date"][:10],
                        "security_id": symbol_map.get(sym),
                        "open": item.get("open"),
                        "high": item.get("high"),
                        "low": item.get("low"),
                        "close": item.get("close"),
                        "adj_close": item.get("adj_close"),
                        "volume": item.get("volume"),
                        "interval": interval,
                    }
                )
            print(f"Fetched {len(payload['data'])} records at offset {offset}.")
            if len(payload["data"]) < limit:
                break
            offset += limit

        df = pd.DataFrame(all_rows)
        returned = set(df["security_id"].map({v: k for k, v in symbol_map.items()}).dropna()) if not df.empty else set()
        missing = [s for s in symbols if s not in returned]
        return df, missing


def get_stock_price_marketstack(symbol_df: pd.DataFrame, start_date: str, end_date: str, api_key: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch EOD prices from Marketstack (backward-compatible signature/return)."""
    vendor = MarketstackVendor(api_key)
    data, no_data = vendor.fetch(symbol_df, start_date, end_date, interval)
    if data.empty and no_data.empty:
        data = pd.DataFrame(columns=STANDARD_COLUMNS)
    return data, no_data
