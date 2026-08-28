"""Module for Marketstack API helper functions."""

import datetime
import os
import requests
import pandas as pd
from pandas import DataFrame


def get_stock_price_marketstack(
    symbol_df: DataFrame,
    start_date: str,
    end_date: str,
    api_key: str,
    interval: str = "1d",
) -> tuple[DataFrame, DataFrame]:
    """Retrieve stock data from Marketstack for multiple symbols within a specified date range.

    Args:
        symbol_df: DataFrame with columns 'symbol' and 'security_id'.
        start_date: Start date in format YYYY-MM-DD.
        end_date: End date in format YYYY-MM-DD.
        api_key: Marketstack API key.
        interval: Only '1d' supported (Marketstack free tier is EOD).

    Returns:
        tuple[DataFrame, DataFrame]:
            - Stock data with columns: as_of_date, security_id, open, high, low, close,
              adj_close, volume, dataload_date, interval.
            - DataFrame with symbols that had no data.
    """

    required_columns = ["symbol", "security_id"]
    missing_columns = [col for col in required_columns if col not in symbol_df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame must contain columns: {missing_columns}")

    if interval != "1d":
        raise ValueError("Marketstack free tier supports only daily (1d) data.")

    # Validate date format
    try:
        start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        if start_dt >= end_dt:
            raise ValueError("start_date must be before end_date")
    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    symbols = symbol_df["symbol"].tolist()
    symbol_map = dict(zip(symbol_df["symbol"], symbol_df["security_id"]))

    base_url = "https://api.marketstack.com/v1/eod"  # HTTPS required on most plans

    all_data = []
    offset = 0
    limit = 1000

    while True:
        params = {
            "access_key": api_key,
            "symbols": ",".join(symbols),
            "date_from": start_date,
            "date_to": end_date,
            "limit": limit,
            "offset": offset,
        }

        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Request to Marketstack failed: {e}")

        data = response.json()

        # Surface API-level errors instead of silently breaking
        if "error" in data:
            error_info = data["error"]
            raise ValueError(f"Marketstack API error: {error_info.get('message', error_info)}")

        if "data" not in data or not data["data"]:
            print("No data returned by API for the given parameters.")
            break

        for item in data["data"]:
            symbol = item.get("symbol")
            if symbol is None:
                continue

            all_data.append(
                {
                    "as_of_date": item["date"][:10],
                    "security_id": symbol_map.get(symbol),
                    "open": item.get("open"),
                    "high": item.get("high"),
                    "low": item.get("low"),
                    "close": item.get("close"),
                    "adj_close": item.get("adj_close"),
                    "volume": item.get("volume"),
                    "dataload_date": current_time,
                    "interval": interval,
                }
            )

        print(f"Fetched {len(data['data'])} records at offset {offset}.")

        if len(data["data"]) < limit:
            break

        offset += limit

    df_combined = pd.DataFrame(all_data)

    if not df_combined.empty:
        numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
        df_combined[numeric_cols] = df_combined[numeric_cols].round(4)

    # Identify symbols with no data
    if not df_combined.empty:
        reverse_map = {v: k for k, v in symbol_map.items()}
        symbols_returned = set(df_combined["security_id"].map(reverse_map).dropna())
    else:
        symbols_returned = set()

    symbols_with_no_data = [{"symbol": sym, "security_id": symbol_map[sym]} for sym in symbols if sym not in symbols_returned]

    df_no_data = pd.DataFrame(symbols_with_no_data)

    if df_combined.empty:
        print("Warning: No valid data retrieved.")

    if not df_no_data.empty:
        print(f"Symbols with no data: {df_no_data['symbol'].tolist()}")

    return df_combined, df_no_data


if __name__ == "__main__":
    # Load API key from environment variable (recommended)
    # Set it in your shell: export MARKETSTACK_API_KEY="your_key_here"
    API_KEY = "58b4ebed24df3f7b1bddbfc40c257aff"

    # Test symbols
    symbols_df = pd.DataFrame(
        {
            "symbol": ["MSFT"],
            "security_id": ["SEC001"],
        }
    )

    start_date = "2025-01-01"
    end_date = "2025-01-31"

    print("Fetching data from Marketstack...\n")

    try:
        df_prices, df_no_data = get_stock_price_marketstack(
            symbol_df=symbols_df,
            start_date=start_date,
            end_date=end_date,
            api_key=API_KEY,
            interval="1d",
        )

        print("\n=== DATA RETRIEVED ===")
        print(f"Rows: {len(df_prices)}")
        print(df_prices.head(), "\n")

        print("=== SYMBOLS WITH NO DATA ===")
        if df_no_data.empty:
            print("None")
        else:
            print(df_no_data)

    except Exception as e:
        print(f"Test failed: {e}")
