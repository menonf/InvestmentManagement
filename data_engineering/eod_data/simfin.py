"""Module for yahoo API helper functions."""

import time

import pandas as pd
import requests


def fetch_fundamentals_simfin(tickers: list[str], sec_ids: list[int]) -> pd.DataFrame:
    """
    Fetch fundamental data (e.g., common shares outstanding) for given tickers and security IDs from SimFin.

    Params:
        tickers: List of stock ticker symbols.
        sec_ids: List of corresponding security IDs.

    Returns:
        DataFrame containing fundamental data ready for database insertion.
    """
    data_list = []

    if len(tickers) != len(sec_ids):
        raise ValueError("Length of tickers and sec_ids lists must be the same.")

    for ticker, sec_id in zip(tickers, sec_ids):
        url = f"https://backend.simfin.com/api/v3/companies/common-shares-outstanding?ticker={ticker}"
        headers = {"accept": "application/json", "Authorization": "MFzWmtbTCssYk6YyGO7YSze13qmFUAWd"}

        while True:
            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()  # Raise for HTTP errors (e.g., 429)
                data = response.json()

                df = pd.json_normalize(data)
                if not df.empty:
                    df["security_id"] = sec_id
                    df["metric_type"] = "shares_outstanding"
                    df["source_vendor"] = "SimFin"
                    df["end_date"] = None
                    df = df.rename(columns={"value": "metric_value", "endDate": "effective_date"})
                    df = df[["security_id", "metric_type", "metric_value", "source_vendor", "effective_date", "end_date"]]
                    data_list.append(df)
                else:
                    print(f"[No Data] Empty response for {ticker}")
                break  # Success — break the retry loop

            except requests.exceptions.HTTPError as http_err:
                if response.status_code == 429:
                    print(f"[Quota Limit] 429 Too Many Requests for {ticker}. Retrying in 10 seconds...")
                    time.sleep(10)
                    continue  # Retry same ticker
                else:
                    print(f"[HTTP Error] {ticker}: {http_err}")
                    break  # Break and move to next ticker

            except Exception as e:
                print(f"[Error] Failed to fetch SimFin data for {ticker}: {e}")
                break  # Break and move to next ticker

    return pd.concat(data_list, ignore_index=True) if data_list else pd.DataFrame()
