"""SimFin fundamentals (common shares outstanding) vendor helper."""

import os
import time
from typing import Optional

import pandas as pd
import requests  # type: ignore[import-untyped]

try:  # config package may not be importable in all contexts
    from config.secrets import simfin_token
except Exception:  # pragma: no cover - fallback path

    def simfin_token() -> Optional[str]:  # type: ignore
        """Return the SimFin API token from the environment."""
        return os.environ.get("SIMFIN_API_TOKEN")


def fetch_fundamentals_simfin(tickers: list[str], sec_ids: list[int], token: Optional[str] = None) -> pd.DataFrame:
    """Fetch common shares outstanding from SimFin.

    Args:
        tickers: List of stock ticker symbols.
        sec_ids: Corresponding security IDs (same order/length as ``tickers``).
        token: SimFin API token. If omitted, resolved via keyring/env
            (``SIMFIN_API_TOKEN``). Never hardcode tokens in source.

    Returns:
        DataFrame ready for DB insertion with columns:
        security_id, metric_type, metric_value, source_vendor, effective_date, end_date.
    """
    if len(tickers) != len(sec_ids):
        raise ValueError("Length of tickers and sec_ids lists must be the same.")

    token = token or simfin_token()
    if not token:
        raise RuntimeError("SimFin token not provided and SIMFIN_API_TOKEN is unset.")

    data_list = []
    for ticker, sec_id in zip(tickers, sec_ids):
        url = f"https://backend.simfin.com/api/v3/companies/common-shares-outstanding?ticker={ticker}"
        headers = {"accept": "application/json", "Authorization": token}

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
