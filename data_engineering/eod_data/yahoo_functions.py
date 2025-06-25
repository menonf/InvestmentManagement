"""Module for yahoo API helper functions."""

import datetime
import time

import pandas as pd
import requests
import yfinance as yf
from pandas import DataFrame


def get_historical_data(tickers: list[str], sec_id: list[str], start_date: str, end_date: str, interval: str) -> DataFrame:
    """Retrieve historical stock data from Yahoo Finance for multiple tickers.

    Params:
        tickers: A list of ticker symbols for the stocks to retrieve data for.
        sec_id: A list of security identifiers for the stocks, corresponding to the tickers.
        start_date: The start date of the historical data in the format YYYY-MM-DD.
        end_date: The end date of the historical data in the format YYYY-MM-DD.

    Returns:
        DataFrame containing the historical data for the specified tickers, with the following columns:

    Raises:
        ValueError: If the lengths of the tickers and sec_id lists do not match.

    Details:
        - This function uses the yfinance library to retrieve data from Yahoo Finance.
        - The function handles potential errors, such as 404 errors for invalid tickers or delisted symbols.
        - The function returns a DataFrame with rounded values to four decimal places.
    """
    tickers_with_no_data = []
    dataframes = []

    if len(tickers) != len(sec_id):
        raise ValueError("Length of tickers and sec_id lists must be the same.")

    for ticker, sec in zip(tickers, sec_id):
        try:
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")
            stock = yf.Ticker(ticker)
            # print(stock.fast_info.currency)
            # print(stock.fast_info.shares)
            # print(stock.fast_info.last_price)
            # print(stock.fast_info.market_cap)
            # print(stock.fast_info.last_price * stock.fast_info.shares)
            historical_data = stock.history(start=start_date, end=end_date, period=interval, auto_adjust=False)
            # historical_data = yf.download(ticker, start=start_date, end=end_date, period=interval, auto_adjust=True)

            if not historical_data.empty:
                historical_data.insert(0, "as_of_date", historical_data.index.date)
                historical_data.insert(1, "security_id", sec)
                historical_data.rename(
                    columns={
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Adj Close": "adj_close",
                        "Volume": "volume",
                        "Dividends": "dividends",
                        "Stock Splits": "stock_splits",
                    },
                    inplace=True,
                )
                dataframes.append(historical_data)
            else:
                tickers_with_no_data.append(ticker)

        except Exception as e:
            error_message = str(e)
            if "404 Client Error" in error_message or "symbol may be delisted" in error_message:
                tickers_with_no_data.append(ticker)
                print(f"An error occurred for symbol {ticker}:", error_message)
            else:
                print(f"An error occurred for symbol {ticker}:", error_message)
            continue

    df_all_historical_data = pd.concat(dataframes)
    df_all_historical_data = df_all_historical_data.reset_index()
    df_all_historical_data["dataload_date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_all_historical_data["interval"] = "1d"
    df_all_historical_data = df_all_historical_data.reset_index(drop=True)

    if len(tickers_with_no_data) > 1:
        print("Tickers that have no data ", tickers_with_no_data)
    return df_all_historical_data.round(4)


def fetch_latest_data(tickers: list[str], sec_id: list[str]) -> DataFrame:
    """Fetch the latest daily stock data for the given tickers and security IDs.

    Params:
        tickers: A list of stock ticker symbols.
        sec_id: A list of security IDs corresponding to the tickers.

    Returns:
        A DataFrame containing the latest daily stock data, formatted for ORM DB insertion.

    Raises:
        ValueError: If the lengths of the tickers and sec_id lists do not match.
        Exception: If an error occurs while fetching data for a ticker.

    Details:
        - Fetches data using Yahoo Finance API.
        - Handles potential errors, including 404 errors and delisted symbols.
        - Formats the DataFrame with additional columns:
            - AsOfDate: The date of the data.
            - SecID: The security ID.
            - Dataload_Date: The timestamp when the data was fetched.
        - Rounds numerical values to 4 decimal places.

    """
    dataframes = []

    if len(tickers) != len(sec_id):
        raise ValueError("Length of tickers and sec_id lists must be the same.")

    for ticker, sec in zip(tickers, sec_id):
        try:
            stock = yf.Ticker(ticker)
            latest_data = stock.history(period="1d")

            # To get dataframe in the ORM DB form to bulk insert to DB
            if not latest_data.empty:
                latest_data.insert(0, "AsOfDate", latest_data.index.date)
                latest_data.insert(1, "SecID", sec)
                latest_data.rename(columns={"Stock Splits": "Stock_Splits"}, inplace=True)
                dataframes.append(latest_data)

        except Exception as e:
            error_message = str(e)
            if "404 Client Error" in error_message or "symbol may be delisted" in error_message:
                print(f"No data found for symbol {ticker}.")
            else:
                print(f"An error occurred for symbol {ticker}:", error_message)
            continue

    df_latest_data = pd.concat(dataframes)
    df_latest_data["Interval"] = "1d"
    df_latest_data["Dataload_Date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_latest_data = df_latest_data.reset_index(drop=True)

    return df_latest_data.round(4)


def fetch_fundamentals_yahoo(tickers: list[str], sec_ids: list[int]) -> pd.DataFrame:
    """Fetch shares outstanding and market cap for given tickers and security IDs.

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
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            for key in ["sharesOutstanding", "marketCap"]:
                value = info.get(key)
                if value is not None and isinstance(value, (int, float)):
                    data_list.append(
                        {
                            "security_id": sec_id,
                            "metric_type": key,
                            "metric_value": float(value),
                            "source_vendor": "Yahoo Finance",
                            "effective_date": datetime.date.today(),
                            "end_date": None,
                        }
                    )

        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")
            continue

    return pd.DataFrame(data_list)


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
