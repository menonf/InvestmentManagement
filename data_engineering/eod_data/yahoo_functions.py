"""Module for yahoo API helper functions."""

import datetime

import pandas as pd
import yfinance as yf
from pandas import DataFrame


def get_stock_data(ticker_df: DataFrame, start_date: str, end_date: str, interval: str = "1d") -> DataFrame:
    """Retrieve stock data from Yahoo Finance for multiple tickers within a specified date range.

    Args:
        ticker_df: DataFrame with columns 'ticker' and 'security_id'.
        start_date: Start date in format YYYY-MM-DD.
        end_date: End date in format YYYY-MM-DD.
        interval: Data interval (default "1d"). Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo.

    Returns:
        DataFrame: Stock data with columns: as_of_date, security_id, open, high, low, close,
                  adj_close, volume, dividends, stock_splits, dataload_date, interval.

    Raises:
        ValueError: If required columns 'ticker' and 'security_id' are missing from ticker_df.

    Examples:
        # Historical data (e.g., last year)
        >>> import pandas as pd
        >>> tickers_df = pd.DataFrame({
        ...     'ticker': ['AAPL', 'GOOGL', 'MSFT'],
        ...     'security_id': ['SEC001', 'SEC002', 'SEC003']
        ... })
        >>> historical_data = get_stock_data(
        ...     ticker_df=tickers_df,
        ...     start_date='2023-01-01',
        ...     end_date='2023-12-31',
        ...     interval='1d'
        ... )

        # Recent/latest data (e.g., last 30 days)
        >>> from datetime import datetime, timedelta
        >>> end_date = datetime.now().strftime('%Y-%m-%d')
        >>> start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        >>> recent_data = get_stock_data(
        ...     ticker_df=tickers_df,
        ...     start_date=start_date,
        ...     end_date=end_date,
        ...     interval='1d'
        ... )

        # Intraday data (latest trading session)
        >>> today = datetime.now().strftime('%Y-%m-%d')
        >>> intraday_data = get_stock_data(
        ...     ticker_df=tickers_df,
        ...     start_date=today,
        ...     end_date=today,
        ...     interval='5m'  # 5-minute intervals
        ... )

    Note:
        - Failed/delisted tickers are logged and skipped
        - Data is rounded to 4 decimal places
        - Returns empty DataFrame if no valid data found
    """
    # Validate input DataFrame
    required_columns = ["ticker", "security_id"]
    missing_columns = [col for col in required_columns if col not in ticker_df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame must contain columns: {missing_columns}")

    tickers_with_no_data = []
    dataframes = []

    # Validate date format once
    try:
        start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")

        if start_dt >= end_dt:
            raise ValueError("start_date must be before end_date")

    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for _, row in ticker_df.iterrows():
        ticker = row["ticker"]
        sec = row["security_id"]
        try:
            stock = yf.Ticker(ticker)
            historical_data = stock.history(start=start_date, end=end_date, interval=interval, auto_adjust=False)

            if not historical_data.empty:
                # Prepare DataFrame
                historical_data = historical_data.copy()
                historical_data.insert(0, "as_of_date", historical_data.index.date)
                historical_data.insert(1, "security_id", sec)

                # Standardize column names
                column_mapping = {
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                    "Dividends": "dividends",
                    "Stock Splits": "stock_splits",
                }
                historical_data.rename(columns=column_mapping, inplace=True)

                # Add metadata
                historical_data["dataload_date"] = current_time
                historical_data["interval"] = interval

                dataframes.append(historical_data)
            else:
                tickers_with_no_data.append(ticker)
                print(f"No data found for ticker: {ticker}")

        except Exception as e:
            error_message = str(e)
            tickers_with_no_data.append(ticker)

            if "404 Client Error" in error_message or "symbol may be delisted" in error_message:
                print(f"Ticker {ticker} may be invalid or delisted: {error_message}")
            else:
                print(f"Error retrieving data for {ticker}: {error_message}")

    # Combine all data
    if not dataframes:
        print("Warning: No valid data retrieved for any ticker")
        return pd.DataFrame()

    df_combined = pd.concat(dataframes, ignore_index=True)

    if tickers_with_no_data:
        print(f"Tickers with no data: {tickers_with_no_data}")

    return df_combined.round(4)


def fetch_fundamentals(securities_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Fetch fundamental data for given securities.

    Parameters
    ----------
    securities_df : pd.DataFrame
        DataFrame with columns 'ticker' and 'security_id'.
    metrics : list[str], optional
        List of metric keys to fetch from Yahoo Finance info.
        If None, defaults to ["sharesOutstanding", "marketCap"].

    Returns
    -------
    pd.DataFrame
        DataFrame containing fundamental data ready for database insertion.

    Examples
    --------
    >>> import pandas as pd
    >>> securities = pd.DataFrame({
    ...     'ticker': ['AAPL', 'MSFT'],
    ...     'security_id': [1, 2]
    ... })

    Default behavior (sharesOutstanding and marketCap):
    >>> result = fetch_fundamentals(securities)

    Custom metrics:
    >>> result = fetch_fundamentals(
    ...     securities,
    ...     ["sharesOutstanding", "marketCap", "totalRevenue"]
    ... )

    Single metric:
    >>> result = fetch_fundamentals(securities, ["marketCap"])

    Extended set of metrics:
    >>> result = fetch_fundamentals(securities, [
    ...     "sharesOutstanding",
    ...     "marketCap",
    ...     "totalRevenue",
    ...     "totalDebt",
    ...     "totalCash",
    ...     "bookValue"
    ... ])

    Real usage example:
    >>> # Get fundamentals data
    >>> if not df_active_securities.empty:
    ...     df_fundamentals = fetch_fundamentals(
    ...         securities_df=df_active_securities,
    ...         metrics=["sharesOutstanding", "marketCap"]
    ...     )
    """
    # Validate input DataFrame
    required_columns = {"ticker", "security_id"}
    if not required_columns.issubset(securities_df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    if securities_df.empty:
        return pd.DataFrame()

    data_list = []

    for _, row in securities_df.iterrows():
        ticker = row["ticker"]
        sec_id = row["security_id"]

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            for key in metrics:
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
