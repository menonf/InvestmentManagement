"""Module for yahoo API helper functions."""

import datetime

import pandas as pd
import yfinance as yf
from pandas import DataFrame


def get_stock_data(symbol_df: DataFrame, start_date: str, end_date: str, interval: str = "1d") -> DataFrame:
    """Retrieve stock data from Yahoo Finance for multiple symbols within a specified date range.

    Args:
        symbol_df: DataFrame with columns 'symbol' and 'security_id'.
        start_date: Start date in format YYYY-MM-DD.
        end_date: End date in format YYYY-MM-DD.
        interval: Data interval (default "1d"). Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo.

    Returns:
        DataFrame: Stock data with columns: as_of_date, security_id, open, high, low, close,
                  adj_close, volume, dividends, stock_splits, dataload_date, interval.

    Raises:
        ValueError: If required columns 'symbol' and 'security_id' are missing from symbol_df.

    Examples:
        # Historical data (e.g., last year)
        >>> import pandas as pd
        >>> symbols_df = pd.DataFrame({
        ...     'symbol': ['AAPL', 'GOOGL', 'MSFT'],
        ...     'security_id': ['SEC001', 'SEC002', 'SEC003']
        ... })
        >>> historical_data = get_stock_data(
        ...     symbol_df=symbols_df,
        ...     start_date='2023-01-01',
        ...     end_date='2023-12-31',
        ...     interval='1d'
        ... )

        # Recent/latest data (e.g., last 30 days)
        >>> from datetime import datetime, timedelta
        >>> end_date = datetime.now().strftime('%Y-%m-%d')
        >>> start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        >>> recent_data = get_stock_data(
        ...     symbol_df=symbols_df,
        ...     start_date=start_date,
        ...     end_date=end_date,
        ...     interval='1d'
        ... )

        # Intraday data (latest trading session)
        >>> today = datetime.now().strftime('%Y-%m-%d')
        >>> intraday_data = get_stock_data(
        ...     symbol_df=symbols_df,
        ...     start_date=today,
        ...     end_date=today,
        ...     interval='5m'  # 5-minute intervals
        ... )

    Note:
        - Failed/delisted symbols are logged and skipped
        - Data is rounded to 4 decimal places
        - Returns empty DataFrame if no valid data found
    """
    # Validate input DataFrame
    required_columns = ["symbol", "security_id"]
    missing_columns = [col for col in required_columns if col not in symbol_df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame must contain columns: {missing_columns}")

    symbols_with_no_data = []
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

    for _, row in symbol_df.iterrows():
        symbol = row["symbol"]
        sec = row["security_id"]
        try:
            stock = yf.Ticker(symbol)
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
                symbols_with_no_data.append(symbol)
                print(f"No data found for symbol: {symbol}")

        except Exception as e:
            error_message = str(e)
            symbols_with_no_data.append(symbol)

            if "404 Client Error" in error_message or "symbol may be delisted" in error_message:
                print(f"Symbol {symbol} may be invalid or delisted: {error_message}")
            else:
                print(f"Error retrieving data for {symbol}: {error_message}")

    # Combine all data
    if not dataframes:
        print("Warning: No valid data retrieved for any symbol")
        return pd.DataFrame()

    df_combined = pd.concat(dataframes, ignore_index=True)

    if symbols_with_no_data:
        print(f"symbols with no data: {symbols_with_no_data}")

    return df_combined.round(4)


def fetch_fundamentals(securities_df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Fetch fundamental data for given securities.

    Parameters
    ----------
    securities_df : pd.DataFrame
        DataFrame with columns 'symbol' and 'security_id'.
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
    ...     'symbol': ['AAPL', 'MSFT'],
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
    required_columns = {"symbol", "security_id"}
    if not required_columns.issubset(securities_df.columns):
        raise ValueError(f"Input DataFrame must contain columns: {required_columns}")

    if securities_df.empty:
        return pd.DataFrame()

    data_list = []

    for _, row in securities_df.iterrows():
        symbol = row["symbol"]
        sec_id = row["security_id"]

        try:
            stock = yf.symbol(symbol)
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
            print(f"Failed to fetch data for {symbol}: {e}")
            continue

    return pd.DataFrame(data_list)
