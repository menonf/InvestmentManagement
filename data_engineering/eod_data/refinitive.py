"""Module for Refinitiv EOD API helper functions with ticker → RIC resolution."""

import datetime

import lseg.data as ld
import pandas as pd
import refinitiv.data as rd
from pandas import DataFrame
from refinitiv.data.content import symbol_conversion

# Constants
INTERVAL_TO_FRQ = {
    "1d": "D",
    "1wk": "W",
    "1mo": "M",
    "5d": "D",  # Refinitiv doesn't have 5d, use daily
}

REFINITIV_FIELDS = [
    "TR.OPENPRICE.Date",
    "TR.OPENPRICE",
    "TR.HIGHPRICE",
    "TR.LOWPRICE",
    "TR.CLOSEPRICE",
    "TR.CLOSEPRICE(Adjusted=1)",
    "TR.ACCUMULATEDVOLUME",
]

# Note: Refinitiv returns both TR.CLOSEPRICE and TR.CLOSEPRICE(Adjusted=1)
# as "Close Price", so we need to rename by position
COLUMN_MAPPING = {
    "Date": "as_of_date",
    "Open Price": "open",
    "High Price": "high",
    "Low Price": "low",
    "Accumulated Volume": "volume",
}

OUTPUT_COLUMNS = [
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


def ensure_refinitiv_session() -> None:
    """Open Refinitiv LSEG and RD sessions."""
    ld.open_session()
    rd.open_session()


def resolve_tickers_to_rics(symbol_df: DataFrame) -> DataFrame:
    """Resolve Yahoo-style tickers to Refinitiv RICs.

    Args:
        symbol_df: DataFrame with columns 'symbol' and 'security_id'

    Returns:
        DataFrame: Original DataFrame with added 'ric' column

    Raises:
        RuntimeError: If RIC resolution fails for all tickers
    """
    ensure_refinitiv_session()

    tickers = symbol_df["symbol"].drop_duplicates().tolist()

    response = symbol_conversion.Definition(
        symbols=tickers,
        from_symbol_type=symbol_conversion.SymbolTypes.TICKER_SYMBOL,
        to_symbol_types=[symbol_conversion.SymbolTypes.RIC],
    ).get_data()

    ric_df = response.data.df.reset_index()

    ric_df.loc[ric_df["RIC"].str.contains("ANSS", na=False), "RIC"] = "ANSS.OQ^G25"

    if ric_df.empty:
        raise RuntimeError("RIC resolution failed for all tickers")

    ric_df = ric_df.rename(columns={"index": "symbol", "RIC": "ric"})

    return symbol_df.merge(ric_df[["symbol", "ric"]], on="symbol", how="left")


def _validate_inputs(symbol_df: DataFrame, start_date: str, end_date: str) -> None:
    """Validate function inputs.

    Raises:
        ValueError: If validation fails
    """
    required_columns = ["symbol", "security_id"]
    missing_columns = [col for col in required_columns if col not in symbol_df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame must contain columns: {missing_columns}")

    try:
        start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")

        if start_dt >= end_dt:
            raise ValueError("start_date must be before end_date")

    except ValueError as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {e}")


def _get_refinitiv_frequency(interval: str) -> str:
    """Map interval to Refinitiv frequency parameter.

    Args:
        interval: Interval string (e.g., '1d', '1wk', '1mo')

    Returns:
        str: Refinitiv frequency code
    """
    frq = INTERVAL_TO_FRQ.get(interval, "D")
    if interval not in INTERVAL_TO_FRQ:
        print(f"Warning: Interval '{interval}' not supported by Refinitiv EOD. Using daily ('D') instead.")
    return frq


def _fetch_refinitiv_data(
    rics: list[str],
    start_date: str,
    end_date: str,
    frq: str,
) -> DataFrame:
    """Fetch data from Refinitiv API.

    Returns:
        DataFrame: Raw data from Refinitiv or empty DataFrame on error
    """
    try:
        return ld.get_data(
            universe=rics,
            fields=REFINITIV_FIELDS,
            parameters={
                "SDate": start_date,
                "EDate": end_date,
                "Frq": frq,
            },
        )
    except Exception as e:
        print(f"Error retrieving data from Refinitiv: {e}")
        return pd.DataFrame()


def _standardize_dataframe(
    df: DataFrame,
    symbol_df_valid: DataFrame,
    current_time: str,
    interval: str,
) -> DataFrame:
    """Standardize Refinitiv data to match Yahoo format.

    Args:
        df: Raw Refinitiv data
        symbol_df_valid: DataFrame with security_id and ric mapping
        current_time: Timestamp string
        interval: Interval string

    Returns:
        DataFrame: Standardized data
    """
    # Handle duplicate "Close Price" columns
    # Refinitiv returns both as "Close Price" - first is unadjusted, second is adjusted
    columns = df.columns.tolist()
    close_price_indices = [i for i, col in enumerate(columns) if col == "Close Price"]

    if len(close_price_indices) == 2:
        columns[close_price_indices[0]] = "close"
        columns[close_price_indices[1]] = "adj_close"
        df.columns = columns

    # Rename other columns
    df = df.rename(columns=COLUMN_MAPPING)

    # Attach security_id
    df = df.merge(symbol_df_valid[["security_id", "ric"]], left_on="Instrument", right_on="ric", how="left")

    # Add missing columns
    df["dividends"] = 0
    df["stock_splits"] = 0
    df["dataload_date"] = current_time
    df["interval"] = interval

    # Reorder columns
    return df[OUTPUT_COLUMNS]


def _track_missing_symbols(
    symbol_df_valid: DataFrame,
    df: DataFrame,
    symbols_with_no_data: list[str],
) -> None:
    """Identify and log symbols with no data returned.

    Args:
        symbol_df_valid: DataFrame with valid RICs
        df: Result DataFrame
        symbols_with_no_data: List to append missing symbols to
    """
    securities_with_data = df["security_id"].unique()
    all_securities = symbol_df_valid["security_id"].unique()
    securities_without_data = set(all_securities) - set(securities_with_data)

    if securities_without_data:
        missing_symbols = symbol_df_valid[symbol_df_valid["security_id"].isin(securities_without_data)]["symbol"].tolist()
        symbols_with_no_data.extend(missing_symbols)
        for symbol in missing_symbols:
            print(f"No data found for symbol: {symbol}")


def get_stock_price(
    symbol_df: DataFrame,
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> DataFrame:
    """Retrieve EOD stock data from Refinitiv using resolved RICs.

    Args:
        symbol_df: DataFrame with columns 'symbol' and 'security_id'.
        start_date: Start date in format YYYY-MM-DD.
        end_date: End date in format YYYY-MM-DD.
        interval: Data interval (default "1d"). Supported: "1d" (daily), "1wk" (weekly), "1mo" (monthly).
                  Maps to Refinitiv Frq parameter: 1d→D, 1wk→W, 1mo→M.

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
        >>> historical_data = get_stock_data_refinitiv_eod(
        ...     symbol_df=symbols_df,
        ...     start_date='2023-01-01',
        ...     end_date='2023-12-31',
        ...     interval='1d'
        ... )

        # Recent/latest data (e.g., last 30 days)
        >>> from datetime import datetime, timedelta
        >>> end_date = datetime.now().strftime('%Y-%m-%d')
        >>> start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        >>> recent_data = get_stock_data_refinitiv_eod(
        ...     symbol_df=symbols_df,
        ...     start_date=start_date,
        ...     end_date=end_date,
        ...     interval='1d'
        ... )

    Note:
        - Failed/delisted symbols are logged and skipped
        - Data is rounded to 4 decimal places
        - Returns empty DataFrame if no valid data found
        - dividends and stock_splits columns are set to 0 (not available from Refinitiv EOD API)
    """
    # Validate inputs
    _validate_inputs(symbol_df, start_date, end_date)

    symbols_with_no_data = []
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    frq = _get_refinitiv_frequency(interval)

    # Resolve tickers to RICs
    try:
        symbol_df = resolve_tickers_to_rics(symbol_df)
    except Exception as e:
        print(f"Error resolving tickers to RICs: {e}")
        return pd.DataFrame()

    # Track symbols without RIC
    symbols_without_ric = symbol_df[symbol_df["ric"].isna()]["symbol"].tolist()
    if symbols_without_ric:
        symbols_with_no_data.extend(symbols_without_ric)
        for symbol in symbols_without_ric:
            print(f"No RIC found for symbol: {symbol}")

    # Filter to valid RICs only
    symbol_df_valid = symbol_df[symbol_df["ric"].notna()].copy()

    if symbol_df_valid.empty:
        print("Warning: No valid data retrieved for any symbol")
        return pd.DataFrame()

    # Fetch data from Refinitiv
    rics = symbol_df_valid["ric"].drop_duplicates().tolist()
    df = _fetch_refinitiv_data(rics, start_date, end_date, frq)

    if df.empty:
        print("Warning: No valid data retrieved for any symbol")
        return pd.DataFrame()

    # Standardize to Yahoo format
    df = _standardize_dataframe(df, symbol_df_valid, current_time, interval)

    # Track symbols with no data
    _track_missing_symbols(symbol_df_valid, df, symbols_with_no_data)

    if symbols_with_no_data:
        print(f"symbols with no data: {symbols_with_no_data}")

    return df.round(4)
