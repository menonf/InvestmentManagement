import requests
import datetime
import pandas as pd


def get_stock_price(symbol_df, token, start_date, end_date):
    """
    Fetch historical data for multiple symbols from Tiingo API.

    Args:
        symbol_df: DataFrame with 'symbol' and 'security_id' columns
        token: API token for Tiingo
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        Combined DataFrame with historical data
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
    headers = {"Content-Type": "application/json"}

    for _, row in symbol_df.iterrows():
        symbol = row["symbol"]
        sec = row["security_id"]
        try:
            url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate={start_date}&endDate={end_date}&token={token}"
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            historical_data = pd.DataFrame(response.json())

            if not historical_data.empty:
                historical_data = historical_data.copy()
                historical_data.insert(0, "security_id", sec)
                historical_data["dataload_date"] = current_time
                dataframes.append(historical_data)
            else:
                symbols_with_no_data.append({"symbol": symbol, "security_id": sec})
                print(f"No data found for symbol: {symbol}")

        except Exception as e:
            symbols_with_no_data.append({"symbol": symbol, "security_id": sec})
            print(f"Error retrieving data for {symbol}: {str(e)}")

    # Combine all data
    if not dataframes:
        print("Warning: No valid data retrieved for any symbol")
        df_combined = pd.DataFrame()
    else:
        df_combined = pd.concat(dataframes, ignore_index=True).round(4)

    df_no_data = pd.DataFrame(symbols_with_no_data)

    if not df_no_data.empty:
        print(f"symbols with no data: {df_no_data['symbol'].tolist()}")

    # rename columns and keep only the desired set
    df_combined = df_combined.rename(
        columns={"date": "as_of_date", "adjClose": "adj_close", "divCash": "dividends", "splitFactor": "stock_splits"}
    )

    if "interval" not in df_combined.columns:
        df_combined["interval"] = "1d" if "daily" in url else None

    # re‑order/subset to the target column list
    df_combined = df_combined[
        [
            "security_id",
            "as_of_date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "dividends",
            "stock_splits",
            "interval",
            "dataload_date",
        ]
    ]

    return df_combined, df_no_data


# Usage example
if __name__ == "__main__":
    token = "81f0783ae3d1756869af76d72b52a86f08e2ca15"

    # Create a DataFrame with required columns
    symbol_df = pd.DataFrame({"symbol": ["AAPL", "GOOGL", "ANSS", "INVALID"], "security_id": [1, 2, 3, 4]})

    # Call the function with proper arguments
    data, no_data = get_stock_price(
        symbol_df=symbol_df,
        token=token,
        start_date="2025-01-01",
        end_date="2025-01-04",
    )
    print(data)
    print(no_data)
