import time
from urllib import parse

import keyring
import pandas as pd
import sqlalchemy as sql
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from analytics.performance import performance_analytics as perf
from analytics.risk import risk_analytics as risk
from data_engineering.database import db_functions as db_func

# Secure credentials
service_name = "ihub_sql_connection"
db = keyring.get_password(service_name, "db")
db_user = keyring.get_password(service_name, "uid")
db_password = keyring.get_password(service_name, "pwd")

# Build connection string
connection_string = f"Driver={{ODBC Driver 18 for SQL Server}};"\
                    f"Server=tcp:ops-store-server.database.windows.net,1433;"\
                    f"Database={db};Uid={db_user};Pwd={db_password};"\
                    f"Encrypt=yes;TrustServerCertificate=no;"
connection_params = parse.quote_plus(connection_string)


max_retries = 3
retry_interval_minutes = 2

for attempt in range(1, max_retries + 1):
    try:
        engine = sql.create_engine("mssql+pyodbc:///?odbc_connect=%s" % connection_params)
        connection = engine.connect()
        session = Session(engine)
        print("Database connection successful.")
        break
    except OperationalError as e:
        print(f"Attempt {attempt} failed with error:\n{e}")
        if attempt < max_retries:
            print(f"Retrying in {retry_interval_minutes} minutes...")
            time.sleep(retry_interval_minutes * 60)
        else:
            print("All retry attempts failed. Exiting.")
            raise

start_date = "2025-01-01"
end_date = "2025-06-06"

# Calculate Portfolio Performance
portfolio_short_names = ["Nasdaq100","QQQ"]
portfolio_market_data = db_func.get_portfolio_market_data(session, engine, start_date, end_date, portfolio_short_names)
portfolio_asset_returns = perf.calculate_portfolio_constituent_returns(portfolio_market_data, "adj_close")["returns"]
portfolio_asset_weights = perf.calculate_portfolio_constituent_weights(portfolio_market_data, "adj_close", "market_weighted")
portfolio_return = (portfolio_asset_returns * portfolio_asset_weights).T.groupby(level=0).sum().T

portfolio_total_return = portfolio_asset_returns * portfolio_asset_weights
perf.plot_cumulative_returns(portfolio_total_return.T.groupby(level=0).sum().T)

# Calculate Portfolio Risk
max_date_index = portfolio_asset_weights.index.max()
df_portfolio_var = []

for PortfolioShortName in portfolio_short_names:
    portfolio_returns = portfolio_asset_returns[PortfolioShortName]
    portfolio_weights = portfolio_asset_weights[PortfolioShortName]
    portfolio_latest_weights = portfolio_asset_weights[max_date_index:max_date_index][PortfolioShortName]

    obj_risk = risk.PortfolioVaR(portfolio_asset_returns,
                                 portfolio_latest_weights,
                                 PortfolioShortName,
                                 lookback_days=24,
                                 horizon_days=1,
                                 confidence_interval=0.99)

    var_result = obj_risk.calculate_var()
    df_var = pd.DataFrame.from_dict(var_result, orient="columns")
    df_portfolio_var.append(df_var)

print(pd.concat(df_portfolio_var))
