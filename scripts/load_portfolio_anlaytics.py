from urllib import parse

import pandas as pd
import sqlalchemy as sql
from sqlalchemy.orm import Session

from analytics.performance import performance_analytics as perf
from analytics.risk import risk_analytics as risk
from data_engineering.database import db_functions as db_func
from data_engineering.eod_data import yahoo_functions as yf_func

connection_string = "Driver={ODBC Driver 17 for SQL Server};\
                    Server=(local);\
                    Database=iHub;\
                    Trusted_Connection=yes;"

connection_params = parse.quote_plus(connection_string)
engine = sql.create_engine("mssql+pyodbc:///?odbc_connect=%s" % connection_params)
connection = engine.connect()
session = Session(engine)

df_securities = db_func.read_security_master(orm_session=session, orm_engine=engine)
df_eod = yf_func.fetch_latest_data(df_securities["Ticker"].tolist(), df_securities["SecID"].tolist())
df_eod = yf_func.get_historical_data(df_securities["Ticker"].tolist(), df_securities["SecID"].tolist(), "2023-10-01", "2023-10-31")
db_func.write_market_data(df_eod, session)


portfolio_short_names = ["QQQ", "GTEF"]
portfolio_market_data = db_func.get_portfolio_market_data(session, engine, "2023-10-01", "2023-10-31", portfolio_short_names)

portfolio_asset_returns = perf.calculate_portfolio_asset_returns(portfolio_market_data, "Close")["LogReturns"]
portfolio_asset_weights = perf.calculate_portfolio_asset_weights(portfolio_market_data, "Close", "price_weighted")
portfolio_asset_weights = yf_func.calculate_weights(portfolio_market_data, "Close", "equal_weighted")
portfolio_return = (portfolio_asset_returns * portfolio_asset_weights).groupby(level=0, axis=1).sum()

max_date_index = portfolio_asset_weights.index.max()
df_portfolio_var = []

for PortfolioShortName in portfolio_short_names:
    portfolio_returns = portfolio_asset_returns[PortfolioShortName]
    portfolio_weights = portfolio_asset_weights[PortfolioShortName]
    portfolio_latest_weights = portfolio_asset_weights[max_date_index:max_date_index][PortfolioShortName]
    portfolio_var = risk.calculate_portfolio_var(portfolio_returns, portfolio_latest_weights, PortfolioShortName)

    df_var = pd.DataFrame.from_dict(portfolio_var, orient="columns")
    df_portfolio_var.append(df_var)


portfolio_total_return = portfolio_asset_returns * portfolio_asset_weights
perf.plot_cumulative_returns(portfolio_total_return.groupby(level=0, axis=1).sum())
print(pd.concat(df_portfolio_var))
