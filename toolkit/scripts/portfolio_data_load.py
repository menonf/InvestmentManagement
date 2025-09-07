
import pandas as pd

from analytics.performance import performance_analytics as perf
from analytics.risk import risk_analytics as risk
from data_engineering.database import db_functions as database

engine, connection, conn_str, session = database.get_db_connection()

start_date = "2025-01-01"
end_date = "2025-06-06"

# Calculate Portfolio Performance
portfolio_short_names = ["Nasdaq100","QQQ"]
portfolio_market_data = database.get_portfolio_market_data(session, engine, start_date, end_date, portfolio_short_names)
portfolio_asset_returns = perf.calculate_portfolio_constituent_returns(portfolio_market_data, "adj_close")["returns"]
portfolio_asset_weights = perf.calculate_portfolio_constituent_weights(portfolio_market_data, "adj_close", "market_weighted")
portfolio_return = (portfolio_asset_returns * portfolio_asset_weights).T.groupby(level=0).sum().T

portfolio_total_return = portfolio_asset_returns * portfolio_asset_weights
perf.plot_cumulative_returns(portfolio_total_return.T.groupby(level=0).sum().T)

# Calculate Portfolio Risk
max_date_index = portfolio_asset_weights.index.max()
