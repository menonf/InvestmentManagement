import time
from datetime import datetime, timedelta
from urllib import parse

import keyring
import pandas as pd
import sqlalchemy as sql
from ib_insync import *
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from data_engineering.database import db_functions as db_func


def get_trading_day():
    today = datetime.today()
    while today.weekday() >= 5:  
        today -= timedelta(days=1)
    return today.strftime('%Y-%m-%d')


# Connect to IB Gateway
ib = IB()
ib.connect('127.0.0.1', 4001, clientId=1) 

# Get account summary
account_summary = ib.accountSummary('U20761295')
account_summary_df = pd.DataFrame([{'tag': row.tag, 'value': row.value} for row in account_summary])

# Get portfolio holdings
positions = ib.positions()
portfolio_holdings= []
for pos in positions:
    contract = pos.contract
    portfolio_holdings.append({
        'as_of_date': get_trading_day(),
        'portfolio_short_name': pos.account,
        'symbol': contract.symbol,
        'security_type': contract.secType,
        'exchange': contract.exchange,
        'currency': contract.currency,
        'held_shares': pos.position,
        'avg_cost': pos.avgCost,
    })

df_portfolio_holdings_data = pd.DataFrame(portfolio_holdings)

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

df_securities = db_func.read_security_master(session, engine)
df_portfolio_data = db_func.read_portfolio(session, engine, df_portfolio_holdings_data['portfolio_short_name'].unique().tolist())

df_portfolio_market_data = pd.merge(pd.merge(df_portfolio_data, df_portfolio_holdings_data),
                                    df_securities, on='symbol', how='inner')

df_portfolio_market_data = df_portfolio_market_data[['as_of_date','port_id', 'security_id', 'held_shares']]

db_func.write_portfolio_holdings(df_portfolio_market_data, session)
ib.disconnect()