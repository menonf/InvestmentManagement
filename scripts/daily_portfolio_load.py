from datetime import datetime, timedelta

import keyring
import pandas as pd
from ib_insync import *

from data_engineering.database import db_functions as database


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

engine, connection, session = database.get_db_connection()

df_securities = database.read_security_master(session, engine)
df_portfolio_data = database.read_portfolio(session, engine, df_portfolio_holdings_data['portfolio_short_name'].unique().tolist())

df_portfolio_market_data = pd.merge(pd.merge(df_portfolio_data, df_portfolio_holdings_data),
                                    df_securities, on='symbol', how='inner')

df_portfolio_market_data = df_portfolio_market_data[['as_of_date','port_id', 'security_id', 'held_shares']]

database.write_portfolio_holdings(df_portfolio_market_data, session)
ib.disconnect()