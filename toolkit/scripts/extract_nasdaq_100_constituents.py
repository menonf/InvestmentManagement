import datetime as dt
import json
from urllib.request import urlopen

import certifi
import keyring
import pandas as pd

from data_engineering.database import database as database
from data_engineering.eod_data import yahoo as yf_func

engine, connection, conn_str, session = database.get_db_connection()
        
df_securities = database.read_security_master(orm_session=session, orm_engine=engine)

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

url = ("https://financialmodelingprep.com/api/v3/nasdaq_constituent?apikey")
df_nasdaq=pd.DataFrame(get_jsonparsed_data(url))[['symbol','name','sector','subSector']]
df_nasdaq = df_nasdaq.rename(columns={'name': 'security_name'})

missing_tickers = df_nasdaq[~df_nasdaq['symbol'].isin(df_securities['symbol'])]
print(missing_tickers)

# Insert missing tickers into SecurityMaster
if not missing_tickers.empty:
    new_records = [
        database.SecurityMaster(
            security_id = None,
            symbol=row.symbol,
            isin= None,
            sedol= None,
            cusip= None,
            loanxid= None,
            security_name = row.security_name,
            country = 'United States',
            currency = 'USD',
            sector= row.sector,
            industry=row.subSector,
            security_type='Common Stock',
            asset_class='Equity',
            exchange_mic='XNAS',
            is_active=1,
            source_vendor='2',
            upsert_date= dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            upsert_by='nasdaq loader'
        )
        for _, row in missing_tickers.iterrows()
    ]
    session.add_all(new_records)
    session.commit()
    print(f"Inserted {len(new_records)} new records into SecurityMaster.")
else:
    print("No new tickers to insert.")
    
# Update is_active for existing tickers
existing_tickers = df_nasdaq['symbol'].tolist()
session.query(database.SecurityMaster)\
        .filter(database.SecurityMaster.symbol.in_(existing_tickers))\
        .update({"is_active": 1}, synchronize_session=False)

session.commit()
print("Updated is_active for existing tickers.")

df_nasdaq = df_nasdaq.merge(df_securities[['symbol', 'security_id']], on='symbol', how='left')


start_date = "2024-12-31"
end_date = "2025-06-06"

df_eod = yf_func.get_stock_data(df_nasdaq, start_date, end_date)


unique_as_of_dates = df_eod[['as_of_date']].drop_duplicates()
df_nasdaq = df_nasdaq.merge(unique_as_of_dates, how='cross')


df_nasdaq['port_id'] = 3
df_nasdaq['held_shares'] = 1
df_nasdaq = df_nasdaq[['as_of_date', 'port_id','security_id', 'held_shares']]
print(df_nasdaq)

database.write_portfolio_holdings(df_nasdaq, session)

df_securities = database.read_security_master(orm_session=session, orm_engine=engine)


df_eod = yf_func.get_historical_data(df_securities[df_securities["is_active"] == 1]["symbol"].tolist(),
                                     df_securities[df_securities["is_active"] == 1]["security_id"].tolist(), 
                                     start_date, 
                                     end_date,
                                     "1d")
database.write_market_data(df_eod, session)

df_fundamentals = yf_func.fetch_fundamentals_yahoo(df_securities[df_securities["is_active"] == 1]["symbol"].tolist(),
                                             df_securities[df_securities["is_active"] == 1]["security_id"].tolist())
database.write_security_fundamentals(df_fundamentals, session)