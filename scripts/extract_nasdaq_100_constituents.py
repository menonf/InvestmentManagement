import datetime as dt
import json
from urllib import parse
from urllib.request import urlopen

import certifi
import pandas as pd
import sqlalchemy as sql
from sqlalchemy.orm import Session

from data_engineering.database import db_functions as db_func
from data_engineering.eod_data import yahoo_functions as yf_func

connection_string = "Driver={ODBC Driver 18 for SQL Server};\
                    Server=tcp:ops-store-server.database.windows.net,1433;\
                    Database=ihub;\
                    Uid=dbomanager;\
                    Pwd=Managemyserver123;\
                    Encrypt=yes;\
                    TrustServerCertificate=no;"

connection_params = parse.quote_plus(connection_string)
engine = sql.create_engine("mssql+pyodbc:///?odbc_connect=%s" % connection_params)
connection = engine.connect()
session = Session(engine)

df_securities = db_func.read_security_master(orm_session=session, orm_engine=engine)

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

url = ("https://financialmodelingprep.com/api/v3/nasdaq_constituent?apikey=eaa83187bd18f51cf2c84830f00e89ae")
df_nasdaq=pd.DataFrame(get_jsonparsed_data(url))[['symbol','name','sector','subSector']]
df_nasdaq = df_nasdaq.rename(columns={'name': 'security_name'})

missing_tickers = df_nasdaq[~df_nasdaq['symbol'].isin(df_securities['symbol'])]
print(missing_tickers)

# Insert missing tickers into SecurityMaster
if not missing_tickers.empty:
    new_records = [
        db_func.SecurityMaster(
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
session.query(db_func.SecurityMaster)\
        .filter(db_func.SecurityMaster.symbol.in_(existing_tickers))\
        .update({"is_active": 1}, synchronize_session=False)

session.commit()
print("Updated is_active for existing tickers.")

df_nasdaq = df_nasdaq.merge(df_securities[['symbol', 'security_id']], on='symbol', how='left')


start_date = "2025-01-01"
end_date = "2025-03-01"
df_eod = yf_func.get_historical_data(df_securities[df_securities["is_active"] == 1]["symbol"].tolist(),
                                     df_securities[df_securities["is_active"] == 1]["security_id"].tolist(),
                                     start_date,
                                     end_date,
                                     "1d")

unique_as_of_dates = df_eod[['as_of_date']].drop_duplicates()
df_nasdaq = df_nasdaq.merge(unique_as_of_dates, how='cross')


df_nasdaq['port_id'] = 3
df_nasdaq['held_shares'] = 1
df_nasdaq = df_nasdaq[['as_of_date', 'port_id','security_id', 'held_shares']]
print(df_nasdaq)

db_func.write_portfolio_holdings(df_nasdaq, session)

df_securities = db_func.read_security_master(orm_session=session, orm_engine=engine)


df_eod = yf_func.get_historical_data(df_securities[df_securities["is_active"] == 1]["symbol"].tolist(),
                                     df_securities[df_securities["is_active"] == 1]["security_id"].tolist(), 
                                     start_date, 
                                     end_date,
                                     "1d")
# df_eod = yf_func.fetch_latest_data(df_securities[df_securities["is_active"] == 1]["symbol"].tolist(),
#                                     df_securities[df_securities["is_active"] == 1]["security_id"].tolist())
df_fundamentals = yf_func.fetch_fundamentals(df_securities[df_securities["is_active"] == 1]["symbol"].tolist(),
                                             df_securities[df_securities["is_active"] == 1]["security_id"].tolist())
db_func.write_market_data(df_eod, session)
db_func.write_security_fundamentals(df_fundamentals, session)