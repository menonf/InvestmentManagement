from datetime import datetime
from urllib import parse

import financedatabase as fd
import pandas as pd
import sqlalchemy as sql
from sqlalchemy.orm import Session

from data_engineering.database import db_functions as db_func


def get_all_equities():
    # Load all equities from FinanceDatabase
    equities = fd.Equities()
    all_equities = equities.select()
    equities_df = all_equities[["name", "isin", "cusip", "figi", "country", "currency", "sector", "industry_group", "industry", "exchange"]]
    
    equities_df["symbol"] = equities_df.index
    equities_df["sedol"] = None
    equities_df["loanxid"] = None
    equities_df["security_type"] = "Common Stock"
    equities_df["asset_class"] = "Equity"
    equities_df["is_active"] = 0
    equities_df["source_vendor"] = "FinanceDatabase"
    equities_df["upsert_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    equities_df["upsert_by"] = "mf"
    
    
    # Reorder columns
    column_order = [
        "symbol", "name", "isin", "sedol", "cusip", "figi", "loanxid", "country", "currency",
        "sector", "industry_group", "industry", "security_type", "asset_class", "exchange",
        "is_active", "source_vendor", "upsert_date", "upsert_by"
    ]
    equities_df = equities_df[column_order]
    
    equities_df = equities_df.where(pd.notna(equities_df), None)
    equities_df = equities_df.dropna(subset=["symbol"])
    
    return equities_df



if __name__ == "__main__":
    
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
        
    
    equities_data = get_all_equities()
    db_func.write_security_master(equities_data, session)
 
