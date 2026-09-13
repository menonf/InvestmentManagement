"""Repair EOD prices corrupted by the in-kernel Refinitiv pull.

Repairs securities whose in-kernel Refinitiv pull corrupts values (BRK.B scale
error, GSPC/SPY level offset). Run as a FRESH subprocess so it is not affected
by the notebook kernel's accumulated session/state. Uses the live Refinitiv feed,
which returns correct values outside the notebook kernel.

Usage: python repair_eod_prices.py <start_date> <end_date>
"""

import sys

sys.path.insert(0, r"C:\SourceCode\InvestmentManagement")

import lseg.data as ld  # noqa: E402  (after sys.path.insert so the project root resolves)
import pandas as pd  # noqa: E402

from data_engineering.database import database  # noqa: E402
from data_engineering.eod_data import refinitiv as rf  # noqa: E402

s = sys.argv[1] if len(sys.argv) > 1 else "2026-08-01"
e = sys.argv[2] if len(sys.argv) > 2 else "2026-08-31"

# security_id -> Refinitiv RIC used by the notebook's xref
TARGETS = {
    69: "BRKb",  # Berkshire B (in-kernel pull returns ~73x real price)
    507: ".SPX",  # S&P 500 index level (in-kernel pull returns wrong start level)
    506: "SPY.P",  # SPY ETF (in-kernel pull returns wrong start level)
}

ld.open_session()
engine, connection, conn_str, session = database.get_db_connection()

frames = []
for sec, ric in TARGETS.items():
    uni = pd.DataFrame([{"symbol": ric, "security_id": sec, "ric": ric}])
    lv = rf.get_stock_price(uni, s, e).copy()
    if lv.empty:
        print(f"  {ric}: EMPTY live pull -- skipped")
        continue
    lv["as_of_date"] = pd.to_datetime(lv["as_of_date"])
    lv["dataload_date"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    lv["interval"] = "1d"
    lv["dividends"] = 0.0
    lv["stock_splits"] = 0.0
    lv["source_vendor"] = "Refinitiv"
    lv["price_currency"] = "USD"
    lv["security_id"] = sec
    frames.append(lv)
    print(
        f"  {ric}: pulled {len(lv)} rows, {lv['adj_close'].iloc[0]:.2f}..{lv['adj_close'].iloc[-1]:.2f} "
        f"(ret {(lv['adj_close'].iloc[-1]/lv['adj_close'].iloc[0]-1)*100:.2f}%)"
    )

if frames:
    allrows = pd.concat(frames, ignore_index=True)
    database.write_market_data(allrows, session)
    print(f"Repaired {len(allrows)} rows for {len(frames)} securities.")
else:
    print("Nothing to repair.")
session.close()
connection.close()
