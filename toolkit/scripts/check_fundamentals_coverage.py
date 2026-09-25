"""Coverage report for stored Refinitiv fundamentals (read-only).

Run from project root:
    .venv/Scripts/python.exe -m toolkit.scripts.check_fundamentals_coverage

Shows: overall density, per-metric cross-sectional + temporal coverage, and
the quarters-per-security depth distribution.
"""
from __future__ import annotations

import pandas as pd
import sqlalchemy as sa

from data_engineering.database import database
from data_engineering.fundamentals import RATIO_COLUMNS

engine, connection, _c, session = database.get_db_connection()
cols = ", ".join(f"'{c.replace(chr(39), chr(39)*2)}'" for c in RATIO_COLUMNS)

# raw stored cells
raw = pd.read_sql_query(
    sa.text(
        f"SELECT security_id, metric_type, effective_date FROM dbo.security_fundamentals "
        f"WHERE source_vendor='refinitiv' AND metric_type IN ({cols})"
    ),
    engine,
)
raw["effective_date"] = pd.to_datetime(raw["effective_date"])

n_sec = raw["security_id"].nunique()
n_dates = raw["effective_date"].nunique()
n_pairs = raw.groupby(["security_id", "effective_date"]).ngroups   # distinct (sec,period) slots present
max_pairs = n_sec * n_dates
print(f"Securities with ANY data : {n_sec}")
print(f"Distinct period-ends     : {n_dates}")
print(f"(sec,period) slots present: {n_pairs:,} of {max_pairs:,} possible  -> overall density {n_pairs/max_pairs:.1%}")
print()

# per-metric coverage
rows = []
for m in RATIO_COLUMNS:
    sub = raw[raw["metric_type"] == m]
    secs = sub["security_id"].nunique()
    dates = sub["effective_date"].nunique()
    cells = len(sub)
    rows.append({
        "metric_type": m,
        "cells": cells,
        "securities": secs,
        "xsec_%": round(100*secs/n_sec, 1),
        "periods": dates,
        "temp_%": round(100*dates/n_dates, 1),
    })
cov = pd.DataFrame(rows).sort_values("cells")
pd.set_option("display.width", 160)
print("Per-metric coverage (sorted by fewest cells):")
print(cov.to_string(index=False))
print()

# depth per security (how many quarters each security has, any metric)
depth = raw.groupby("security_id")["effective_date"].nunique()
print("Quarters-per-security (any metric):")
print(f"  min={depth.min()}  mean={depth.mean():.1f}  median={depth.median():.0f}  max={depth.max()}")
print(f"  securities with <=2 quarters: {(depth<=2).sum()}")
print()

# valuation multiples depth specifically
mult = ["P/E", "P/B", "EV/EBIT", "P/S"]
mdepth = raw[raw["metric_type"].isin(mult)].groupby("security_id")["effective_date"].nunique()
print(f"Quarters-per-security WITH a valuation multiple (P/E/P/B/EV/EBIT/P/S):")
print(f"  min={mdepth.min()}  mean={mdepth.mean():.1f}  median={mdepth.median():.0f}  max={mdepth.max()}")
print(f"  securities with 0 valuation-multiples: {(mdepth==0).sum()} of {n_sec}")
print()

# temporal trend: is recent data better? count cells per year
raw["yr"] = raw["effective_date"].dt.year
print("Cells per fiscal year (all metrics combined):")
print(raw.groupby("yr").size().to_string())
print()
print("Valuation-multiple cells per fiscal year:")
print(raw[raw["metric_type"].isin(mult)].groupby("yr").size().to_string())

session.close()
connection.close()
