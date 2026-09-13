"""Load Refinitiv fundamental ratios into dbo.security_fundamentals.

Populates the 18 notebook ratios for a security universe (e.g. an index
constituents table) so the DB-backed ``StaticFundamentalsProvider`` and
``MLReturnFactor`` can score them without a live LSEG session.

Usage:
    python toolkit/scripts/load_refinitiv_fundamentals.py --universe SP500 --as-of 2025-12-31
    python toolkit/scripts/load_refinitiv_fundamentals.py --limit 50   # first N securities

Requires the local LSEG/Refinitiv Workspace session to be open (the provider
opens it lazily). Writes are idempotent via database.write_security_fundamentals.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

sys.path.insert(0, r"C:\SourceCode\InvestmentManagement")

from analytics.factors.fundamentals import (
    RATIO_COLUMNS,
    RefinitivFundamentalsProvider,
    collect_and_store_fundamentals,
)
from data_engineering.database import database


def main(args=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", default=None, help="portfolio_short_name to load constituents for")
    ap.add_argument("--as-of", default=pd.Timestamp.today().strftime("%Y-%m-%d"),
                    help="snapshot date for the fundamentals (effective_date)")
    ap.add_argument("--limit", type=int, default=None, help="max number of securities to load")
    ap.add_argument("--source-vendor", default="refinitiv", help="vendor label stored in DB")
    ap.add_argument("--frequency", default="FY", choices=["FY", "Q"],
                    help="FY = latest fiscal-year snapshot; Q = quarterly history (2020->today)")
    ap.add_argument("--start", default=None,
                    help="history start date for --frequency Q (default: 2020-01-01)")
    if isinstance(args, argparse.Namespace):
        a = args
    else:
        # No explicit args (e.g. run as `python script.py --universe MAG8`):
        # parse_args() with no argument reads sys.argv; passing [] would ignore it.
        a = ap.parse_args(args if args is not None else None)

    engine, connection, conn_str, session = database.get_db_connection()
    try:
        # Resolve the universe of securities.
        if a.universe:
            sec = pd.read_sql(
                f"""\
                SELECT DISTINCT sm.security_id, sm.name AS symbol
                FROM dbo.security_master sm
                JOIN dbo.portfolio_holdings ph ON ph.security_id = sm.security_id
                JOIN dbo.portfolio p ON p.port_id = ph.port_id
                WHERE p.portfolio_short_name = '{a.universe}'
                """,
                engine,
            )
        else:
            sec = database.read_security_master(session, engine)
            # read_security_master uses 'name' (company name), not 'symbol'.
            if "symbol" not in sec.columns:
                sec = sec.rename(columns={"name": "symbol"})
            sec = sec[sec["is_active"] == 1][["security_id", "symbol"]]

        if sec.empty:
            print("No securities found for the given universe.")
            return

        if a.limit:
            sec = sec.head(a.limit)

        provider_kwargs = {}
        if a.frequency == "Q":
            provider_kwargs["frequency"] = "Q"
            provider_kwargs["start_date"] = a.start or "2020-01-01"

        freq_label = "quarterly history" if a.frequency == "Q" else "annual snapshot"
        print(f"Loading Refinitiv {freq_label} for {len(sec)} securities "
              f"as_of {a.as_of} ...")
        n = collect_and_store_fundamentals(
            provider_name="refinitiv",
            symbols=sec[["security_id", "symbol"]],
            as_of_date=a.as_of,
            orm_session=session,
            orm_engine=engine,
            source_vendor=a.source_vendor,
            provider_kwargs=provider_kwargs or None,
        )
        print(f"Done. Wrote {n} (security, metric) rows to dbo.security_fundamentals.")
    finally:
        session.close()
        connection.close()


if __name__ == "__main__":
    main()
