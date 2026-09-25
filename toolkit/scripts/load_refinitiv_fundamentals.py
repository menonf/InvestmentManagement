"""Load Refinitiv fundamental ratios into dbo.security_fundamentals.

Populates the 18 ratios for a security universe (the constituents of a
portfolio, or every active security) so the database-backed
``StaticFundamentalsProvider`` and ``MLReturnFactor`` can score without a live
LSEG session. Securities missed by the bulk pull are retried one at a time.

Usage (from the project root, after ``pip install -e .``):
    python toolkit/scripts/load_refinitiv_fundamentals.py --universe SP500 --frequency Q --start 2020-01-01
    python toolkit/scripts/load_refinitiv_fundamentals.py --universe MAG8 --frequency FY --as-of 2025-12-31
    python toolkit/scripts/load_refinitiv_fundamentals.py --limit 50      # smoke test on 50 names

Requires LSEG Workspace running locally. Writes are idempotent (MERGE upsert).
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from data_engineering.database import database
from data_engineering.loaders import load_fundamentals_with_retry
from data_engineering.loaders.portfolios import portfolio_member_ids


def resolve_universe(session, engine, portfolio_short_name: str | None, limit: int | None) -> pd.DataFrame:
    """``[security_id, symbol]`` for a portfolio's members, or every active security."""
    master = database.read_security_master(session, engine)
    if portfolio_short_name:
        ids = portfolio_member_ids(engine, portfolio_short_name)
        master = master[master["security_id"].isin(ids)]
    else:
        master = master[master["is_active"] == 1]
    universe = master[["security_id", "name"]].rename(columns={"name": "symbol"}).drop_duplicates("security_id")
    return universe.head(limit) if limit else universe


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--universe", default=None, help="portfolio_short_name whose members to load (default: all active securities)")
    parser.add_argument("--as-of", default=pd.Timestamp.today().strftime("%Y-%m-%d"), help="snapshot / history end date")
    parser.add_argument("--frequency", default="FY", choices=["FY", "Q"], help="FY = latest fiscal-year snapshot; Q = quarterly history")
    parser.add_argument("--start", default="2020-01-01", help="history start for --frequency Q")
    parser.add_argument("--limit", type=int, default=None, help="max number of securities")
    parser.add_argument("--source-vendor", default="refinitiv", help="vendor label stored in the DB")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine, connection, _conn_str, session = database.get_db_connection()
    try:
        universe = resolve_universe(session, engine, args.universe, args.limit)
        if universe.empty:
            print("No securities found for the given universe.")
            return
        print(f"Loading Refinitiv {'quarterly history' if args.frequency == 'Q' else 'annual snapshot'} for {len(universe)} securities ...")
        summary = load_fundamentals_with_retry(universe, session, engine, args.start, args.as_of, frequency=args.frequency, source_vendor=args.source_vendor)
        print(f"Done. Wrote {summary['rows_written']} rows; covered {len(summary['covered'])} securities; missing {len(summary['missing'])}.")
    finally:
        session.close()
        connection.close()


if __name__ == "__main__":
    main()
