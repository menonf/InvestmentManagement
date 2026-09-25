"""Re-pull EOD prices for a few securities in a FRESH process and upsert them.

The in-kernel Refinitiv pull occasionally returns corrupted values for
specific instruments (a share-class scale error, an index level offset) after
a long notebook session. Running the pull from a fresh interpreter avoids the
accumulated session state.

Usage:
    python toolkit/scripts/repair_eod_prices.py 2025-01-01 2025-12-31 [--targets 83=BRKb 638=.SPX 637=SPY.P]
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from data_engineering.database import database
from data_engineering.loaders import load_eod_prices

DEFAULT_TARGETS = {83: "BRKb", 638: ".SPX", 637: "SPY.P"}  # security_id -> RIC (matches the demo database)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("start")
    parser.add_argument("end")
    parser.add_argument("--targets", nargs="*", default=None, help="security_id=RIC pairs (default: the demo's fragile names)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    targets = dict(DEFAULT_TARGETS)
    if args.targets:
        targets = {int(k): v for k, v in (t.split("=", 1) for t in args.targets)}
    universe = pd.DataFrame([{"security_id": sec, "symbol": ric, "ric": ric} for sec, ric in targets.items()])

    engine, connection, _conn_str, session = database.get_db_connection()
    try:
        summary = load_eod_prices(universe, session, args.start, args.end, batch_size=1, max_retries=2, implausible_move_threshold=None)
        print(f"Repaired {summary.rows_written} rows for {summary.securities_loaded}/{summary.securities_requested} securities.")
    finally:
        session.close()
        connection.close()


if __name__ == "__main__":
    main()
