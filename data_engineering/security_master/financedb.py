"""Ingest the equity universe from FinanceDatabase into the security master.

Fetches the full equity universe from the open-source FinanceDatabase package,
enriches it, and upserts into the two-table security master:

    dbo.security_master      - one row per canonical security
    dbo.security_vendor_xref - one row per vendor × security

Matching cascade (highest → lowest priority):
    1. Symbol via security_vendor_xref       ← catches ALL previously ingested FD securities
    2. ISIN                                  ← catches matches from other vendors (e.g., Refinitiv)

Usage:
    pip install financedatabase -U
    python financedatabase_ingest.py
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import logging
from datetime import datetime
from typing import Any, Optional

import financedatabase as fd
import pandas as pd
import sqlalchemy as sql
from sqlalchemy.orm import Session

from data_engineering.database import database

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / Configuration
# ---------------------------------------------------------------------------

VENDOR = "FinanceDatabase"
SCRIPT_NAME = "financedatabase_ingest.py"
DB_BATCH_SIZE = 1000  # rows per DB commit

# ===========================================================================
# Helpers
# ===========================================================================


def clean(val: Any) -> Optional[str]:
    """Return None for blank / null / NaN / NA values; stripped string otherwise."""
    if val is None:
        return None
    text = str(val).strip()
    return None if text in ("", "nan", "None", "<NA>") else text


def _safe_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Return the column name from df with clean applied, or an all-None series."""
    if name in df.columns:
        return df[name].map(clean)
    return pd.Series([None] * len(df), dtype=object)


# ===========================================================================
# Step 1 — Fetch universe from FinanceDatabase
# ===========================================================================


def fetch_fd_universe() -> pd.DataFrame:
    """Fetch the full Equities dataset from FinanceDatabase.

    The resulting dataframe uses the ticker symbol as the index.
    """
    log.info("Initializing FinanceDatabase Equities module...")

    # Initialize and select the entire equity universe
    equities = fd.Equities()
    df = equities.select()

    # Reset index to pull the ticker symbol into a standard column
    df = df.reset_index()

    # Depending on the fd version, the index column might be named 'symbol' or 'index'
    if "symbol" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "symbol"})

    df = df.dropna(subset=["symbol"])
    df = df.drop_duplicates(subset=["symbol"])

    log.info("Universe fetch complete: %d unique symbols found.", len(df))
    return df


# =============================================================================
# Build Master / Xref
# =============================================================================


def build_frames(df: pd.DataFrame) -> dict[str, Any]:
    """Split the FinanceDatabase universe into master and xref frames.

    Args:
        df: Raw FinanceDatabase equities DataFrame.

    Returns:
        Tuple of (security_master rows, vendor_xref rows) as lists of dicts.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    master = pd.DataFrame(
        {
            "name": _safe_column(df, "name"),
            "isin": _safe_column(df, "isin"),  # Available in recent fd versions
            "sedol": None,
            "cusip": None,
            "figi": None,
            "country": _safe_column(df, "country"),
            "currency": _safe_column(df, "currency"),
            "sector": _safe_column(df, "sector"),
            "industry_group": _safe_column(df, "industry_group"),
            "industry": _safe_column(df, "industry"),
            "security_type": "Equity",
            "asset_class": "Equities",
            "is_active": 1,
            "upsert_date": now,
            "upsert_by": "financedatabase_loader",
            "_vendor_ticker": df["symbol"],
        }
    )

    xref = pd.DataFrame(
        {
            "vendor": VENDOR,
            "vendor_ticker": df["symbol"].map(clean),
            "vendor_exchange_code": _safe_column(df, "exchange"),
            "vendor_currency": _safe_column(df, "currency"),
            "is_primary": 0,
            "is_active": 1,
            "upsert_date": now,
            "upsert_by": "financedatabase_loader",
            "_vendor_ticker": df["symbol"],
        }
    )

    return master, xref


# =============================================================================
# Security Resolution
# =============================================================================
def resolve_security_ids(master: pd.DataFrame, session: Session, engine: sql.Engine) -> pd.DataFrame:
    """Resolve internal security_ids for the ingested master rows.

    Args:
        master: Incoming security_master frame.
        session: Active ORM session.
        engine: SQLAlchemy engine.

    Returns:
        The master frame enriched with resolved ``security_id`` values.
    """
    existing = database.read_security_master(session, engine)

    # 1. Build lookup maps for cross-vendor resolution
    isin_map = {}
    if "isin" in existing.columns:
        isin_map = existing.set_index("isin")["security_id"].dropna().to_dict()

    xref_existing = database.read_security_vendor_xref(session, engine, VENDOR)
    ticker_map = xref_existing.set_index("vendor_ticker")["security_id"].to_dict()

    # 2. Vectorized map for existing records
    master["security_id"] = master["_vendor_ticker"].map(ticker_map)

    missing_mask = master["security_id"].isna()
    if missing_mask.any() and "isin" in master.columns:
        master.loc[missing_mask, "security_id"] = master.loc[missing_mask, "isin"].map(isin_map)

    # 3. Insert truly new records and retrieve their auto-incremented IDs
    new_mask = master["security_id"].isna()

    if new_mask.any():
        insert_df = master[new_mask].copy()

        cols_to_drop = [c for c in insert_df.columns if c.startswith("_")] + ["security_id"]
        insert_df = insert_df.drop(columns=cols_to_drop, errors="ignore")

        # Cast to object to prevent PyODBC Numpy datatype crashes
        insert_df = insert_df.astype(object).where(pd.notnull(insert_df), None)
        new_dicts = insert_df.to_dict(orient="records")

        log.info("Inserting %s new securities", len(new_dicts))

        new_ids = []
        s = Session(engine)

        try:
            # Chunking to prevent memory spikes
            for i in range(0, len(new_dicts), DB_BATCH_SIZE):
                batch = new_dicts[i : i + DB_BATCH_SIZE]

                # Create ORM objects so SQLAlchemy tracks them
                objects = [database.SecurityMaster(**row) for row in batch]
                s.add_all(objects)

                # Flush pushes to the DB and populates auto-increment PKs instantly
                # without committing the transaction yet
                s.flush()

                # Extract the exact IDs in the exact order they were inserted
                new_ids.extend([obj.security_id for obj in objects])

            # Commit the entire batch of inserts
            s.commit()

        except Exception as e:
            s.rollback()
            log.error(f"Failed to insert SecurityMaster chunk: {e}")
            raise

        # 4. Assign the generated IDs perfectly back to the dataframe
        master.loc[new_mask, "security_id"] = new_ids

    master = master.dropna(subset=["security_id"])
    master["security_id"] = master["security_id"].astype(int)

    return master


# =============================================================================
# Write Vendor Xref
# =============================================================================


def write_xref(master: pd.DataFrame, xref: pd.DataFrame, engine: sql.Engine) -> None:
    """Upsert the vendor xref rows for the ingested universe.

    Args:
        master: Security master frame (used to map vendor tickers to ids).
        xref: Vendor xref rows to write.
        engine: SQLAlchemy engine.
    """
    id_map = master.set_index("_vendor_ticker")["security_id"].to_dict()

    xref["security_id"] = xref["_vendor_ticker"].map(id_map)
    xref = xref.dropna(subset=["security_id"])

    # 1. Force the float64 column back to native integers
    xref["security_id"] = xref["security_id"].astype(int)

    xref = xref.drop(columns=[c for c in xref.columns if c.startswith("_")])

    # 2. CRITICAL FIX: Cast to object and strip Pandas NaNs/NaTs into pure Python None
    # so PyODBC doesn't choke on numpy datatypes during the bulk insert.
    xref = xref.astype(object).where(pd.notnull(xref), None)

    log.info("Writing %s xref rows", len(xref))

    for i in range(0, len(xref), DB_BATCH_SIZE):
        batch = xref.iloc[i : i + DB_BATCH_SIZE]
        s = Session(engine)

        try:
            database.write_security_vendor_xref(batch, s, vendor=VENDOR)
            s.commit()
        except Exception as e:
            s.rollback()
            log.error(f"Failed to write xref batch: {e}")
            raise


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    """Run the FinanceDatabase ingestion pipeline end to end."""
    log.info("START FinanceDatabase ingestion")

    universe = fetch_fd_universe()
    master, xref = build_frames(universe)

    engine, conn, _, session = database.get_db_connection()

    try:
        master = resolve_security_ids(master, session, engine)
        write_xref(master, xref, engine)
    finally:
        conn.close()

    log.info("DONE")


if __name__ == "__main__":
    main()
