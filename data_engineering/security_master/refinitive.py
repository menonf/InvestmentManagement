"""
Refinitiv Security Master Ingestion
=====================================
Fetches the full equity universe from Refinitiv Data Library, enriches it with
GICS sector/industry data and additional identifiers, then upserts into the
two-table security master:

    dbo.security_master      - one row per canonical security
    dbo.security_vendor_xref - one row per vendor × security (Refinitiv-specific codes)

Matching cascade (highest → lowest priority):
    1. RIC via security_vendor_xref          ← catches ALL previously ingested securities
    2. ISIN → CUSIP → FIGI → SEDOL
    3. VALOR → WKN → Common Code → PermID
    4. Ticker+Exchange → Name+Country        ← fuzzy last-resort

Why RIC is first:
    RIC is always present in Refinitiv data.  Any security ingested previously
    already has its RIC stored in security_vendor_xref.vendor_ticker.  Checking
    the xref table first means a security with no ISIN/CUSIP is still matched
    on every subsequent run rather than being inserted as a duplicate.

Usage:
    python refinitiv_ingest.py
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import logging
import math
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import lseg.data as ld
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

VENDOR = "Refinitiv"
SCRIPT_NAME = "refinitiv_ingest.py"

PAGE_SIZE = 1000  # rows per search page — kept small so top+skip never exceeds 10k
GICS_BATCH_SIZE = 5000  # RICs per get_data call (API limit)
DB_BATCH_SIZE = 1000  # rows per DB commit
RETRY_SLEEP_SEC = 5  # pause between failed API calls
MAX_API_RETRIES = 3
MAX_UNIVERSE_PAGES = None  # set to an int (e.g. 3) for test runs; None = full run

# ---------------------------------------------------------------------------
# Partition list — one search shard per exchange.
# Each exchange must have fewer than 10,000 securities so that
# (top + skip) never exceeds the Elasticsearch hard ceiling.
# ---------------------------------------------------------------------------
EXCHANGE_PARTITIONS = [
    # "NSE",   # National Stock Exchange of India
    # "BSE",   # Bombay Stock Exchange
    # Add more: "MCX", "NYQ", "NSQ", "LSE", etc.
    "NSQ",
    "NYQ",
]

# Fields returned by ld.discovery.search
SEARCH_SELECT = (
    "TickerSymbol, "
    "IssuerCommonName, "
    "SEDOL, "
    "CUSIP, "
    "RIC, "
    "ISIN, "
    "RCSExchangeCountryLeaf, "
    "RCSCurrencyLeaf, "
    "RCSAssetCategoryLeaf, "
    "RCSAssetClass, "
    "ExchangeCode"
)

# Fields fetched via ld.get_data (keyed on RIC)
ENRICH_FIELDS = [
    "TR.ISIN",
    "TR.GICSSector",
    "TR.GICSIndustry",
    "TR.GICSSubIndustry",
    "TR.VALOR",  # Swiss Valorennummer
    "TR.WKN",  # German Wertpapierkennnummer
    "TR.CommonCode",  # Euroclear / Clearstream Common Code
    "TR.PermID",  # Refinitiv permanent identifier (open, non-recycled)
]

# Internal join-key column prefix (stripped before any DB write)
_INTERNAL_PREFIX = "_"

# Ordered cascade of (lookup_map_name, row_key) pairs used for matching.
# This list drives _cascade_lookup and must stay in priority order.
IDENTIFIER_CASCADE = [
    ("xref_ric", "_ric"),
    ("isin", "_isin"),
    ("cusip", "_cusip"),
    ("figi", "figi"),
    ("sedol", "_sedol"),
    ("valor", "_valor"),
    ("wkn", "_wkn"),
    ("common_code", "_common_code"),
    ("perm_id", "_perm_id"),
    ("ticker_exch", "_ticker_exch"),
    ("name_country", "_name_country"),
]

GICS_FIELDS = [
    "TR.GICSSector",
    "TR.GICSIndustry",
    "TR.GICSSubIndustry",
]

# ===========================================================================
# Helpers
# ===========================================================================


def clean(val) -> Optional[str]:
    """Return ``None`` for blank / null / NaN / NA values; stripped string otherwise."""
    if val is None:
        return None
    text = str(val).strip()
    return None if text in ("", "nan", "None", "<NA>") else text


def _safe_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Return the column *name* from *df* with ``clean`` applied, or an all-None series.

    FIX: cast to dtype=object *before* mapping. ``clean`` already returns
    ``None`` for NaN/blank values, but if the source column is float-typed
    (as CUSIP/SEDOL/etc. can be when a column is all-NaN or mixed), pandas
    is free to coerce those ``None`` results straight back into ``NaN``
    when it re-infers the column's dtype. Locking the dtype to ``object``
    stops that silent re-coercion.
    """
    if name in df.columns:
        return df[name].astype(object).map(clean)
    return pd.Series([None] * len(df), dtype=object)


def _sanitize_for_db(d: dict) -> dict:
    """Final safety net applied to every row dict right before it is handed
    to pyodbc for insertion.

    Root cause of the 22003 'Numeric value out of range' error: pyodbc
    infers each bind parameter's ODBC type from the *Python* type of the
    value, not from the target SQL column. A stray ``float('nan')`` has no
    valid ODBC numeric representation, so the driver rejects the entire
    batch — even for a column like ``cusip`` that is VARCHAR, not numeric.

    ``_safe_column`` prevents this at the column level, but per-row
    reconstruction via ``DataFrame.iterrows()`` -> ``Series.to_dict()``
    can still resurface a NaN (a row that happens to be all "None"/numeric
    values can get its per-row Series re-inferred as float64 by pandas).
    This sweep guarantees nothing NaN-like ever reaches pyodbc, regardless
    of how it got reintroduced upstream.
    """
    return {k: (None if pd.isna(v) else v) for k, v in d.items()}


def _api_call_with_retry(func, *, description: str = "API call"):
    """
    Execute *func* with up to ``MAX_API_RETRIES`` attempts.

    Returns the result of *func* on success or re-raises the last exception.
    """
    for attempt in range(1, MAX_API_RETRIES + 1):
        try:
            return func()
        except Exception as exc:
            log.warning(
                "%s - attempt %d/%d failed: %s",
                description,
                attempt,
                MAX_API_RETRIES,
                exc,
            )
            if attempt == MAX_API_RETRIES:
                raise
            time.sleep(RETRY_SLEEP_SEC)


# ===========================================================================
# Step 1 — Fetch universe from Refinitiv (partitioned by exchange)
# ===========================================================================


def _subdivide_and_fetch(exchange: str, prefix: str) -> pd.DataFrame:
    """
    Sub-divide the search space by appending characters to the current prefix.
    If a partition exceeds 10,000 securities, this function breaks it down
    alphabetically (e.g. 'A' -> 'AA', 'AB', 'AC' ... 'A9').
    """
    # Includes standard alphanumeric characters plus period, covering nearly all RIC structures.
    CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789."
    sub_frames = []

    # 1. Fetch the exact match for the current prefix to avoid missing it.
    # (Since `startswith(RIC, 'A')` covers 'AA' but we need to ensure 'A' itself isn't dropped
    # when we switch to checking 'AA', 'AB', etc.)
    if prefix:
        exact_filter = f"AssetType eq 'equity' and ExchangeCode eq '{exchange}' and RIC eq '{prefix}'"
        exact_match = _api_call_with_retry(
            lambda ex=exchange, pf=prefix: ld.discovery.search(
                view=ld.discovery.Views.EQUITY_QUOTES, filter=exact_filter, select=SEARCH_SELECT, top=1
            ),
            description=f"Exact match for '{prefix}'",
        )
        if exact_match is not None and not exact_match.empty:
            sub_frames.append(exact_match)

    # 2. Iterate through all possible next characters to fetch sub-partitions
    for char in CHARS:
        new_prefix = prefix + char
        df_sub = _fetch_partition(exchange, new_prefix)
        if not df_sub.empty:
            sub_frames.append(df_sub)

    if not sub_frames:
        return pd.DataFrame()

    return pd.concat(sub_frames, ignore_index=True)


def _fetch_partition(exchange: str, prefix: str = "") -> pd.DataFrame:
    """
    Fetch equity quotes for a single *exchange* code, optionally filtered by RIC *prefix*.

    Refinitiv's search endpoint enforces a hard ceiling of ``top + skip <= 10_000``.
    If a partition hits this limit, the current partial fetch is discarded and dynamically
    sub-divided into alphabetical sub-groups to safely bypass the ceiling.
    """
    pages: List[pd.DataFrame] = []
    skip = 0
    page_count = 0
    API_CEILING = 10_000

    while True:
        current_skip = skip  # freeze value before entering retry closure

        # Guard: abort before the API rejects the request
        if current_skip + PAGE_SIZE > API_CEILING:
            log.warning(
                "Exchange '%s' (prefix: '%s') has ≥%d securities — cannot fetch beyond the "
                "Elasticsearch 10k ceiling. Dynamically sub-dividing partition...",
                exchange,
                prefix if prefix else "<ALL>",
                API_CEILING,
            )
            # Discard current pages (they are incomplete) and fetch via sub-divisions
            return _subdivide_and_fetch(exchange, prefix)

        # Build the OData filter
        filter_expr = f"AssetType eq 'equity' and ExchangeCode eq '{exchange}'"
        if prefix:
            filter_expr += f" and startswith(RIC, '{prefix}')"

        page = _api_call_with_retry(
            lambda s=current_skip, f_expr=filter_expr: ld.discovery.search(
                view=ld.discovery.Views.EQUITY_QUOTES,
                filter=f_expr,
                select=SEARCH_SELECT,
                top=PAGE_SIZE,
                skip=s,
            ),
            description=f"Search ex={exchange} prefix='{prefix}' skip={current_skip}",
        )

        if page is None or page.empty:
            log.info("Exchange '%s' prefix '%s': empty page at skip=%d — partition complete.", exchange, prefix, current_skip)
            break

        page_count += 1
        pages.append(page)
        fetched = len(page)
        skip += fetched
        log.info(
            "Exchange '%s' | prefix '%s' | page %d | skip=%d | fetched=%d | running total=%d",
            exchange,
            prefix,
            page_count,
            current_skip,
            fetched,
            skip,
        )

        if MAX_UNIVERSE_PAGES and page_count >= MAX_UNIVERSE_PAGES:
            log.info("Reached MAX_UNIVERSE_PAGES=%d for exchange '%s' prefix '%s'; stopping.", MAX_UNIVERSE_PAGES, exchange, prefix)
            break

        if fetched < PAGE_SIZE:
            break  # last partial page — we are done

    if not pages:
        return pd.DataFrame()

    return pd.concat(pages, ignore_index=True)


def fetch_refinitiv_universe() -> pd.DataFrame:
    """
    Iterate over ``EXCHANGE_PARTITIONS`` and concatenate all results into a
    single de-duplicated DataFrame.

    Each partition is fetched independently so that no individual shard
    approaches the Refinitiv / Elasticsearch 10,000-row (top + skip) ceiling.
    """
    all_frames: List[pd.DataFrame] = []

    log.info(
        "Starting Refinitiv universe fetch across %d exchange partition(s): %s",
        len(EXCHANGE_PARTITIONS),
        EXCHANGE_PARTITIONS,
    )

    for exchange in EXCHANGE_PARTITIONS:
        df_partition = _fetch_partition(exchange)
        if not df_partition.empty:
            all_frames.append(df_partition)
            log.info("Partition '%s' complete: %d rows.", exchange, len(df_partition))

    if not all_frames:
        raise RuntimeError("Refinitiv search returned no data across all partitions.")

    df = pd.concat(all_frames, ignore_index=True).drop_duplicates(subset=["RIC"])
    log.info(
        "Universe fetch complete: %d unique RICs across %d partition(s).",
        len(df),
        len(EXCHANGE_PARTITIONS),
    )
    return df


# =============================================================================
# GICS Enrichment
# =============================================================================


def fetch_gics(rics: List[str]) -> pd.DataFrame:
    batches = math.ceil(len(rics) / GICS_BATCH_SIZE)
    out = []

    log.info("Fetching GICS in %s batches", batches)

    for i in range(batches):
        batch = rics[i * GICS_BATCH_SIZE : (i + 1) * GICS_BATCH_SIZE]

        try:
            # FIX: Wrap the target function in a lambda and pass the description kwarg
            df = _api_call_with_retry(
                lambda b=batch: ld.get_data(universe=b, fields=GICS_FIELDS), description=f"GICS batch {i+1}/{batches}"
            )
            df = df.rename(columns={"Instrument": "RIC"})
            out.append(df)

        except Exception as e:
            log.warning("Skipping GICS batch %s due to failure: %s", i + 1, e)

    if not out:
        return pd.DataFrame(columns=["RIC"] + GICS_FIELDS)

    return pd.concat(out, ignore_index=True)


# =============================================================================
# Build Master / Xref
# =============================================================================


def build_frames(universe: pd.DataFrame, gics: pd.DataFrame):

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = universe.merge(gics, on="RIC", how="left")

    # Safely identify which GICS column names Refinitiv actually returned
    sector_col = "TR.GICSSector" if "TR.GICSSector" in df.columns else "GICS Sector Name"
    ind_col = "TR.GICSIndustry" if "TR.GICSIndustry" in df.columns else "GICS Industry Name"
    sub_col = "TR.GICSSubIndustry" if "TR.GICSSubIndustry" in df.columns else "GICS Sub-Industry Name"

    master = pd.DataFrame(
        {
            "name": _safe_column(df, "IssuerCommonName"),
            "isin": _safe_column(df, "ISIN"),
            "sedol": _safe_column(df, "SEDOL"),
            "cusip": _safe_column(df, "CUSIP"),
            "figi": None,
            "country": _safe_column(df, "RCSExchangeCountryLeaf"),
            "currency": _safe_column(df, "RCSCurrencyLeaf"),
            "sector": _safe_column(df, sector_col),
            "industry_group": _safe_column(df, ind_col),
            "industry": _safe_column(df, sub_col),
            "security_type": _safe_column(df, "RCSAssetCategoryLeaf"),
            "asset_class": _safe_column(df, "RCSAssetClass"),
            "is_active": 1,
            "upsert_date": now,
            "upsert_by": "refinitiv_loader",
            "_ric": df["RIC"],  # RIC is guaranteed because it's our merge key
        }
    )

    xref = pd.DataFrame(
        {
            "vendor": VENDOR,
            "vendor_ticker": df["RIC"].map(clean),
            "vendor_exchange_code": _safe_column(df, "ExchangeCode"),
            "vendor_currency": _safe_column(df, "RCSCurrencyLeaf"),
            "is_primary": 0,
            "is_active": 1,
            "upsert_date": now,
            "upsert_by": "refinitiv_loader",
            "_ric": df["RIC"],
        }
    )

    return master, xref


# =============================================================================
# Security Resolution
# =============================================================================


def resolve_security_ids(master: pd.DataFrame, session: Session, engine: sql.Engine):
    existing = database.read_security_master(session, engine)

    # 1. Build lookup maps for all identifiers
    isin_map = existing.set_index("isin")["security_id"].dropna().to_dict() if "isin" in existing.columns else {}
    cusip_map = existing.set_index("cusip")["security_id"].dropna().to_dict() if "cusip" in existing.columns else {}
    figi_map = existing.set_index("figi")["security_id"].dropna().to_dict() if "figi" in existing.columns else {}

    # RIC fallback via xref
    xref_existing = database.read_security_vendor_xref(session, engine, VENDOR)
    ric_map = xref_existing.set_index("vendor_ticker")["security_id"].to_dict() if not xref_existing.empty else {}

    # 2. Vectorized cascading resolution (priority order matching your docstring)
    # Priority 1: RIC
    master["security_id"] = master["_ric"].map(ric_map)

    # Priority 2: ISIN -> CUSIP -> FIGI
    for col, mapping in [("isin", isin_map), ("cusip", cusip_map), ("figi", figi_map)]:
        missing_mask = master["security_id"].isna()
        if missing_mask.any() and col in master.columns:
            master.loc[missing_mask, "security_id"] = master.loc[missing_mask, col].map(mapping)

    # 3. Insert truly new records and retrieve their auto-incremented IDs
    new_mask = master["security_id"].isna()

    if new_mask.any():
        insert_df = master[new_mask].copy()

        cols_to_drop = [c for c in insert_df.columns if c.startswith("_")] + ["security_id"]
        insert_df = insert_df.drop(columns=cols_to_drop, errors="ignore")

        # Cast to object and replace Pandas NaT/NaN with pure Python None to prevent PyODBC crashes
        insert_df = insert_df.astype(object).where(pd.notnull(insert_df), None)
        new_dicts = insert_df.to_dict(orient="records")

        log.info("Inserting %s new securities", len(new_dicts))

        new_ids = []
        s = Session(engine)

        try:
            for i in range(0, len(new_dicts), DB_BATCH_SIZE):
                batch = new_dicts[i : i + DB_BATCH_SIZE]

                # Create ORM objects
                objects = [database.SecurityMaster(**row) for row in batch]
                s.add_all(objects)

                # Flush generates auto-increment PKs immediately without committing
                s.flush()

                new_ids.extend([obj.security_id for obj in objects])

            s.commit()

        except Exception as e:
            s.rollback()
            log.error(f"Failed to insert SecurityMaster chunk: {e}")
            raise
        finally:
            s.close()

        # 4. Assign the perfectly ordered generated IDs back to the dataframe
        master.loc[new_mask, "security_id"] = new_ids

    master = master.dropna(subset=["security_id"])
    master["security_id"] = master["security_id"].astype(int)

    return master


# =============================================================================
# Write Vendor Xref
# =============================================================================


def write_xref(master: pd.DataFrame, xref: pd.DataFrame, engine: sql.Engine):
    id_map = master.set_index("_ric")["security_id"].to_dict()

    xref["security_id"] = xref["_ric"].map(id_map)
    xref = xref.dropna(subset=["security_id"])

    # 1. Force the float64 column back to native integers
    xref["security_id"] = xref["security_id"].astype(int)

    xref = xref.drop(columns=[c for c in xref.columns if c.startswith("_")])

    # 2. Prevent PyODBC NaN crashes on the xref table
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
        finally:
            s.close()


# ===========================================================================
# Main
# ===========================================================================


def main():

    log.info("START Refinitiv ingestion")

    ld.open_session()

    try:
        universe = fetch_refinitiv_universe()
        rics = universe["RIC"].dropna().unique().tolist()
        gics = fetch_gics(rics)
        master, xref = build_frames(universe, gics)

    finally:
        ld.close_session()

    engine, conn, _, session = database.get_db_connection()

    try:
        master = resolve_security_ids(master, session, engine)
        write_xref(master, xref, engine)
    finally:
        conn.close()

    log.info("DONE")


if __name__ == "__main__":
    main()
