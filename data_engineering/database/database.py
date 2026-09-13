"""SQLAlchemy ORM module to connect to database objects."""

import re
import time
from datetime import date, datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
from urllib import parse

import keyring
import pandas as pd
import sqlalchemy as sql
from pandas import DataFrame
from sqlalchemy import Engine, Date, Float, Integer, String, delete, update
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

# ---------------------------------------------------------------------------
# ORM Base & Models
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """SQLAlchemy base class."""

    pass


class SecurityMaster(Base):
    """
    Maps to dbo.security_master.

    One row per canonical real-world security, identified by universal
    identifiers (ISIN, CUSIP, FIGI).  Vendor-specific codes (RICs, Bloomberg
    tickers, etc.) live in SecurityVendorXref.
    """

    __tablename__ = "security_master"
    __table_args__ = {"schema": "dbo"}

    security_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[Optional[str]] = mapped_column(nullable=True)
    isin: Mapped[Optional[str]] = mapped_column(nullable=True)
    sedol: Mapped[Optional[str]] = mapped_column(nullable=True)
    cusip: Mapped[Optional[str]] = mapped_column(nullable=True)
    figi: Mapped[Optional[str]] = mapped_column(nullable=True)
    country: Mapped[Optional[str]] = mapped_column(nullable=True)
    currency: Mapped[Optional[str]] = mapped_column(nullable=True)
    sector: Mapped[Optional[str]] = mapped_column(nullable=True)
    industry_group: Mapped[Optional[str]] = mapped_column(nullable=True)
    industry: Mapped[Optional[str]] = mapped_column(nullable=True)
    security_type: Mapped[str] = mapped_column()
    asset_class: Mapped[str] = mapped_column()
    region: Mapped[Optional[str]] = mapped_column(nullable=True)
    exchange_mic: Mapped[Optional[str]] = mapped_column(nullable=True)
    listing_country: Mapped[Optional[str]] = mapped_column(nullable=True)
    is_active: Mapped[bool] = mapped_column()
    upsert_date: Mapped[datetime] = mapped_column()
    upsert_by: Mapped[Optional[str]] = mapped_column(nullable=True)


class SecurityVendorXref(Base):
    """
    Maps to dbo.security_vendor_xref.

    One row per vendor × security.  Stores each vendor's native instrument
    code (RIC, Bloomberg ticker, Aladdin ID, etc.) alongside the canonical
    security_id from SecurityMaster, so backtests can resolve the right code
    for whichever vendor is in use.

    Columns
    -------
    vendor               : Source system name, e.g. 'Refinitiv', 'Bloomberg', 'Aladdin'.
    vendor_ticker        : The vendor's own instrument identifier, e.g. 'MSFT.O', 'MSFT US Equity'.
    vendor_exchange_code : Exchange code in the vendor's notation.
    vendor_currency      : Currency code in the vendor's notation (if it differs from ISO standard).
    loanxid              : Aladdin loan identifier — NULL for non-Aladdin vendors.
    is_primary           : 1 if this vendor is the golden source for this security, else 0.
    is_active            : 1 if this mapping is current, else 0.
    """

    __tablename__ = "security_vendor_xref"
    __table_args__ = {"schema": "dbo"}

    xref_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    security_id: Mapped[int] = mapped_column()
    vendor: Mapped[str] = mapped_column()
    vendor_ticker: Mapped[Optional[str]] = mapped_column(nullable=True)
    vendor_exchange_code: Mapped[Optional[str]] = mapped_column(nullable=True)
    vendor_currency: Mapped[Optional[str]] = mapped_column(nullable=True)
    loanxid: Mapped[Optional[str]] = mapped_column(nullable=True)
    is_primary: Mapped[bool] = mapped_column()
    is_active: Mapped[bool] = mapped_column()
    upsert_date: Mapped[datetime] = mapped_column()
    upsert_by: Mapped[Optional[str]] = mapped_column(nullable=True)


class SecurityFundamentals(Base):
    """Maps to dbo.security_fundamentals."""

    __tablename__ = "security_fundamentals"
    __table_args__ = {"schema": "dbo"}
    __mapper_args__ = {"primary_key": ["security_id", "metric_type", "effective_date", "source_vendor"]}

    security_id: Mapped[int] = mapped_column()
    metric_type: Mapped[str] = mapped_column()
    metric_value: Mapped[float] = mapped_column()
    source_vendor: Mapped[str] = mapped_column()
    effective_date: Mapped[date] = mapped_column()
    end_date: Mapped[Optional[date]] = mapped_column(nullable=True)


class MarketData(Base):
    """Maps to dbo.market_data."""

    __tablename__ = "market_data"
    __table_args__ = {"schema": "dbo"}

    md_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    as_of_date: Mapped[date] = mapped_column(Date)
    security_id: Mapped[int] = mapped_column()
    open: Mapped[float] = mapped_column()
    high: Mapped[float] = mapped_column()
    low: Mapped[float] = mapped_column()
    close: Mapped[float] = mapped_column()
    adj_close: Mapped[float] = mapped_column()
    volume: Mapped[int] = mapped_column()
    dividends: Mapped[float] = mapped_column()
    stock_splits: Mapped[float] = mapped_column()
    interval: Mapped[str] = mapped_column()
    dataload_date: Mapped[str] = mapped_column()
    price_currency: Mapped[str] = mapped_column(String(8), default="USD")
    source_vendor: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


class Portfolio(Base):
    """Maps to dbo.portfolio."""

    __tablename__ = "portfolio"
    __table_args__ = {"schema": "dbo"}

    port_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    portfolio_short_name: Mapped[str] = mapped_column()
    portfolio_name: Mapped[str] = mapped_column()
    portfolio_type: Mapped[str] = mapped_column()
    is_active: Mapped[str] = mapped_column()
    reporting_currency: Mapped[str] = mapped_column(String(8), default="USD")
    base_currency: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)


class PortfolioHoldings(Base):
    """Maps to dbo.portfolio_holdings."""

    __tablename__ = "portfolio_holdings"
    __table_args__ = {"schema": "dbo"}

    ph_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    as_of_date: Mapped[str] = mapped_column()
    port_id: Mapped[int] = mapped_column()
    security_id: Mapped[int] = mapped_column()
    held_shares: Mapped[float] = mapped_column()
    upsert_date: Mapped[str] = mapped_column()
    upsert_by: Mapped[str] = mapped_column()


class IndexConstituents(Base):
    """Maps to reference.index_constituents."""

    __tablename__ = "index_constituents"
    __table_args__ = {"schema": "reference"}

    constituent_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    index_id: Mapped[int] = mapped_column()
    security_id: Mapped[int] = mapped_column()
    exchange_ticker: Mapped[str] = mapped_column()
    start_date: Mapped[str] = mapped_column()
    end_date: Mapped[Optional[str]] = mapped_column(nullable=True)
    source_vendor: Mapped[str] = mapped_column()
    upsert_date: Mapped[str] = mapped_column()
    upsert_by: Mapped[str] = mapped_column()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(date_str: str, fmt: str = "%Y-%m-%d") -> str:
    """Parse and reformat a date string."""
    return datetime.strptime(date_str, fmt).strftime(fmt)


def _execute_with_session(
    orm_session: Session,
    operation: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> None:
    """
    Execute a database operation with standardised error handling.

    Commits on success, rolls back on failure, and always closes the session.
    """
    try:
        operation(*args, **kwargs)
        orm_session.commit()
    except SQLAlchemyError as e:
        print(f"Database error: {e}")
        orm_session.rollback()
    except Exception as e:
        print(f"Unexpected error: {e}")
        orm_session.rollback()
    finally:
        orm_session.close()


def _read_table(orm_session: Session, orm_engine: Engine, model: Type[Base]) -> DataFrame:
    """Fetch all columns from an ORM-mapped table."""
    query = orm_session.query(*model.__table__.columns)
    return pd.read_sql_query(query.statement, con=orm_engine)


def _bulk_delete_insert(
    orm_session: Session,
    model: Type[Base],
    data_list: List[Dict[str, Any]],
    *filter_criteria: Any,
) -> None:
    """Delete matching rows then bulk-insert new records."""
    orm_session.query(model).filter(*filter_criteria).delete(synchronize_session=False)
    orm_session.bulk_insert_mappings(model, data_list)  # type: ignore


def _camel_to_snake(name: str) -> str:
    """Convert a CamelCase string to snake_case."""
    return re.sub("([A-Z])", r"_\1", name).lower().lstrip("_")


def _is_transient_connection_error(error: Exception) -> bool:
    """Return whether an error indicates a dropped SQL Server connection."""
    message = str(error)
    return any(code in message for code in ("08S01", "08006", "10054"))


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def get_db_connection(
    service_name: str = "ihub_sql_connection",
    server: str = "ops-store-server.database.windows.net",
    driver: str = "ODBC Driver 18 for SQL Server",
    max_retries: int = 3,
    retry_interval_minutes: int = 2,
) -> Tuple[sql.Engine, sql.Connection, str, Session]:
    """
    Establish a connection to SQL Server with retry logic.

    Returns
    -------
    Tuple of (engine, connection, connection_string, session).
    """
    db = keyring.get_password(service_name, "db")
    db_user = keyring.get_password(service_name, "uid")
    db_password = keyring.get_password(service_name, "pwd")

    # Local-instance override: if keyring stores a `server` entry plus a
    # `trusted=1` flag, build a Windows-auth (Trusted_Connection) connection
    # string for a local SQL Server (e.g. MENONPC\SQLEXPRESS). This lets the
    # whole app switch from the Azure host to a local instance without editing
    # any caller. Setting `trusted=0` (or removing it) reverts to Azure.
    local_server = keyring.get_password(service_name, "server")
    trusted = keyring.get_password(service_name, "trusted") == "1"

    if local_server and trusted:
        # Use a normal SQLAlchemy URL (server + database in the authority) so
        # both the ORM engine AND the ipython-sql `%sql` magic resolve the same
        # database. The bare `mssql+pyodbc:///?odbc_connect=...` form makes
        # ipython-sql fall back to a default DB (master), so INSERTs via the
        # magic silently land in the wrong place / don't persist.
        connection_string = (
            f"mssql+pyodbc://{local_server}/{db}"
            f"?trusted_connection=yes&driver={parse.quote_plus(driver)}"
            f"&TrustServerCertificate=yes&Encrypt=no&autocommit=true"
        )
    else:
        connection_string = (
            f"mssql+pyodbc://{db_user}:{db_password}"
            f"@{server}:1433/{db}"
            f"?driver={parse.quote_plus(driver)}&Encrypt=yes&TrustServerCertificate=no&autocommit=true"
        )

    for attempt in range(1, max_retries + 1):
        try:
            engine = sql.create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_recycle=1800,
            )
            connection = engine.connect()
            session = Session(engine)
            print("Database connection successful.")
            return engine, connection, connection_string, session
        except OperationalError as e:
            print(f"Attempt {attempt} failed with error:\n{e}")
            if attempt < max_retries:
                print(f"Retrying in {retry_interval_minutes} minutes...")
                time.sleep(retry_interval_minutes * 60)
            else:
                print("All retry attempts failed. Exiting.")
                raise

    raise RuntimeError("Database connection failed: maximum retries exceeded")


# ---------------------------------------------------------------------------
# Security Master
# ---------------------------------------------------------------------------


def read_security_master(orm_session: Session, orm_engine: Engine) -> DataFrame:
    """Fetch all records from security_master."""
    query = orm_session.query(SecurityMaster)
    return pd.read_sql_query(query.statement, con=orm_engine)


def write_security_master(equities_df: DataFrame, orm_session: Session) -> None:
    """
    Insert new records into security_master.

    Expects a DataFrame whose columns match SecurityMaster exactly (no
    security_id column — the DB generates it).  For a full upsert that
    de-duplicates on ISIN/CUSIP/FIGI, use the ingestion scripts which call
    this function after resolving existing security_ids themselves.
    """
    if equities_df.empty:
        print("No security master records to insert.")
        return

    def _insert(data: List[Dict[str, Any]]) -> None:
        orm_session.bulk_insert_mappings(SecurityMaster, data)  # type: ignore

    _execute_with_session(orm_session, _insert, equities_df.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# Security Vendor Xref
# ---------------------------------------------------------------------------


def read_security_vendor_xref(
    orm_session: Session,
    orm_engine: Engine,
    vendor: Optional[str] = None,
) -> DataFrame:
    """
    Fetch records from security_vendor_xref.

    Parameters
    ----------
    vendor : If provided, filters to a single vendor (e.g. 'Refinitiv').
    """
    query = orm_session.query(*SecurityVendorXref.__table__.columns)
    if vendor:
        query = query.filter(SecurityVendorXref.vendor == vendor)
    return pd.read_sql_query(query.statement, con=orm_engine)


def read_security_master_by_vendor(
    orm_session: Session,
    orm_engine: Engine,
    vendor: str,
) -> DataFrame:
    """
    Active securities joined to their active vendor xref for one vendor.

    Mirrors the canonical vendor-resolution query::

        SELECT
            sm.security_id,
            sm.name,
            xref.vendor,
            xref.vendor_ticker,
            xref.vendor_currency
        FROM security_master sm
        JOIN security_vendor_xref xref
          ON xref.security_id = sm.security_id
         AND xref.vendor      = @vendor
         AND xref.is_active   = 1
        WHERE sm.is_active = 1;

    Only the columns needed for vendor resolution are returned
    (security_id, name, vendor, vendor_ticker, vendor_currency), with
    clean, unprefixed names since none of them collide across the two
    tables.
    """
    sm, xref = SecurityMaster, SecurityVendorXref

    query = (
        orm_session.query(
            sm.security_id.label("security_id"),
            sm.name.label("name"),
            xref.vendor.label("vendor"),
            xref.vendor_ticker.label("vendor_ticker"),
            xref.vendor_currency.label("vendor_currency"),
        )
        .join(xref, xref.security_id == sm.security_id)
        .filter(
            xref.vendor == vendor,
            xref.is_active == 1,
            sm.is_active == 1,
        )
    )
    return pd.read_sql_query(query.statement, con=orm_engine)


def write_security_vendor_xref(
    xref_df: DataFrame,
    orm_session: Session,
    vendor: Optional[str] = None,
) -> None:
    """
    Upsert records into security_vendor_xref for a given vendor.

    Deletes all existing rows matching (vendor × security_ids in the batch)
    then bulk-inserts the new rows.  Re-running is therefore idempotent.

    Parameters
    ----------
    xref_df : DataFrame with columns matching SecurityVendorXref (no xref_id).
    vendor  : Vendor name used to scope the delete.  If None, inferred from
              the first row of xref_df.
    """
    if xref_df.empty:
        print("No vendor xref records to write.")
        return

    resolved_vendor = vendor or xref_df["vendor"].iloc[0]
    security_ids = xref_df["security_id"].unique().tolist()

    def _upsert(data_list: List[Dict[str, Any]]) -> None:
        _bulk_delete_insert(
            orm_session,
            SecurityVendorXref,
            data_list,
            SecurityVendorXref.vendor == resolved_vendor,
            SecurityVendorXref.security_id.in_(security_ids),
        )
        print(f"Wrote {len(data_list)} vendor xref rows for {resolved_vendor}.")

    _execute_with_session(orm_session, _upsert, xref_df.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


def read_portfolio(
    orm_session: Session,
    orm_engine: Engine,
    portfolio_short_names: List[str],
) -> DataFrame:
    """Fetch records from portfolio table filtered by short names."""
    query = orm_session.query(
        Portfolio.port_id,
        Portfolio.portfolio_short_name,
        Portfolio.portfolio_name,
        Portfolio.portfolio_type,
    ).filter(Portfolio.portfolio_short_name.in_(portfolio_short_names))
    return pd.read_sql_query(query.statement, con=orm_engine)


def write_portfolio_holdings(df_holdings: DataFrame, orm_session: Session) -> None:
    """
    Upsert records into portfolio_holdings.

    Deletes existing records for the same portfolio(s) and as_of_date before
    inserting, so re-running a daily load is safe.
    """
    if df_holdings.empty:
        print("No data to write.")
        return

    as_of_date = df_holdings["as_of_date"].iloc[0]
    port_ids = df_holdings["port_id"].unique().tolist()

    df_holdings = df_holdings.copy()
    df_holdings["upsert_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_holdings["upsert_by"] = "daily_portfolio_load.py"

    def _upsert(data_list: List[Dict[str, Any]]) -> None:
        _bulk_delete_insert(
            orm_session,
            PortfolioHoldings,
            data_list,
            PortfolioHoldings.as_of_date == as_of_date,
            PortfolioHoldings.port_id.in_(port_ids),
        )

    _execute_with_session(orm_session, _upsert, df_holdings.to_dict(orient="records"))


def read_portfolio_holdings(
    orm_session: Session,
    orm_engine: Engine,
    start_date: str,
    end_date: str,
) -> DataFrame:
    """Fetch records from portfolio_holdings within a date range."""
    start_date, end_date = _parse_date(start_date), _parse_date(end_date)

    query = orm_session.query(
        PortfolioHoldings.as_of_date,
        PortfolioHoldings.port_id,
        PortfolioHoldings.security_id,
        PortfolioHoldings.held_shares,
    ).filter(PortfolioHoldings.as_of_date.between(start_date, end_date))

    return pd.read_sql_query(query.statement, con=orm_engine)


# ---------------------------------------------------------------------------
# Market Data
# ---------------------------------------------------------------------------


def read_market_data(
    orm_session: Session,
    orm_engine: Engine,
    start_date: str,
    end_date: str,
) -> DataFrame:
    """Fetch records from market_data within a date range."""
    start_date, end_date = _parse_date(start_date), _parse_date(end_date)

    query = orm_session.query(*MarketData.__table__.columns).filter(MarketData.as_of_date.between(start_date, end_date))
    return pd.read_sql_query(query.statement, con=orm_engine)


def write_market_data(market_data: DataFrame, orm_session: Session) -> None:
    """
    Upsert records into market_data.

    Deletes existing records for the same security(s) and date(s) before
    inserting, so re-running is safe.
    """
    if market_data.empty:
        print("No market data to write.")
        return

    # Clean rows so the bulk insert doesn't abort (and roll back the whole batch).
    # market_data columns are NOT NULL, so any NaT date or NULL price makes pyodbc
    # reject the ENTIRE batch. Strategy:
    #   - drop rows with no usable date (genuinely unusable);
    #   - fill a missing open/high/low from that row's own close (standard EOD tolerance);
    #   - fill missing volume with 0;
    #   - drop rows that still lack close/adj_close (cannot compute returns anyway).
    market_data = market_data.copy()
    market_data = market_data.dropna(subset=["as_of_date"])
    before = len(market_data)
    price_cols = ["open", "high", "low", "close", "adj_close"]
    present_price = [c for c in price_cols if c in market_data.columns]
    for c in ["open", "high", "low", "adj_close"]:
        if c in market_data.columns:
            market_data[c] = market_data[c].fillna(market_data["close"])
    if "volume" in market_data.columns:
        market_data["volume"] = market_data["volume"].fillna(0)
    market_data = market_data.dropna(subset=[c for c in ("close", "adj_close") if c in market_data.columns])
    dropped = before - len(market_data)
    if dropped:
        print(f"  cleaned {dropped} unusable row(s) before write")
    if market_data.empty:
        print("No valid market data to write after cleaning.")
        return

    as_of_dates = market_data["as_of_date"].unique().tolist()
    security_ids = market_data["security_id"].unique().tolist()

    def _upsert(data_list: List[Dict[str, Any]]) -> None:
        _bulk_delete_insert(
            orm_session,
            MarketData,
            data_list,
            MarketData.as_of_date.in_(as_of_dates),
            MarketData.security_id.in_(security_ids),
        )

    _execute_with_session(orm_session, _upsert, market_data.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# Index Constituents
# ---------------------------------------------------------------------------


def read_index_constituents(
    orm_session: Session,
    orm_engine: Engine,
    start_date: str,
    end_date: str,
) -> DataFrame:
    """Fetch all records from index_constituents."""
    _parse_date(start_date)  # validate inputs
    _parse_date(end_date)
    return _read_table(orm_session, orm_engine, IndexConstituents)


def write_index_constituents(index_constituents: DataFrame, orm_session: Session) -> None:
    """
    Upsert records into index_constituents.

    Deletes existing records for the same index_id(s) before inserting.
    """
    if index_constituents.empty:
        print("No index constituent records to write.")
        return

    index_ids = index_constituents["index_id"].unique().tolist()

    def _upsert(data_list: List[Dict[str, Any]]) -> None:
        _bulk_delete_insert(
            orm_session,
            IndexConstituents,
            data_list,
            IndexConstituents.index_id.in_(index_ids),
        )

    _execute_with_session(orm_session, _upsert, index_constituents.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# Security Fundamentals
# ---------------------------------------------------------------------------


def read_security_fundamentals(
    orm_session: Session,
    orm_engine: Engine,
    metric_type: Optional[str] = None,
) -> DataFrame:
    """
    Fetch records from security_fundamentals.

    Parameters
    ----------
    metric_type : If provided, filters by metric_type and renames
                  metric_value to the snake_case metric name.
    """
    query = orm_session.query(*SecurityFundamentals.__table__.columns)
    if metric_type:
        query = query.filter(SecurityFundamentals.metric_type == metric_type)

    df = pd.read_sql_query(query.statement, con=orm_engine)

    if metric_type and "metric_value" in df.columns:
        df.rename(columns={"metric_value": _camel_to_snake(metric_type)}, inplace=True)

    return df


def write_security_fundamentals(
    fundamental_data: DataFrame,
    orm_session: Session,
    _retry_attempt: int = 0,
) -> None:
    """
    Write security fundamentals with a NON-DESTRUCTIVE upsert.

    Rows are matched on the natural key
    ``(security_id, metric_type, source_vendor, effective_date)``:

    - key exists    : the row's ``metric_value`` (and ``end_date``) is updated in place.
    - key absent    : a new row is inserted.

    Crucially, this never deletes or closes rows that are NOT in the incoming
    frame. A prior "versioning" implementation closed/deleted existing rows when
    the incoming row count differed, which meant re-writing a *windowed* subset
    (e.g. a 2025-only shares pull) would silently destroy the out-of-window
    history. That behaviour is gone: every existing row outside the incoming key
    set is preserved.

    Parameters
    ----------
    fundamental_data : DataFrame with columns security_id, metric_type,
                       metric_value, source_vendor, effective_date, and
                       optionally end_date.
    """
    if fundamental_data.empty:
        print("No fundamental data to insert.")
        return

    # A fundamentals row with no resolved security is invalid and, worse,
    # pyodbc cannot bind pandas' nullable <NA> (raises HY105 / ProgrammingError).
    # Drop those rows defensively so any caller is protected.
    if fundamental_data["security_id"].isna().any():
        dropped = int(fundamental_data["security_id"].isna().sum())
        print(f"Dropping {dropped} fundamentals row(s) with unresolved security_id.")
        fundamental_data = fundamental_data[fundamental_data["security_id"].notna()].copy()
    if fundamental_data.empty:
        print("No fundamental data to insert after dropping unresolved security_id.")
        return

    # Coerce security_id to a plain (nullable=False) int so it binds cleanly.
    fundamental_data["security_id"] = fundamental_data["security_id"].astype("int64").astype(int)

    # Normalise effective_date to a python date for exact key matching.
    fundamental_data["effective_date"] = pd.to_datetime(
        fundamental_data["effective_date"]
    ).dt.date

    # Snapshot end_date as a date (or None) for insertion.
    fundamental_data["end_date"] = fundamental_data["end_date"].apply(
        lambda ed: pd.to_datetime(ed).date() if ed is not None and not pd.isna(ed) else None
    )

    # Non-destructive set-based upsert via a staging temp table + MERGE.
    # (The previous IN-list approach blew past pyodbc's parameter cap for
    # time-series, which carry hundreds of thousands of distinct dates.)
    # Only the incoming key set is touched; rows outside it are preserved.
    df = fundamental_data[
        ["security_id", "metric_type", "metric_value", "source_vendor", "effective_date", "end_date"]
    ].copy()

    # De-duplicate the natural key so the MERGE source has exactly one row per
    # (security_id, metric_type, source_vendor, effective_date).
    df = df.drop_duplicates(
        subset=["security_id", "metric_type", "source_vendor", "effective_date"],
        keep="last",
    )
    n = len(df)
    print(f"Upserting {n} security_fundamentals rows (non-destructive MERGE)...")
    bind = orm_session.connection()  # share the SAME connection the temp table lives on
    try:
        # Fresh temp table.
        orm_session.execute(sql.text("IF OBJECT_ID('tempdb..#sf_stage') IS NOT NULL DROP TABLE #sf_stage"))
        orm_session.execute(sql.text(
            "CREATE TABLE #sf_stage ("
            " security_id INT, metric_type VARCHAR(64), metric_value FLOAT, "
            " source_vendor VARCHAR(64), effective_date DATE, end_date DATE)"
        ))
        # Stage via raw pyodbc fast_executemany (0.8ms/row vs SQLAlchemy's
        # ~9ms/row per-row round-trip). We use the SAME underlying connection
        # the session uses, so the #sf_stage temp table stays visible to the
        # MERGE below. (SQLAlchemy's executemany ignores fast_executemany and is
        # too slow for ~100k-row frames.)
        raw_conn = bind.connection.driver_connection  # pyodbc.Connection
        cur = raw_conn.cursor()
        cur.fast_executemany = True
        cur.executemany(
            "INSERT INTO #sf_stage (security_id, metric_type, metric_value, "
            "source_vendor, effective_date, end_date) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    int(r.security_id),
                    str(r.metric_type),
                    None if pd.isna(r.metric_value) else float(r.metric_value),
                    str(r.source_vendor),
                    r.effective_date,
                    r.end_date,
                )
                for r in df.itertuples(index=False)
            ],
        )
        cur.close()
        orm_session.execute(sql.text(
            "MERGE dbo.security_fundamentals AS tgt "
            "USING #sf_stage AS src "
            "  ON tgt.security_id    = src.security_id "
            " AND tgt.metric_type    = src.metric_type "
            " AND tgt.source_vendor  = src.source_vendor "
            " AND tgt.effective_date = src.effective_date "
            "WHEN MATCHED THEN "
            "  UPDATE SET tgt.metric_value = src.metric_value, tgt.end_date = src.end_date "
            "WHEN NOT MATCHED THEN "
            "  INSERT (security_id, metric_type, metric_value, source_vendor, effective_date, end_date) "
            "  VALUES (src.security_id, src.metric_type, src.metric_value, src.source_vendor, src.effective_date, src.end_date);"
        ))
        orm_session.execute(sql.text("DROP TABLE #sf_stage"))
        orm_session.commit()
        print(f"Security fundamentals data successfully written ({n} rows upserted).")
    except SQLAlchemyError as e:
        orm_session.rollback()
        if _is_transient_connection_error(e) and _retry_attempt < 2:
            bind.invalidate()
            orm_session.close()
            print("Transient SQL Server connection failure; retrying fundamentals upsert...")
            time.sleep(2 ** _retry_attempt)
            write_security_fundamentals(fundamental_data, orm_session, _retry_attempt + 1)
            return
        print(f"Database error during security_fundamentals upsert: {e}")
        raise
    except Exception as e:
        orm_session.rollback()
        if _is_transient_connection_error(e) and _retry_attempt < 2:
            bind.invalidate()
            orm_session.close()
            print("Transient SQL Server connection failure; retrying fundamentals upsert...")
            time.sleep(2 ** _retry_attempt)
            write_security_fundamentals(fundamental_data, orm_session, _retry_attempt + 1)
            return
        print(f"Unexpected error during security_fundamentals upsert: {e}")
        raise


# ---------------------------------------------------------------------------
# Composite Queries
# ---------------------------------------------------------------------------


def get_portfolio_market_data(
    orm_session: Session,
    orm_engine: Engine,
    start_date: str,
    end_date: str,
    portfolio_short_names: List[str],
) -> DataFrame:
    """
    Join portfolio → holdings → security master → market data → fundamentals.

    Fundamentals are aligned to market dates using a group-wise backward
    merge_asof, so each market data row carries the most recently available
    fundamental value for that security.

    Parameters
    ----------
    start_date             : "YYYY-MM-DD"
    end_date               : "YYYY-MM-DD"
    portfolio_short_names  : List of portfolio short names to filter by.

    Returns
    -------
    Merged DataFrame with one row per (portfolio, security, date).
    """
    df_securities = read_security_master(orm_session, orm_engine)
    df_market_data = read_market_data(orm_session, orm_engine, start_date, end_date)
    # Defensive: market_data can carry duplicate (security_id, as_of_date) rows
    # (e.g. a re-pull that wrote over an existing date without deleting first).
    # Those would fan every holding into 2 rows in the merge below and (via
    # weights.py's mean-pivot) halve the reconstructed-index weights. Keep one
    # row per security/date.
    if {"security_id", "as_of_date"}.issubset(df_market_data.columns):
        before = len(df_market_data)
        df_market_data = df_market_data.drop_duplicates(subset=["security_id", "as_of_date"], keep="last")
        if len(df_market_data) != before:
            print(f"[get_portfolio_market_data] dropped {before - len(df_market_data)} "
                  f"duplicate market_data row(s) on (security_id, as_of_date)")
    df_portfolio = read_portfolio(orm_session, orm_engine, portfolio_short_names)
    df_holdings = read_portfolio_holdings(orm_session, orm_engine, start_date, end_date)
    df_fundamentals = read_security_fundamentals(orm_session, orm_engine, "shares_outstanding")

    # The `portfolio` table can accumulate DUPLICATE rows (e.g. the demo notebook
    # was run more than once, or cell 5's truncate missed it), so the same
    # portfolio_short_name maps to two port_ids. The downstream merge on `port_id`
    # then fans every holding into TWO rows, doubling the SP500 frame and -- because
    # weights.py pivots with aggfunc='mean' -- halving every constituent weight to
    # ~0.5 (the reconstructed index then returns ~half of the real one). Keep only
    # the lowest port_id per short name so each portfolio appears exactly once.
    if "port_id" in df_portfolio.columns and "portfolio_short_name" in df_portfolio.columns:
        before = len(df_portfolio)
        df_portfolio = (
            df_portfolio.sort_values("port_id")
            .drop_duplicates(subset=["portfolio_short_name"], keep="first")
            .reset_index(drop=True)
        )
        if len(df_portfolio) != before:
            print(f"[get_portfolio_market_data] dropped {before - len(df_portfolio)} "
                  f"duplicate portfolio row(s) (same short_name, multiple port_ids)")

    # portfolio_holdings.as_of_date is stored as a string, while market_data.as_of_date
    # comes back as a datetime.date. The inner merge on ["security_id","as_of_date"]
    # would otherwise match nothing (str vs date) and yield an empty frame -> the
    # downstream pd.concat raises "No objects to concatenate". Normalize both to dates.
    df_holdings["as_of_date"] = pd.to_datetime(df_holdings["as_of_date"], errors="coerce").dt.date
    df_holdings = df_holdings.dropna(subset=["as_of_date"])

    # Merge portfolio → holdings → securities → market data
    df_portfolio_market_data = (
        df_portfolio.merge(df_holdings, on="port_id")
        .merge(df_securities, on="security_id")
        .merge(df_market_data, on=["security_id", "as_of_date"])
    )

    if df_portfolio_market_data.empty:
        # Surface an actionable message instead of failing later on pd.concat.
        names = ", ".join(portfolio_short_names)
        raise ValueError(
            f"get_portfolio_market_data returned 0 rows for portfolios [{names}]. "
            "Likely cause: market_data is empty for the requested window, or holdings "
            "and market_data share no (security_id, as_of_date) pairs."
        )

    # Coerce dates and drop invalid rows
    df_portfolio_market_data["as_of_date"] = pd.to_datetime(df_portfolio_market_data["as_of_date"], errors="coerce")
    df_fundamentals["effective_date"] = pd.to_datetime(df_fundamentals["effective_date"], errors="coerce")
    df_portfolio_market_data.dropna(subset=["as_of_date"], inplace=True)
    df_fundamentals.dropna(subset=["effective_date"], inplace=True)

    # Align dtypes for join key
    df_portfolio_market_data["security_id"] = df_portfolio_market_data["security_id"].astype(int)
    df_fundamentals["security_id"] = df_fundamentals["security_id"].astype(int)

    # Group-wise backward merge_asof to align fundamentals to market dates
    merged_rows = []
    fundamental_cols = set(df_fundamentals.columns) - set(df_portfolio_market_data.columns)

    for sec_id, df_left in df_portfolio_market_data.groupby("security_id"):
        df_left = df_left.sort_values("as_of_date").reset_index(drop=True)
        df_right = df_fundamentals[df_fundamentals["security_id"] == sec_id].sort_values("effective_date").reset_index(drop=True)

        if not df_right.empty:
            df_merged = pd.merge_asof(
                df_left,
                df_right,
                by="security_id",
                left_on="as_of_date",
                right_on="effective_date",
                direction="backward",
            )
        else:
            df_merged = df_left.copy()
            for col in fundamental_cols:
                df_merged[col] = pd.NA

        # Backward merge_asof leaves NaN for any date BEFORE the first
        # fundamentals effective_date (e.g. a constituent whose
        # shares_outstanding only starts months/years into the window). Those
        # NaN shares make weights.py assign the name ZERO weight for the entire
        # pre-period, which both under-weights the reconstructed index and, when
        # many names share the gap, distorts its shape vs the real index. Carry
        # each security's EARLIEST available fundamental value forward across
        # its pre-period (stale-but-nonzero) so every priced constituent keeps a
        # weight. Per-security only -- never cross security boundaries.
        if fundamental_cols:
            _fc = list(fundamental_cols)
            df_merged[_fc] = (
                df_merged.groupby("security_id")[_fc]
                .transform(lambda s: s.ffill().bfill())
            )

        merged_rows.append(df_merged)

    return pd.concat(
        [df.dropna(axis=1, how="all") for df in merged_rows if not df.empty],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Analytics Layer (persistent daily analytics — README differentiator)
#   factor_scores, portfolio_returns, attribution
# Models are defined once in schema_analytics and reused by both backends.
# ---------------------------------------------------------------------------

from .schema_analytics import (  # noqa: E402  (import after module defs)
    FactorScores,
    PortfolioReturns,
    Attribution,
    create_analytics_tables,
)
from .schema_fx import (  # noqa: E402
    FxRates,
    RiskSnapshots,
    FactorExposures,
    create_fx_tables,
)


def write_factor_scores(df: DataFrame, orm_session: Session) -> None:
    """Upsert factor scores. One row per (date, security, factor, universe)."""
    if df.empty:
        print("No factor scores to write.")
        return
    df = df.copy()
    df["upsert_date"] = datetime.now().strftime("%Y-%m-%d")
    as_of = df["as_of_date"].unique().tolist()
    secs = df["security_id"].unique().tolist()

    def _upsert(data_list: List[Dict[str, Any]]) -> None:
        _bulk_delete_insert(
            orm_session, FactorScores, data_list,
            FactorScores.as_of_date.in_(as_of),
            FactorScores.security_id.in_(secs),
        )

    _execute_with_session(orm_session, _upsert, df.to_dict(orient="records"))


def read_factor_scores(orm_session, orm_engine, as_of_date=None, factor_name=None) -> DataFrame:
    """Read factor scores, optionally filtered by date / factor."""
    q = orm_session.query(*FactorScores.__table__.columns)
    if as_of_date:
        q = q.filter(FactorScores.as_of_date == _parse_date(as_of_date))
    if factor_name:
        q = q.filter(FactorScores.factor_name == factor_name)
    return pd.read_sql_query(q.statement, con=orm_engine)


def write_portfolio_returns(df: DataFrame, orm_session: Session) -> None:
    """Upsert daily portfolio returns."""
    if df.empty:
        print("No portfolio returns to write.")
        return
    df = df.copy()
    df["upsert_date"] = datetime.now().strftime("%Y-%m-%d")
    as_of = df["as_of_date"].unique().tolist()
    ports = df["port_id"].unique().tolist()

    def _upsert(data_list: List[Dict[str, Any]]) -> None:
        _bulk_delete_insert(
            orm_session, PortfolioReturns, data_list,
            PortfolioReturns.as_of_date.in_(as_of),
            PortfolioReturns.port_id.in_(ports),
        )

    _execute_with_session(orm_session, _upsert, df.to_dict(orient="records"))


def write_attribution(df: DataFrame, orm_session: Session) -> None:
    """Upsert daily performance attribution rows."""
    if df.empty:
        print("No attribution rows to write.")
        return
    df = df.copy()
    df["upsert_date"] = datetime.now().strftime("%Y-%m-%d")
    as_of = df["as_of_date"].unique().tolist()
    ports = df["port_id"].unique().tolist()

    def _upsert(data_list: List[Dict[str, Any]]) -> None:
        _bulk_delete_insert(
            orm_session, Attribution, data_list,
            Attribution.as_of_date.in_(as_of),
            Attribution.port_id.in_(ports),
        )

    _execute_with_session(orm_session, _upsert, df.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# FX / Risk / Factor-exposure helpers (schema_fx)
# ---------------------------------------------------------------------------

def write_fx_rates(df: DataFrame, orm_session: Session) -> None:
    """Upsert FX rates. One row per (from, to, date, vendor)."""
    if df.empty:
        print("No FX rates to write.")
        return
    df = df.copy()
    df["upsert_date"] = datetime.now().strftime("%Y-%m-%d")
    _execute_with_session(
        orm_session,
        lambda data: _bulk_delete_insert(
            orm_session, FxRates, data,
            FxRates.as_of_date.in_(df["as_of_date"].unique().tolist()),
            FxRates.source_vendor.in_(df["source_vendor"].unique().tolist()),
        ),
        df.to_dict(orient="records"),
    )


def write_risk_snapshots(df: DataFrame, orm_session: Session) -> None:
    """Upsert daily risk snapshots. One row per (date, port, metric, vendor)."""
    if df.empty:
        print("No risk snapshots to write.")
        return
    df = df.copy()
    df["upsert_date"] = datetime.now().strftime("%Y-%m-%d")
    _execute_with_session(
        orm_session,
        lambda data: _bulk_delete_insert(
            orm_session, RiskSnapshots, data,
            RiskSnapshots.as_of_date.in_(df["as_of_date"].unique().tolist()),
            RiskSnapshots.port_id.in_(df["port_id"].unique().tolist()),
        ),
        df.to_dict(orient="records"),
    )


def write_factor_exposures(df: DataFrame, orm_session: Session) -> None:
    """Upsert daily portfolio factor exposures."""
    if df.empty:
        print("No factor exposures to write.")
        return
    df = df.copy()
    df["upsert_date"] = datetime.now().strftime("%Y-%m-%d")
    _execute_with_session(
        orm_session,
        lambda data: _bulk_delete_insert(
            orm_session, FactorExposures, data,
            FactorExposures.as_of_date.in_(df["as_of_date"].unique().tolist()),
            FactorExposures.port_id.in_(df["port_id"].unique().tolist()),
        ),
        df.to_dict(orient="records"),
    )


def compute_and_store_factors(prices: DataFrame, factors: dict, universe: str,
                              source_vendor: str, orm_session: Session) -> None:
    """Compute factor scores for a price panel and persist to factor_scores.

    Args:
        prices: wide DataFrame, index=date, columns=security_id, values=price.
        factors: dict mapping factor_name -> callable/Factor returning a score
                 DataFrame with the same shape as ``prices``.
        universe: label for the universe (e.g. 'SP500').
        source_vendor: vendor label for the price source.
        orm_session: SQLAlchemy session for the target backend.
    """
    from analytics.factors import Factor  # local import avoids hard dep

    rows = []
    today = datetime.now().strftime("%Y-%m-%d")
    for name, fac in factors.items():
        scores = fac.compute(prices) if isinstance(fac, Factor) else fac(prices)
        if scores is None or scores.notna().sum().sum() == 0:
            print(f"  skip factor '{name}': produced no non-null scores (check lookback vs history)")
            continue
        rank = scores.rank(axis=1, pct=True)
        for dt, row in scores.iterrows():
            for sec, val in row.items():
                if pd.isna(val):
                    continue
                rows.append({
                    "as_of_date": pd.to_datetime(dt).strftime("%Y-%m-%d"),
                    "security_id": int(sec),
                    "factor_name": name,
                    "factor_value": float(val),
                    "rank_pct": float(rank.loc[dt, sec]) if not pd.isna(rank.loc[dt, sec]) else None,
                    "universe": universe,
                    "source_vendor": source_vendor,
                    "upsert_date": today,
                })
    if rows:
        write_factor_scores(pd.DataFrame(rows), orm_session)
