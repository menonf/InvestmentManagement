"""SQLAlchemy ORM Module — Databricks Delta Lake backend.

Dependencies:
    pip install databricks-sqlalchemy databricks-sql-connector pandas keyring sqlalchemy

Connection secrets expected in keyring under service_name:
    "host"       → Databricks workspace hostname  e.g. adb-xxxx.azuredatabricks.net
    "http_path"  → SQL warehouse HTTP path        e.g. /sql/1.0/warehouses/xxxx
    "token"      → Personal access token (PAT)
    "catalog"    → Unity Catalog name             e.g. main
    "schema"     → Default schema                 e.g. default
"""

from __future__ import annotations

import datetime
import re
import time
from datetime import date
from typing import Any, Callable, List, Optional, Tuple, Type

import keyring
import pandas as pd
import sqlalchemy as sql
from pandas import DataFrame
from sqlalchemy import Engine, String, delete, insert, update
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

# ---------------------------------------------------------------------------
# ORM Base & Models
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """SQLAlchemy Base Class."""

    pass


class SecurityMaster(Base):
    """Maps to security_master table (default schema)."""

    __tablename__ = "security_master"

    security_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[Optional[str]] = mapped_column()
    name: Mapped[Optional[str]] = mapped_column()
    isin: Mapped[Optional[str]] = mapped_column()
    sedol: Mapped[Optional[str]] = mapped_column()
    cusip: Mapped[Optional[str]] = mapped_column()
    figi: Mapped[Optional[str]] = mapped_column()
    loanxid: Mapped[Optional[str]] = mapped_column()
    country: Mapped[Optional[str]] = mapped_column()
    currency: Mapped[Optional[str]] = mapped_column()
    sector: Mapped[Optional[str]] = mapped_column()
    industry_group: Mapped[Optional[str]] = mapped_column()
    industry: Mapped[Optional[str]] = mapped_column()
    security_type: Mapped[str] = mapped_column()
    asset_class: Mapped[str] = mapped_column()
    region: Mapped[Optional[str]] = mapped_column()
    exchange_mic: Mapped[Optional[str]] = mapped_column()
    listing_country: Mapped[Optional[str]] = mapped_column()
    exchange: Mapped[Optional[str]] = mapped_column()
    is_active: Mapped[Optional[int]] = mapped_column()
    source_vendor: Mapped[str] = mapped_column()
    upsert_date: Mapped[Optional[datetime.datetime]] = mapped_column()
    upsert_by: Mapped[Optional[str]] = mapped_column()


class SecurityFundamentals(Base):
    """Maps to security_fundamentals table (default schema).

    Now uses sf_id as the true primary key — matching the Databricks DDL.
    The compound-key workaround from the Azure SQL version is no longer needed.
    """

    __tablename__ = "security_fundamentals"

    sf_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    security_id: Mapped[int] = mapped_column()
    metric_type: Mapped[str] = mapped_column()
    metric_value: Mapped[float] = mapped_column()
    source_vendor: Mapped[str] = mapped_column()
    effective_date: Mapped[date] = mapped_column()
    end_date: Mapped[Optional[date]] = mapped_column(nullable=True)


class MarketData(Base):
    """Maps to market_data table (default schema)."""

    __tablename__ = "market_data"

    md_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    as_of_date: Mapped[date] = mapped_column()
    security_id: Mapped[int] = mapped_column()
    open: Mapped[Optional[float]] = mapped_column()
    high: Mapped[Optional[float]] = mapped_column()
    low: Mapped[Optional[float]] = mapped_column()
    close: Mapped[Optional[float]] = mapped_column()
    adj_close: Mapped[Optional[float]] = mapped_column()
    volume: Mapped[Optional[float]] = mapped_column()
    dividends: Mapped[Optional[float]] = mapped_column()
    stock_splits: Mapped[Optional[float]] = mapped_column()
    interval: Mapped[Optional[str]] = mapped_column()
    dataload_date: Mapped[Optional[datetime.datetime]] = mapped_column()
    price_currency: Mapped[Optional[str]] = mapped_column(String(8))
    source_vendor: Mapped[Optional[str]] = mapped_column(String(64))


class Portfolio(Base):
    """Maps to portfolio table (default schema)."""

    __tablename__ = "portfolio"

    port_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    portfolio_short_name: Mapped[Optional[str]] = mapped_column()
    portfolio_name: Mapped[Optional[str]] = mapped_column()
    portfolio_type: Mapped[Optional[str]] = mapped_column()
    is_active: Mapped[Optional[int]] = mapped_column()
    reporting_currency: Mapped[Optional[str]] = mapped_column(String(8))
    base_currency: Mapped[Optional[str]] = mapped_column(String(8))


class PortfolioHoldings(Base):
    """Maps to portfolio_holdings table (default schema)."""

    __tablename__ = "portfolio_holdings"

    ph_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    as_of_date: Mapped[Optional[datetime.date]] = mapped_column()
    port_id: Mapped[Optional[int]] = mapped_column()
    security_id: Mapped[Optional[int]] = mapped_column()
    held_shares: Mapped[Optional[float]] = mapped_column()
    upsert_date: Mapped[Optional[datetime.datetime]] = mapped_column()
    upsert_by: Mapped[Optional[str]] = mapped_column()


class IndexConstituents(Base):
    """Maps to reference.index_constituents table."""

    __tablename__ = "index_constituents"
    __table_args__ = {"schema": "reference"}

    constituent_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    index_id: Mapped[int] = mapped_column()
    security_id: Mapped[Optional[int]] = mapped_column()
    exchange_ticker: Mapped[str] = mapped_column()
    start_date: Mapped[date] = mapped_column()
    end_date: Mapped[Optional[date]] = mapped_column(nullable=True)
    source_vendor: Mapped[Optional[str]] = mapped_column()
    upsert_date: Mapped[datetime.datetime] = mapped_column()
    upsert_by: Mapped[str] = mapped_column()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_date(date_str: str, fmt: str = "%Y-%m-%d") -> str:
    """Parse and reformat a date string."""
    return datetime.datetime.strptime(date_str, fmt).strftime(fmt)


def _execute_with_session(orm_session: Session, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
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


def _read_table(orm_session: Session, orm_engine: Engine, model: Type[DeclarativeBase]) -> DataFrame:
    """Fetch all columns from an ORM-mapped table."""
    query = orm_session.query(*model.__table__.columns)
    return pd.read_sql_query(query.statement, con=orm_engine)


def _bulk_insert(orm_session: Session, model: Type[DeclarativeBase], data_list: List[dict[str, Any]]) -> None:
    """Bulk-insert records using SQLAlchemy 2.0-style execute+insert.

    Replaces the deprecated bulk_insert_mappings().
    """
    if data_list:
        orm_session.execute(insert(model), data_list)


def _bulk_delete_insert(
    orm_session: Session,
    model: Type[DeclarativeBase],
    data_list: List[dict[str, Any]],
    *filter_criteria: Any,
) -> None:
    """Delete matching rows then bulk-insert new records."""
    orm_session.query(model).filter(*filter_criteria).delete(synchronize_session=False)
    _bulk_insert(orm_session, model, data_list)


def _camel_to_snake(name: str) -> str:
    """Convert a CamelCase string to snake_case."""
    return re.sub("([A-Z])", r"_\1", name).lower().lstrip("_")


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------


def get_db_connection(
    service_name: str = "ihub_databricks_connection",
    max_retries: int = 3,
    retry_interval_minutes: int = 2,
) -> Tuple[sql.Engine, sql.Connection, str, Session]:
    """
    Establish a SQLAlchemy connection to a Databricks SQL warehouse.

    Reads the following secrets from keyring under service_name:
        host       — workspace hostname, e.g. adb-xxxx.azuredatabricks.net
        http_path  — SQL warehouse HTTP path, e.g. /sql/1.0/warehouses/xxxx
        token      — personal access token (PAT)
        catalog    — Unity Catalog name (e.g. 'main')
        schema     — default schema (e.g. 'default')

    Returns:
        Tuple of (engine, connection, connection_string, session).
    """
    host = keyring.get_password(service_name, "host")
    http_path = keyring.get_password(service_name, "http_path")
    token = keyring.get_password(service_name, "token")
    catalog = keyring.get_password(service_name, "catalog")
    schema = keyring.get_password(service_name, "schema")

    # databricks-sqlalchemy connection string format:
    # databricks+connector://token:<PAT>@<host>/<catalog>?http_path=<http_path>&schema=<schema>
    connection_string = f"databricks+connector://token:{token}@{host}/{catalog}" f"?http_path={http_path}&schema={schema}"

    for attempt in range(1, max_retries + 1):
        try:
            engine = sql.create_engine(connection_string)
            connection = engine.connect()
            session = Session(engine)
            print("Databricks connection successful.")
            return engine, connection, connection_string, session
        except OperationalError as e:
            print(f"Attempt {attempt} failed:\n{e}")
            if attempt < max_retries:
                print(f"Retrying in {retry_interval_minutes} minutes...")
                time.sleep(retry_interval_minutes * 60)
            else:
                print("All retry attempts failed.")
                raise

    raise RuntimeError("Databricks connection failed: maximum retries exceeded")


# ---------------------------------------------------------------------------
# Security Master
# ---------------------------------------------------------------------------


def read_security_master(orm_session: Session, orm_engine: Engine) -> DataFrame:
    """Fetch all records from security_master."""
    query = orm_session.query(SecurityMaster)
    return pd.read_sql_query(query.statement, con=orm_engine)


def write_security_master(equities_df: DataFrame, orm_session: Session) -> None:
    """Insert records into security_master."""

    def _insert(data: list[dict[str, Any]]) -> None:
        _bulk_insert(orm_session, SecurityMaster, data)

    _execute_with_session(orm_session, _insert, equities_df.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


def read_portfolio(
    orm_session: Session,
    orm_engine: Engine,
    portfolio_short_names: List[str],
) -> DataFrame:
    """Fetch records from portfolio filtered by short names."""
    query = orm_session.query(
        Portfolio.port_id,
        Portfolio.portfolio_short_name,
        Portfolio.portfolio_name,
        Portfolio.portfolio_type,
    ).filter(Portfolio.portfolio_short_name.in_(portfolio_short_names))
    return pd.read_sql_query(query.statement, con=orm_engine)


def write_portfolio_holdings(df_holdings: DataFrame, orm_session: Session) -> None:
    """Upsert records into portfolio_holdings (delete + insert for given portfolio/date)."""
    if df_holdings.empty:
        print("No data to write.")
        return

    as_of_date = df_holdings["as_of_date"].iloc[0]
    port_ids = df_holdings["port_id"].unique().tolist()

    df_holdings = df_holdings.copy()
    df_holdings["upsert_date"] = datetime.datetime.now()
    df_holdings["upsert_by"] = "daily_portfolio_load.py"

    def _upsert(data_list: list[dict[str, Any]]) -> None:
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
    """Fetch portfolio_holdings records within a date range."""
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
    """Fetch market_data records within a date range."""
    start_date, end_date = _parse_date(start_date), _parse_date(end_date)

    query = orm_session.query(*MarketData.__table__.columns).filter(MarketData.as_of_date.between(start_date, end_date))
    return pd.read_sql_query(query.statement, con=orm_engine)


def write_market_data(market_data: DataFrame, orm_session: Session) -> None:
    """Upsert market_data (delete by security + date, then insert)."""
    as_of_dates = market_data["as_of_date"].unique().tolist()
    security_ids = market_data["security_id"].unique().tolist()

    def _upsert(data_list: list[dict[str, Any]]) -> None:
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
    """Fetch all records from reference.index_constituents."""
    _parse_date(start_date)  # validate inputs
    _parse_date(end_date)
    return _read_table(orm_session, orm_engine, IndexConstituents)


def write_index_constituents(index_constituents: DataFrame, orm_session: Session) -> None:
    """Upsert index_constituents (delete by index_id, then insert)."""
    index_ids = index_constituents["index_id"].unique().tolist()

    def _upsert(data_list: list[dict[str, Any]]) -> None:
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

    If metric_type is provided, filters by that type and renames
    metric_value to the snake_case metric name.
    """
    query = orm_session.query(*SecurityFundamentals.__table__.columns)
    if metric_type:
        query = query.filter(SecurityFundamentals.metric_type == metric_type)

    df = pd.read_sql_query(query.statement, con=orm_engine)

    if metric_type and "metric_value" in df.columns:
        df.rename(columns={"metric_value": _camel_to_snake(metric_type)}, inplace=True)

    return df


def write_security_fundamentals(fundamental_data: DataFrame, orm_session: Session) -> None:
    """
    Write security fundamentals with upsert/versioning logic.

    - New data     → insert.
    - Same count   → overwrite (delete + insert).
    - Diff count   → close existing records (set end_date) + insert new ones.
    """
    if fundamental_data.empty:
        print("No fundamental data to insert.")
        return

    source_vendor = fundamental_data["source_vendor"].iloc[0]
    effective_date = fundamental_data["effective_date"].iloc[0]
    security_ids = fundamental_data["security_id"].unique().tolist()
    metric_types = fundamental_data["metric_type"].unique().tolist()

    active_record_filters = [
        SecurityFundamentals.security_id.in_(security_ids),
        SecurityFundamentals.metric_type.in_(metric_types),
        SecurityFundamentals.source_vendor == source_vendor,
        SecurityFundamentals.effective_date == effective_date,
        SecurityFundamentals.end_date.is_(None),
    ]

    existing_count = orm_session.query(SecurityFundamentals).filter(*active_record_filters).count()
    new_count = len(fundamental_data)

    data_list = fundamental_data.to_dict(orient="records")
    for record in data_list:
        record["end_date"] = None

    def _write(data_list: list[dict[str, Any]]) -> None:
        if existing_count == 0:
            print(f"No existing records found. Inserting {new_count} new records.")
            _bulk_insert(orm_session, SecurityFundamentals, data_list)

        elif existing_count == new_count:
            print(f"Record counts match ({existing_count}). Overwriting existing records.")
            orm_session.execute(delete(SecurityFundamentals).where(*active_record_filters))
            _bulk_insert(orm_session, SecurityFundamentals, data_list)

        else:
            print(f"Record counts differ (existing: {existing_count}, new: {new_count}). Versioning data.")
            orm_session.execute(update(SecurityFundamentals).where(*active_record_filters).values(end_date=date.today()))
            _bulk_insert(orm_session, SecurityFundamentals, data_list)

        print("Security fundamentals data successfully written.")

    _execute_with_session(orm_session, _write, data_list)


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
    Join portfolio, holdings, security master, market data, and fundamentals.

    Returns a merged DataFrame with point-in-time fundamentals aligned to
    each market date via a backward merge_asof.
    """
    df_securities = read_security_master(orm_session, orm_engine)
    df_market_data = read_market_data(orm_session, orm_engine, start_date, end_date)
    df_portfolio = read_portfolio(orm_session, orm_engine, portfolio_short_names)
    df_holdings = read_portfolio_holdings(orm_session, orm_engine, start_date, end_date)
    df_fundamentals = read_security_fundamentals(orm_session, orm_engine, "shares_outstanding")

    df_portfolio_market_data = (
        df_portfolio.merge(df_holdings, on="port_id")
        .merge(df_securities, on="security_id")
        .merge(df_market_data, on=["security_id", "as_of_date"])
    )

    df_portfolio_market_data["as_of_date"] = pd.to_datetime(df_portfolio_market_data["as_of_date"], errors="coerce")
    df_fundamentals["effective_date"] = pd.to_datetime(df_fundamentals["effective_date"], errors="coerce")
    df_portfolio_market_data.dropna(subset=["as_of_date"], inplace=True)
    df_fundamentals.dropna(subset=["effective_date"], inplace=True)

    df_portfolio_market_data["security_id"] = df_portfolio_market_data["security_id"].astype(int)
    df_fundamentals["security_id"] = df_fundamentals["security_id"].astype(int)

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

from .schema_analytics import (  # noqa: E402
    Attribution,
    FactorScores,
    PortfolioReturns,
)
from .schema_fx import (  # noqa: E402
    FactorExposures,
    FxRates,
    RiskSnapshots,
)


def write_factor_scores(df: DataFrame, orm_session: Session) -> None:
    """Upsert factor scores. One row per (date, security, factor, universe)."""
    if df.empty:
        print("No factor scores to write.")
        return
    df = df.copy()
    df["upsert_date"] = datetime.datetime.now()
    as_of = df["as_of_date"].unique().tolist()
    secs = df["security_id"].unique().tolist()

    def _upsert(data_list: list[dict[str, Any]]) -> None:
        _bulk_delete_insert(
            orm_session,
            FactorScores,
            data_list,
            FactorScores.as_of_date.in_(as_of),
            FactorScores.security_id.in_(secs),
        )

    _execute_with_session(orm_session, _upsert, df.to_dict(orient="records"))


def read_factor_scores(
    orm_session: Session, orm_engine: Engine, as_of_date: Optional[date] = None, factor_name: Optional[str] = None
) -> DataFrame:
    """Read factor scores, optionally filtered by date / factor."""
    q = orm_session.query(*FactorScores.__table__.columns)
    if as_of_date:
        q = q.filter(FactorScores.as_of_date == as_of_date)
    if factor_name:
        q = q.filter(FactorScores.factor_name == factor_name)
    return pd.read_sql_query(q.statement, con=orm_engine)


def write_portfolio_returns(df: DataFrame, orm_session: Session) -> None:
    """Upsert daily portfolio returns."""
    if df.empty:
        print("No portfolio returns to write.")
        return
    df = df.copy()
    df["upsert_date"] = datetime.datetime.now()
    as_of = df["as_of_date"].unique().tolist()
    ports = df["port_id"].unique().tolist()

    def _upsert(data_list: list[dict[str, Any]]) -> None:
        _bulk_delete_insert(
            orm_session,
            PortfolioReturns,
            data_list,
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
    df["upsert_date"] = datetime.datetime.now()
    as_of = df["as_of_date"].unique().tolist()
    ports = df["port_id"].unique().tolist()

    def _upsert(data_list: list[dict[str, Any]]) -> None:
        _bulk_delete_insert(
            orm_session,
            Attribution,
            data_list,
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
    df["upsert_date"] = datetime.datetime.now()
    _execute_with_session(
        orm_session,
        lambda data: _bulk_delete_insert(
            orm_session,
            FxRates,
            data,
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
    df["upsert_date"] = datetime.datetime.now()
    _execute_with_session(
        orm_session,
        lambda data: _bulk_delete_insert(
            orm_session,
            RiskSnapshots,
            data,
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
    df["upsert_date"] = datetime.datetime.now()
    _execute_with_session(
        orm_session,
        lambda data: _bulk_delete_insert(
            orm_session,
            FactorExposures,
            data,
            FactorExposures.as_of_date.in_(df["as_of_date"].unique().tolist()),
            FactorExposures.port_id.in_(df["port_id"].unique().tolist()),
        ),
        df.to_dict(orient="records"),
    )


def compute_and_store_factors(
    prices: DataFrame, factors: dict[str, Any], universe: str, source_vendor: str, orm_session: Session
) -> None:
    """Compute factor scores for a price panel and persist to factor_scores."""
    from analytics.factors import Factor

    rows = []
    today = datetime.datetime.now()
    for name, fac in factors.items():
        scores = fac.compute(prices) if isinstance(fac, Factor) else fac(prices)
        if scores is None:
            continue
        rank = scores.rank(axis=1, pct=True)
        for dt, row in scores.iterrows():
            for sec, val in row.items():
                if pd.isna(val):
                    continue
                rows.append(
                    {
                        "as_of_date": pd.to_datetime(dt).date(),
                        "security_id": int(sec),
                        "factor_name": name,
                        "factor_value": float(val),
                        "rank_pct": float(rank.loc[dt, sec]) if not pd.isna(rank.loc[dt, sec]) else None,
                        "universe": universe,
                        "source_vendor": source_vendor,
                        "upsert_date": today,
                    }
                )
    if rows:
        write_factor_scores(pd.DataFrame(rows), orm_session)
