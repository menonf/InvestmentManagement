"""SQLAlchemy ORM Module to connect to database objects."""

import datetime
import re
import time
from typing import Optional, Tuple
from urllib import parse

import keyring
import pandas as pd
import sqlalchemy as sql
from pandas import DataFrame
from sqlalchemy import Engine, delete
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy Base Class."""

    pass


class SecurityMaster(Base):
    """SQLAlchemy ORM Class maps on to security_master table."""

    __tablename__ = "security_master"
    __table_args__ = {"schema": "dbo"}
    security_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column()
    name: Mapped[str] = mapped_column()
    isin: Mapped[str] = mapped_column()
    sedol: Mapped[str] = mapped_column()
    cusip: Mapped[str] = mapped_column()
    figi: Mapped[str] = mapped_column()
    loanxid: Mapped[str] = mapped_column()
    country: Mapped[str] = mapped_column()
    currency: Mapped[str] = mapped_column()
    sector: Mapped[str] = mapped_column()
    industry_group: Mapped[str] = mapped_column()
    industry: Mapped[str] = mapped_column()
    security_type: Mapped[str] = mapped_column()
    asset_class: Mapped[str] = mapped_column()
    exchange: Mapped[str] = mapped_column()
    is_active: Mapped[str] = mapped_column()
    source_vendor: Mapped[str] = mapped_column()
    upsert_date: Mapped[str] = mapped_column()
    upsert_by: Mapped[str] = mapped_column()


class SecurityFundamentals(Base):
    """SQLAlchemy ORM class mapping to security_fundamentals table."""

    __tablename__ = "security_fundamentals"
    __table_args__ = {"schema": "dbo"}
    __mapper_args__ = {"primary_key": ["security_id", "metric_type", "effective_date", "source_vendor"]}
    security_id: Mapped[int] = mapped_column()
    metric_type: Mapped[str] = mapped_column()
    metric_value: Mapped[float] = mapped_column()
    source_vendor: Mapped[str] = mapped_column()
    effective_date: Mapped[datetime.date] = mapped_column()
    end_date: Mapped[datetime.date] = mapped_column(nullable=True)


class MarketData(Base):
    """SQLAlchemy ORM Class maps on to market_data table."""

    __tablename__ = "market_data"
    __table_args__ = {"schema": "dbo"}
    md_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    as_of_date: Mapped[str] = mapped_column()
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


class Portfolio(Base):
    """SQLAlchemy ORM Class maps on to portfolio table."""

    __tablename__ = "portfolio"
    __table_args__ = {"schema": "dbo"}
    port_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    portfolio_short_name: Mapped[str] = mapped_column()
    portfolio_name: Mapped[str] = mapped_column()
    portfolio_type: Mapped[str] = mapped_column()
    is_active: Mapped[str] = mapped_column()


class PortfolioHoldings(Base):
    """SQLAlchemy ORM Class maps on to portfolio_holdings table."""

    __tablename__ = "portfolio_holdings"
    __table_args__ = {"schema": "dbo"}
    ph_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    as_of_date: Mapped[str] = mapped_column()
    port_id: Mapped[int] = mapped_column()
    security_id: Mapped[int] = mapped_column()
    held_shares: Mapped[float] = mapped_column()
    upsert_date: Mapped[str] = mapped_column()
    upsert_by: Mapped[str] = mapped_column()


def get_db_connection(
    service_name: str = "ihub_sql_connection",
    server: str = "ops-store-server.database.windows.net",
    driver: str = "ODBC Driver 18 for SQL Server",
    max_retries: int = 3,
    retry_interval_minutes: int = 2,
) -> Tuple[sql.Engine, sql.Connection, Session]:
    """
    Establish a connection to SQL Server with retry logic.

    Returns:
        (engine, connection, session): SQLAlchemy engine, raw connection, and ORM session
    """
    db = keyring.get_password(service_name, "db")
    db_user = keyring.get_password(service_name, "uid")
    db_password = keyring.get_password(service_name, "pwd")

    connection_string = (
        f"Driver={{{driver}}};"
        f"Server=tcp:{server},1433;"
        f"Database={db};Uid={db_user};Pwd={db_password};"
        "Encrypt=yes;TrustServerCertificate=no;"
    )
    connection_params = parse.quote_plus(connection_string)

    for attempt in range(1, max_retries + 1):
        try:
            engine = sql.create_engine(f"mssql+pyodbc:///?odbc_connect={connection_params}")
            connection = engine.connect()
            session = Session(engine)
            print("Database connection successful.")
            return engine, connection, session
        except OperationalError as e:
            print(f"Attempt {attempt} failed with error:\n{e}")
            if attempt < max_retries:
                print(f"Retrying in {retry_interval_minutes} minutes...")
                time.sleep(retry_interval_minutes * 60)
            else:
                print("All retry attempts failed. Exiting.")
                raise

    raise RuntimeError("Database connection failed: maximum retries exceeded")


def read_security_master(orm_session: Session, orm_engine: Engine) -> DataFrame:
    """Fetch all records from security_master table.

    Params:
        orm_session: SQLAlchemy Session object.
        orm_engine: SQLAlchemy database engine object.

    Returns:
        DataFrame containing Security Master table records.

    """
    query = orm_session.query(SecurityMaster)
    df_securityMaster = pd.read_sql_query(query.statement, con=orm_engine)

    return df_securityMaster


def write_security_master(equities_df: pd.DataFrame, session: Session) -> None:
    """Write records to security_master table.

    Params:
        orm_session: SQLAlchemy Session object.
        orm_engine: SQLAlchemy database engine object.

    Returns:
        DataFrame containing security_master table records.

    """
    try:
        data_list = equities_df.to_dict(orient="records")
        session.bulk_insert_mappings(SecurityMaster, data_list)  # type: ignore
        session.commit()
    except SQLAlchemyError as e:
        print(f"An error occurred: {str(e)}")
        session.rollback()
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        session.rollback()
    finally:
        session.close()


def read_portfolio(orm_session: Session, orm_engine: Engine, portfolio_short_names: list[str]) -> DataFrame:
    """Fetch records from portfolio table.

    Params:
        orm_session: SQLAlchemy Session object.
        orm_engine: SQLAlchemy database engine object.

    Returns:
        DataFrame containing portfolio table records.

    """
    query = orm_session.query(
        Portfolio.port_id, Portfolio.portfolio_short_name, Portfolio.portfolio_name, Portfolio.portfolio_type
    ).filter(Portfolio.portfolio_short_name.in_(portfolio_short_names))
    df_portfolio = pd.read_sql_query(query.statement, con=orm_engine)
    return df_portfolio


def read_portfolio_holdings(orm_session: Session, orm_engine: Engine, start_date: str, end_date: str) -> DataFrame:
    """Fetch records from portfolio_holdings table.

    Params:
        orm_session: SQLAlchemy Session object.
        orm_engine: SQLAlchemy database engine object.

    Returns:
        DataFrame containing portfolio_holdings table records.

    """
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")

    query = orm_session.query(
        PortfolioHoldings.as_of_date,
        PortfolioHoldings.port_id,
        PortfolioHoldings.security_id,
        PortfolioHoldings.held_shares,
    ).filter(PortfolioHoldings.as_of_date.between(start_date, end_date))

    df_portfolio_holdings = pd.read_sql_query(query.statement, con=orm_engine)

    return df_portfolio_holdings


def read_market_data(orm_session: Session, orm_engine: Engine, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch records from market_data table.

    Params:
        orm_session: SQLAlchemy Session object.
        orm_engine: SQLAlchemy database engine object.
        start_date: Start date as a string in "YYYY-MM-DD" format.
        end_date: End date as a string in "YYYY-MM-DD" format.

    Returns:
        DataFrame containing market_data table records.
    """
    # Parse input dates to datetime objects
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")

    # Query all columns dynamically
    query = orm_session.query(*MarketData.__table__.columns).filter(MarketData.as_of_date.between(start_date, end_date))

    df_market_data = pd.read_sql_query(query.statement, con=orm_engine)

    return df_market_data


def write_market_data(market_data: DataFrame, orm_session: Session) -> None:
    """Write records to market_data table.

    Params:
        orm_session: SQLAlchemy Session object.
        orm_engine: SQLAlchemy database engine object.

    Returns:
        DataFrame containing market_data table records.

    """
    try:
        data_list = market_data.to_dict(orient="records")
        as_of_dates = market_data["as_of_date"].unique().tolist()

        orm_session.query(MarketData).filter(MarketData.as_of_date.in_(as_of_dates)).delete(synchronize_session=False)
        orm_session.bulk_insert_mappings(MarketData, data_list)  # type: ignore
        orm_session.commit()

    except SQLAlchemyError as e:
        print(f"An error occurred: {str(e)}")
        orm_session.rollback()
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        orm_session.rollback()
    finally:
        orm_session.close()


def read_security_fundamentals(orm_session: Session, orm_engine: Engine, metric_type: Optional[str] = None) -> pd.DataFrame:
    """Fetch records from security_fundamentals table with optional filtering on metric_type.

    Params:
        orm_session: SQLAlchemy Session object.
        orm_engine: SQLAlchemy database engine object.
        metric_type: Optional string to filter by metric_type. If None, fetch all records.

    Returns:
        DataFrame containing security_fundamentals table records.
    """
    # Build query
    query = orm_session.query(*SecurityFundamentals.__table__.columns)

    # Apply filter if metric_type is provided
    if metric_type:
        query = query.filter(SecurityFundamentals.metric_type == metric_type)

    df_security_fundamentals = pd.read_sql_query(query.statement, con=orm_engine)
    # Rename metric_value column if metric_type is provided
    if metric_type and "metric_value" in df_security_fundamentals.columns:
        df_security_fundamentals.rename(
            columns={"metric_value": re.sub("([A-Z])", r"_\1", metric_type).lower().lstrip("_")}, inplace=True
        )

    return df_security_fundamentals


def write_security_fundamentals(fundamental_data: pd.DataFrame, orm_session: Session) -> None:
    """Write records to the security_fundamentals table.

    Params:
        fundamental_data: DataFrame containing security fundamentals.
        orm_session: SQLAlchemy Session object.
    """
    try:
        if fundamental_data.empty:
            print("No fundamental data to insert.")
            return

        # Remove previous latest records for the same security_id and metric_type
        security_ids = fundamental_data["security_id"].unique().tolist()
        metric_types = fundamental_data["metric_type"].unique().tolist()

        orm_session.execute(
            delete(SecurityFundamentals)
            .where(SecurityFundamentals.security_id.in_(security_ids))
            .where(SecurityFundamentals.metric_type.in_(metric_types))
            .where(SecurityFundamentals.end_date.is_(None))  # Remove latest records
        )

        # Insert new records
        data_list = fundamental_data.to_dict(orient="records")
        orm_session.bulk_insert_mappings(SecurityFundamentals, data_list)  # type: ignore
        orm_session.commit()

    except SQLAlchemyError as e:
        print(f"Database error: {e}")
        orm_session.rollback()
    except Exception as e:
        print(f"Unexpected error: {e}")
        orm_session.rollback()
    finally:
        orm_session.close()


def write_portfolio_holdings(df_holdings: pd.DataFrame, orm_session: Session) -> None:
    """
    Write records to portfolio_holdings table.

    Params:
        df_holdings: DataFrame containing portfolio holdings data.
        orm_session: SQLAlchemy Session object.
    """
    try:
        if df_holdings.empty:
            print("No data to write.")
            return

        as_of_date = df_holdings["as_of_date"].iloc[0]
        port_ids = df_holdings["port_id"].unique().tolist()

        # Delete existing records for same portfolio and date
        orm_session.query(PortfolioHoldings).filter(
            PortfolioHoldings.as_of_date == as_of_date, PortfolioHoldings.port_id.in_(port_ids)
        ).delete(synchronize_session=False)

        df_holdings["upsert_date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df_holdings["upsert_by"] = "daily_portfolio_load.py"

        data_list = df_holdings.to_dict(orient="records")
        orm_session.bulk_insert_mappings(PortfolioHoldings, data_list)  # type: ignore
        orm_session.commit()

    except SQLAlchemyError as e:
        print(f"An error occurred: {str(e)}")
        orm_session.rollback()
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        orm_session.rollback()
    finally:
        orm_session.close()


def get_portfolio_market_data(
    orm_session: Session,
    orm_engine: Engine,
    start_date: str,
    end_date: str,
    portfolio_short_names: list[str],
) -> DataFrame:
    """
    Join all the SQLAlchemy Portfolio Data objects to return Portfolio Market Data.

    Params:
        orm_session: SQLAlchemy Session object.
        orm_engine: SQLAlchemy database engine object.

    Returns:
        DataFrame containing Portfolio Market Data.
    """
    # Load individual tables
    df_securities = read_security_master(orm_session, orm_engine)
    df_market_data = read_market_data(orm_session, orm_engine, start_date, end_date)
    df_portfolio_data = read_portfolio(orm_session, orm_engine, portfolio_short_names)
    df_portfolio_holdings_data = read_portfolio_holdings(orm_session, orm_engine, start_date, end_date)
    df_security_fundamentals = read_security_fundamentals(orm_session, orm_engine, "shares_outstanding")

    # Merge base market data
    df_portfolio_market_data = pd.merge(
        pd.merge(pd.merge(df_portfolio_data, df_portfolio_holdings_data), df_securities), df_market_data
    )

    # Ensure datetime format
    df_portfolio_market_data["as_of_date"] = pd.to_datetime(df_portfolio_market_data["as_of_date"], errors="coerce")
    df_security_fundamentals["effective_date"] = pd.to_datetime(df_security_fundamentals["effective_date"], errors="coerce")

    # Drop rows with missing dates
    df_portfolio_market_data = df_portfolio_market_data.dropna(subset=["as_of_date"])
    df_security_fundamentals = df_security_fundamentals.dropna(subset=["effective_date"])

    # Ensure matching dtypes for security_id
    df_portfolio_market_data["security_id"] = df_portfolio_market_data["security_id"].astype(int)
    df_security_fundamentals["security_id"] = df_security_fundamentals["security_id"].astype(int)

    # Prepare for merge_asof using group-wise merge
    merged_rows = []

    for sec_id in df_portfolio_market_data["security_id"].unique():
        df_left = df_portfolio_market_data[df_portfolio_market_data["security_id"] == sec_id].copy()
        df_left = df_left.sort_values("as_of_date").reset_index(drop=True)

        df_right = df_security_fundamentals[df_security_fundamentals["security_id"] == sec_id].copy()
        df_right = df_right.sort_values("effective_date").reset_index(drop=True)

        if not df_right.empty:
            df_merged = pd.merge_asof(
                df_left, df_right, by="security_id", left_on="as_of_date", right_on="effective_date", direction="backward"
            )
        else:
            # Fill with missing columns if fundamental data is absent
            missing_cols = set(df_security_fundamentals.columns) - set(df_left.columns)
            for col in missing_cols:
                df_left[col] = pd.NA
            df_merged = df_left

        merged_rows.append(df_merged)

    df_merged_all = pd.concat(merged_rows, ignore_index=True)

    return df_merged_all
