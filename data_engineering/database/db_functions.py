"""SQLAlchemy ORM Module to connect to database objects."""

import datetime
import re
from typing import Optional

import pandas as pd
from pandas import DataFrame
from sqlalchemy import Engine, delete
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy Base Class."""

    pass


class SecurityMaster(Base):
    """SQLAlchemy ORM Class maps on to SQL SecurityMaster table."""

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
    """SQLAlchemy ORM Class maps on to SQL MarketData table."""

    __tablename__ = "market_data"
    __table_args__ = {"schema": "dbo"}
    md_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    as_of_date: Mapped[str] = mapped_column()
    security_id: Mapped[int] = mapped_column()
    open: Mapped[float] = mapped_column()
    high: Mapped[float] = mapped_column()
    low: Mapped[float] = mapped_column()
    close: Mapped[float] = mapped_column()
    volume: Mapped[int] = mapped_column()
    dividends: Mapped[float] = mapped_column()
    stock_splits: Mapped[float] = mapped_column()
    interval: Mapped[str] = mapped_column()
    dataload_date: Mapped[str] = mapped_column()


class Portfolio(Base):
    """SQLAlchemy ORM Class maps on to SQL Portfolio table."""

    __tablename__ = "portfolio"
    __table_args__ = {"schema": "dbo"}
    port_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    portfolio_short_name: Mapped[str] = mapped_column()
    portfolio_name: Mapped[str] = mapped_column()
    portfolio_type: Mapped[str] = mapped_column()
    is_active: Mapped[str] = mapped_column()


class PortfolioHoldings(Base):
    """SQLAlchemy ORM Class maps on to SQL PortfolioHoldings table."""

    __tablename__ = "portfolio_holdings"
    __table_args__ = {"schema": "dbo"}
    ph_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    as_of_date: Mapped[str] = mapped_column()
    port_id: Mapped[int] = mapped_column()
    security_id: Mapped[int] = mapped_column()
    held_shares: Mapped[float] = mapped_column()


def read_security_master(orm_session: Session, orm_engine: Engine) -> DataFrame:
    """Fetch all records from Security Master table.

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
    """Fetch records from Portfolio table.

    Params:
        orm_session: SQLAlchemy Session object.
        orm_engine: SQLAlchemy database engine object.

    Returns:
        DataFrame containing Portfolio table records.

    """
    query = orm_session.query(
        Portfolio.port_id, Portfolio.portfolio_short_name, Portfolio.portfolio_name, Portfolio.portfolio_type
    ).filter(Portfolio.portfolio_short_name.in_(portfolio_short_names))
    df_portfolio = pd.read_sql_query(query.statement, con=orm_engine)
    return df_portfolio


def read_portfolio_holdings(orm_session: Session, orm_engine: Engine, start_date: str, end_date: str) -> DataFrame:
    """Fetch records from PortfolioHoldings table.

    Params:
        orm_session: SQLAlchemy Session object.
        orm_engine: SQLAlchemy database engine object.

    Returns:
        DataFrame containing PortfolioHoldings table records.

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
    """Fetch records from MarketData table.

    Params:
        orm_session: SQLAlchemy Session object.
        orm_engine: SQLAlchemy database engine object.
        start_date: Start date as a string in "YYYY-MM-DD" format.
        end_date: End date as a string in "YYYY-MM-DD" format.

    Returns:
        DataFrame containing MarketData table records.
    """
    # Parse input dates to datetime objects
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")

    # Query all columns dynamically
    query = orm_session.query(*MarketData.__table__.columns).filter(MarketData.as_of_date.between(start_date, end_date))

    df_market_data = pd.read_sql_query(query.statement, con=orm_engine)

    return df_market_data


def write_market_data(market_data: DataFrame, orm_session: Session) -> None:
    """Write records to MarketData table.

    Params:
        orm_session: SQLAlchemy Session object.
        orm_engine: SQLAlchemy database engine object.

    Returns:
        DataFrame containing MarketData table records.

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
    """Fetch records from SecurityFundamentals table with optional filtering on metric_type.

    Params:
        orm_session: SQLAlchemy Session object.
        orm_engine: SQLAlchemy database engine object.
        metric_type: Optional string to filter by metric_type. If None, fetch all records.

    Returns:
        DataFrame containing SecurityFundamentals table records.
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


def write_portfolio_holdings(df_holdings: DataFrame, orm_session: Session) -> None:
    """Write records to PortfolioHoldings table.

    Params:
        df_holdings: DataFrame containing portfolio holdings data.
        orm_session: SQLAlchemy Session object.
    """
    try:
        data_list = df_holdings.to_dict(orient="records")
        orm_session.query(PortfolioHoldings)
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
    """Join all the SQLAlchemy Portfolio Data objects to return Portfolio Market Data.

    Params:
        orm_session: SQLAlchemy Session object.
        orm_engine: SQLAlchemy database engine object.

    Returns:
        DataFrame containing Portfolio Market Data table records.

    """
    df_securities = read_security_master(orm_session, orm_engine)
    df_market_data = read_market_data(orm_session, orm_engine, start_date, end_date)
    df_portfolio_data = read_portfolio(orm_session, orm_engine, portfolio_short_names)
    df_portfolio_holdings_data = read_portfolio_holdings(orm_session, orm_engine, start_date, end_date)
    df_security_fundamentals = read_security_fundamentals(orm_session, orm_engine, "sharesOutstanding")

    df_portfolio_market_data = pd.merge(
        pd.merge(pd.merge(df_portfolio_data, df_portfolio_holdings_data), df_securities), df_market_data
    )

    df_portfolio_market_data = pd.merge(df_portfolio_market_data, df_security_fundamentals, on="security_id")

    return df_portfolio_market_data
