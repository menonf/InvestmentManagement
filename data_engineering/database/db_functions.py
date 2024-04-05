"""SQLAlchemy ORM Module to connect to database objects."""
import datetime

import pandas as pd
from pandas import DataFrame
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy Base Class."""

    pass


class SecurityMaster(Base):
    """SQLAlchemy ORM Class maps on to SQL SecurityMaster table."""

    __tablename__ = "SecurityMaster"
    __table_args__ = {"schema": "dbo"}
    SecID: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    Ticker: Mapped[str] = mapped_column()
    Sector: Mapped[str] = mapped_column()
    Country: Mapped[str] = mapped_column()
    SecurityType: Mapped[str] = mapped_column()
    SecurityName: Mapped[str] = mapped_column()


class MarketData(Base):
    """SQLAlchemy ORM Class maps on to SQL MarketData table."""

    __tablename__ = "MarketData"
    __table_args__ = {"schema": "dbo"}
    MD_ID: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    AsOfDate: Mapped[str] = mapped_column()
    SecID: Mapped[int] = mapped_column()
    Open: Mapped[float] = mapped_column()
    High: Mapped[float] = mapped_column()
    Low: Mapped[float] = mapped_column()
    Close: Mapped[float] = mapped_column()
    Volume: Mapped[int] = mapped_column()
    Dividends: Mapped[float] = mapped_column()
    Stock_Splits: Mapped[float] = mapped_column()
    Dataload_Date: Mapped[str] = mapped_column()


class Portfolio(Base):
    """SQLAlchemy ORM Class maps on to SQL Portfolio table."""

    __tablename__ = "Portfolio"
    __table_args__ = {"schema": "dbo"}
    PortID: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    PortfolioShortName: Mapped[str] = mapped_column()
    PortfolioName: Mapped[str] = mapped_column()


class PortfolioHoldings(Base):
    """SQLAlchemy ORM Class maps on to SQL PortfolioHoldings table."""

    __tablename__ = "PortfolioHoldings"
    __table_args__ = {"schema": "dbo"}
    HID: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    AsOfDate: Mapped[str] = mapped_column()
    PortID: Mapped[int] = mapped_column()
    SecID: Mapped[int] = mapped_column()
    HeldShares: Mapped[float] = mapped_column()


def read_security_master(orm_session: Session, orm_engine: Engine) -> DataFrame:
    """Fetch records from Security Master table.

    Params:
        orm_session: SQLAlchemy Session object.
        orm_engine: SQLAlchemy database engine object.

    Returns:
        DataFrame containing Security Master table records.

    """
    query = orm_session.query(
        SecurityMaster.SecID,
        SecurityMaster.Ticker,
        SecurityMaster.Sector,
        SecurityMaster.Country,
        SecurityMaster.SecurityType,
        SecurityMaster.SecurityName,
    )

    df_securityMaster = pd.read_sql_query(query.statement, con=orm_engine)

    return df_securityMaster


def read_portfolio(orm_session: Session, orm_engine: Engine, portfolio_short_names: list[str]) -> DataFrame:
    """Fetch records from Portfolio table.

    Params:
        orm_session: SQLAlchemy Session object.
        orm_engine: SQLAlchemy database engine object.

    Returns:
        DataFrame containing Portfolio table records.

    """
    query = orm_session.query(Portfolio.PortID, Portfolio.PortfolioShortName, Portfolio.PortfolioName).filter(
        Portfolio.PortfolioShortName.in_(portfolio_short_names)
    )
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
        PortfolioHoldings.AsOfDate,
        PortfolioHoldings.PortID,
        PortfolioHoldings.SecID,
        PortfolioHoldings.HeldShares,
    ).filter(PortfolioHoldings.AsOfDate.between(start_date, end_date))

    df_portfolio_holdings = pd.read_sql_query(query.statement, con=orm_engine)

    return df_portfolio_holdings


def read_market_data(orm_session: Session, orm_engine: Engine, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch records from MarketData table.

    Params:
        orm_session: SQLAlchemy Session object.
        orm_engine: SQLAlchemy database engine object.

    Returns:
        DataFrame containing MarketData table records.

    """
    # Parse input dates to datetime objects
    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")

    # Query only the necessary columns and filter by date range
    query = orm_session.query(
        MarketData.AsOfDate,
        MarketData.SecID,
        MarketData.Open,
        MarketData.High,
        MarketData.Low,
        MarketData.Close,
        MarketData.Volume,
        MarketData.Dividends,
        MarketData.Stock_Splits,
    ).filter(MarketData.AsOfDate.between(start_date, end_date))

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
        as_of_dates = market_data["AsOfDate"].unique().tolist()

        orm_session.query(MarketData).filter(MarketData.AsOfDate.in_(as_of_dates)).delete(synchronize_session=False)
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

    df_portfolio_market_data = pd.merge(
        pd.merge(pd.merge(df_portfolio_data, df_portfolio_holdings_data), df_securities),
        df_market_data,
    )

    return df_portfolio_market_data
