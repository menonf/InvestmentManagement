"""SQLAlchemy ORM models for the core (``dbo`` / ``reference``) tables.

Analytics-layer tables live in :mod:`.schema_analytics` and :mod:`.schema_fx`.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from sqlalchemy import Date, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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


__all__ = [
    "Base",
    "SecurityMaster",
    "SecurityVendorXref",
    "SecurityFundamentals",
    "MarketData",
    "Portfolio",
    "PortfolioHoldings",
    "IndexConstituents",
]
