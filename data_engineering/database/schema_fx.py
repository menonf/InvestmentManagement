"""Shared FX / risk / factor-exposure ORM models.

One schema definition reused by both backends (Azure SQL via ``database.py`` and
Databricks via ``databricks.py``). Covers the remaining analytics-layer tables
from docs/schema_recommendations.md: fx_rates (T0.1), risk_snapshots (T2.3),
factor_exposures (T1.2).

All timestamps use ``date`` (not string) to stay consistent with the analytics
layer and avoid the legacy string/date divergence between backends.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy import Date, Engine, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class FxBase(DeclarativeBase):
    """Base for FX/risk/factor-exposure analytics tables."""

    pass


class FxRates(FxBase):
    """Daily currency exchange rates (base -> quote)."""

    __tablename__ = "fx_rates"
    __table_args__ = (
        UniqueConstraint(
            "from_currency",
            "to_currency",
            "as_of_date",
            "source_vendor",
            name="uq_fx_rates",
        ),
        {"schema": "analytics"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    from_currency: Mapped[str] = mapped_column(String(8))
    to_currency: Mapped[str] = mapped_column(String(8))
    as_of_date: Mapped[date] = mapped_column(Date)
    rate: Mapped[float] = mapped_column(Float)
    source_vendor: Mapped[str] = mapped_column(String(64))
    upsert_date: Mapped[date] = mapped_column(Date)


class RiskSnapshots(FxBase):
    """Daily risk metric snapshots per portfolio (trended over time)."""

    __tablename__ = "risk_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "as_of_date",
            "port_id",
            "metric",
            "source_vendor",
            name="uq_risk_snapshots",
        ),
        {"schema": "analytics"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    as_of_date: Mapped[date] = mapped_column(Date)
    port_id: Mapped[int] = mapped_column(Integer)
    metric: Mapped[str] = mapped_column(String(32))  # VaR | CVaR | volatility | beta | ...
    value: Mapped[float] = mapped_column(Float)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lookback_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    source_vendor: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    upsert_date: Mapped[date] = mapped_column(Date)


class FactorExposures(FxBase):
    """Daily portfolio-level factor exposure (weights x scores)."""

    __tablename__ = "factor_exposures"
    __table_args__ = (
        UniqueConstraint(
            "as_of_date",
            "port_id",
            "factor_name",
            "source_vendor",
            name="uq_factor_exposures",
        ),
        {"schema": "analytics"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    as_of_date: Mapped[date] = mapped_column(Date)
    port_id: Mapped[int] = mapped_column(Integer)
    factor_name: Mapped[str] = mapped_column(String(64))
    exposure: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    benchmark_exposure: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    active_exposure: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source_vendor: Mapped[str] = mapped_column(String(64))
    upsert_date: Mapped[date] = mapped_column(Date)


def create_fx_tables(engine: Engine) -> None:
    """Create the FX/risk/exposure tables if they do not exist (idempotent)."""
    FxBase.metadata.create_all(engine)
