"""Shared analytics-layer ORM models (factor_scores, portfolio_returns, attribution).

One schema definition, used by BOTH backends (Azure SQL via ``database.py`` and
Databricks via ``databricks.py``), so the two engines stay in lock-step. This is
the "Analytics" layer of the README's persistent-analytics design: daily
time-series fact tables that downstream tools (Riskfolio-Lib, ARCH, Dash) read
directly.

Grain
-----
factor_scores   : one row per (as_of_date, security_id, factor_name, universe)
portfolio_returns: one row per (as_of_date, port_id)
attribution     : one row per (as_of_date, port_id, dimension, key)

All timestamps use ``date`` (not string) to avoid the divergence that exists
between the two legacy backends.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy import Date, Engine, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class AnalyticsBase(DeclarativeBase):
    """Base for analytics-layer tables (separate from each backend's Base)."""

    pass


class FactorScores(AnalyticsBase):
    """Daily cross-sectional factor scores per security."""

    __tablename__ = "factor_scores"
    __table_args__ = (
        UniqueConstraint(
            "as_of_date",
            "security_id",
            "factor_name",
            "universe",
            "source_vendor",
            name="uq_factor_scores",
        ),
        {"schema": "analytics"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    as_of_date: Mapped[date] = mapped_column(Date)
    security_id: Mapped[int] = mapped_column(Integer)
    factor_name: Mapped[str] = mapped_column(String(64))
    factor_value: Mapped[float] = mapped_column(Float)
    rank_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    universe: Mapped[str] = mapped_column(String(64), default="ALL")
    source_vendor: Mapped[str] = mapped_column(String(64))
    upsert_date: Mapped[date] = mapped_column(Date)


class PortfolioReturns(AnalyticsBase):
    """Daily portfolio-level returns vs benchmark."""

    __tablename__ = "portfolio_returns"
    __table_args__ = (
        UniqueConstraint("as_of_date", "port_id", name="uq_portfolio_returns"),
        {"schema": "analytics"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    as_of_date: Mapped[date] = mapped_column(Date)
    port_id: Mapped[int] = mapped_column(Integer)
    gross_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    net_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    benchmark_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    source_vendor: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    upsert_date: Mapped[date] = mapped_column(Date)


class Attribution(AnalyticsBase):
    """Daily performance attribution by dimension (sector / region / factor / ...)."""

    __tablename__ = "attribution"
    __table_args__ = (
        UniqueConstraint(
            "as_of_date",
            "port_id",
            "dimension",
            "key",
            "benchmark_key",
            name="uq_attribution",
        ),
        {"schema": "analytics"},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    as_of_date: Mapped[date] = mapped_column(Date)
    port_id: Mapped[int] = mapped_column(Integer)
    dimension: Mapped[str] = mapped_column(String(32))  # 'sector' | 'region' | 'factor' | ...
    key: Mapped[str] = mapped_column(String(64))  # portfolio segment key
    benchmark_key: Mapped[str] = mapped_column(String(64))  # matching benchmark segment
    contribution: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    benchmark_contribution: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    upsert_date: Mapped[date] = mapped_column(Date)


def create_analytics_tables(engine: Engine) -> None:
    """Create the analytics tables if they do not exist (idempotent)."""
    AnalyticsBase.metadata.create_all(engine)
