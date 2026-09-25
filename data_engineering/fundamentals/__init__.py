"""Vendor-agnostic fundamentals (ratio) providers.

    from data_engineering.fundamentals import get_fundamentals_provider, RATIO_COLUMNS

    provider = get_fundamentals_provider("static", orm_session=s, orm_engine=e)
    panel = provider.get_panel(symbols, "2025-06-30")   # security_id x 18 ratios

Providers:
    static    - point-in-time reads from ``dbo.security_fundamentals`` (recommended for scoring)
    refinitiv - LSEG annual snapshot or quarterly history
    yahoo     - latest values via yfinance
    memory    - point-in-time panels from a history frame already in memory (tests / CSV research)
    simfin    - stub
"""

from __future__ import annotations

from typing import Any

from .base import FundamentalsProvider
from .database_provider import DatabaseFundamentalsProvider, StaticFundamentalsProvider
from .memory import InMemoryFundamentalsProvider
from .ratios import RATIO_COLUMNS, RAW_ITEMS, compute_ratios, empty_panel
from .refinitiv import REFINITIV_RAW_FIELDS, REFINITIV_REQUEST_FIELDS, RefinitivFundamentalsProvider, compute_refinitiv_ratios
from .simfin import SimFinFundamentalsProvider
from .store import collect_and_store_fundamentals, history_to_long, panel_to_long
from .yahoo import YahooFundamentalsProvider

registry: dict[str, type[FundamentalsProvider]] = {
    "static": StaticFundamentalsProvider,
    "database": StaticFundamentalsProvider,
    "yahoo": YahooFundamentalsProvider,
    "simfin": SimFinFundamentalsProvider,
    "refinitiv": RefinitivFundamentalsProvider,
    "memory": InMemoryFundamentalsProvider,
}


def get_fundamentals_provider(name: str, **kwargs: Any) -> FundamentalsProvider:
    """Instantiate a registered provider by name, passing only kwargs it accepts."""
    key = name.lower()
    if key not in registry:
        raise KeyError(f"Unknown fundamentals provider '{name}'. Available: {sorted(registry)}")
    cls = registry[key]
    import inspect

    accepted = inspect.signature(cls.__init__).parameters
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(**filtered)


__all__ = [
    "RATIO_COLUMNS",
    "RAW_ITEMS",
    "compute_ratios",
    "empty_panel",
    "FundamentalsProvider",
    "StaticFundamentalsProvider",
    "DatabaseFundamentalsProvider",
    "InMemoryFundamentalsProvider",
    "YahooFundamentalsProvider",
    "SimFinFundamentalsProvider",
    "RefinitivFundamentalsProvider",
    "REFINITIV_REQUEST_FIELDS",
    "REFINITIV_RAW_FIELDS",
    "compute_refinitiv_ratios",
    "collect_and_store_fundamentals",
    "panel_to_long",
    "history_to_long",
    "registry",
    "get_fundamentals_provider",
]
