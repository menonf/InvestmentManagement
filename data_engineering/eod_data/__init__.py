"""EOD data package: unified vendor access for multi-vendor backtesting.

Use :func:`get_vendor` to obtain a price vendor by name at runtime:

    from data_engineering.eod_data import get_vendor
    vendor = get_vendor("yahoo")            # live Yahoo
    vendor = get_vendor("refinitiv")        # live LSEG/Refinitiv
    df, no_data = vendor.fetch(symbols, start, end)

Each vendor normalizes to the same schema (see ``base.STANDARD_COLUMNS``), so
the engine can swap vendors without downstream changes.
"""

from __future__ import annotations

from typing import Any

from .base import STANDARD_COLUMNS, PriceVendor
from .marketstack import MarketstackVendor
from .refinitiv import RefinitivVendor
from .tiingo import TiingoVendor
from .yahoo import YahooVendor

_REGISTRY: dict[str, type[PriceVendor]] = {
    "yahoo": YahooVendor,
    "refinitiv": RefinitivVendor,
    "tiingo": TiingoVendor,
    "marketstack": MarketstackVendor,
}


def register_vendor(key: str, vendor_cls: type[PriceVendor]) -> None:
    """Register an additional vendor class (e.g. a custom backtest simulator)."""
    _REGISTRY[key.lower()] = vendor_cls


def get_vendor(name: str, **kwargs: Any) -> PriceVendor:
    """Return a configured :class:`PriceVendor` instance for ``name``.

    For credentialed vendors (tiingo, marketstack) pass the token/key via
    ``kwargs`` or rely on ``*Vendor.from_env()`` inside the constructor.
    """
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown vendor '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


__all__ = [
    "get_vendor",
    "register_vendor",
    "PriceVendor",
    "STANDARD_COLUMNS",
    "YahooVendor",
    "RefinitivVendor",
    "TiingoVendor",
    "MarketstackVendor",
]
