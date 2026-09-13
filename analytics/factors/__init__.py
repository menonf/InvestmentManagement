"""Factor library for the multi-vendor backtesting engine.

Import factors here so callers can do:

    from analytics.factors import MomentumFactor, ValueFactor, registry

Factors are vendor-agnostic: they consume the standardized price panel produced
by ``data_engineering.eod_data`` vendors, so the same factor runs on Yahoo,
Refinitiv, Tiingo or Marketstack data without changes.
"""

from __future__ import annotations

from typing import Any

from .factors import Factor, MomentumFactor, ValueFactor
from .ml_factor import MLReturnFactor

# Name -> class registry for config-driven factor selection.
registry: dict[str, type[Factor]] = {
    "momentum": MomentumFactor,
    "value": ValueFactor,
    "ml_return": MLReturnFactor,
}


def get_factor(name: str, **kwargs: Any) -> Factor:
    """Instantiate a registered factor by name."""
    key = name.lower()
    if key not in registry:
        raise KeyError(f"Unknown factor '{name}'. Available: {sorted(registry)}")
    return registry[key](**kwargs)


__all__ = [
    "Factor",
    "MomentumFactor",
    "ValueFactor",
    "MLReturnFactor",
    "registry",
    "get_factor",
]
