"""Factor library for the multi-vendor backtesting engine.

    from analytics.factors import MomentumFactor, ValueFactor, MLReturnFactor, CompositeFactor

Factors consume the standardised price panel (date x security_id) and/or a
fundamentals ratio panel (security_id x RATIO_COLUMNS) produced by
``data_engineering``; they never talk to a vendor, so the same factor runs on
Refinitiv, Yahoo or Tiingo data unchanged.

Modules:
    base        Factor / PanelFactor contracts
    transforms  zscore, rank, winsorize, neutralize, combine, sample_on
    momentum    MomentumFactor (12-1, optional volatility scaling)
    value       ValueFactor (mean z-score of inverted price multiples)
    ml_value    MLReturnFactor (scikit-learn regressors on the 18 ratios)
    ml_training build_modelling_table / time_based_split / train_models
    composite   CompositeFactor (weighted blend on a rebalance schedule)
"""

from __future__ import annotations

from typing import Any

from .base import Factor, PanelFactor
from .composite import CompositeFactor
from .ml_training import models_available
from .ml_value import DEFAULT_ENSEMBLE, MODEL_KEYS, MODELS_DIR, MLReturnFactor, build_estimator
from .momentum import MomentumFactor
from .transforms import combine, fill_neutral, neutralize, rank_pct, sample_on, winsorize, zscore
from .value import ValueFactor

registry: dict[str, type[Factor]] = {
    "momentum": MomentumFactor,
    "value": ValueFactor,
    "ml_return": MLReturnFactor,
    "composite": CompositeFactor,
}


def get_factor(name: str, **kwargs: Any) -> Factor:
    """Instantiate a registered factor by name."""
    key = name.lower()
    if key not in registry:
        raise KeyError(f"Unknown factor '{name}'. Available: {sorted(registry)}")
    return registry[key](**kwargs)


__all__ = [
    "Factor",
    "PanelFactor",
    "MomentumFactor",
    "ValueFactor",
    "MLReturnFactor",
    "CompositeFactor",
    "MODEL_KEYS",
    "DEFAULT_ENSEMBLE",
    "MODELS_DIR",
    "build_estimator",
    "models_available",
    "zscore",
    "rank_pct",
    "winsorize",
    "neutralize",
    "fill_neutral",
    "combine",
    "sample_on",
    "registry",
    "get_factor",
]
