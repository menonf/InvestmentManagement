"""Signal -> portfolio weights (quantile long/short and long-only books)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from pandas import DataFrame


def quantile_weights(
    signal: DataFrame,
    quantile: float = 0.1,
    long_only: bool = False,
    membership: Optional[DataFrame] = None,
    min_names: int = 4,
    max_weight: Optional[float] = None,
) -> DataFrame:
    """Equal-weight the top ``quantile`` of the signal (long) and bottom (short) on each date.

    Args:
        signal: date x security_id scores; higher = more attractive. NaN =
            not eligible on that date.
        quantile: fraction of eligible names in each leg (0.1 = top/bottom decile).
        long_only: build only the long leg (weights sum to +1).
        membership: optional boolean date x security_id frame restricting
            eligibility to point-in-time universe members (avoids trading a
            name before it joined / after it left the index).
        min_names: dates with fewer eligible names get zero weights.
        max_weight: optional per-name cap; excess is redistributed pro-rata.

    Returns:
        date x security_id weights: long leg sums to +1, short leg to -1
        (dollar-neutral), zeros elsewhere.
    """
    if not 0 < quantile <= 0.5:
        raise ValueError("quantile must be in (0, 0.5]")
    sig = signal.astype(float)
    if membership is not None:
        sig = sig.where(membership.reindex(index=sig.index, columns=sig.columns).fillna(False).astype(bool))

    n_valid = sig.notna().sum(axis=1)
    n_leg = np.maximum(1, np.ceil(quantile * n_valid)).astype(int)
    n_leg = n_leg.where(n_valid >= min_names, 0)

    # rank 1 = highest score; rank from bottom for the short leg
    rank_desc = sig.rank(axis=1, ascending=False, method="first")
    rank_asc = sig.rank(axis=1, ascending=True, method="first")
    long_mask = rank_desc.le(n_leg, axis=0) & sig.notna()
    short_mask = rank_asc.le(n_leg, axis=0) & sig.notna() & ~long_mask

    weights = DataFrame(0.0, index=sig.index, columns=sig.columns)
    long_n = long_mask.sum(axis=1).replace(0, np.nan)
    weights = weights.mask(long_mask, 1.0)
    weights = weights.div(long_n, axis=0).where(long_mask, 0.0)
    if not long_only:
        short_n = short_mask.sum(axis=1).replace(0, np.nan)
        short_w = (
            DataFrame(0.0, index=sig.index, columns=sig.columns).mask(short_mask, -1.0).div(short_n, axis=0).where(short_mask, 0.0)
        )
        weights = weights + short_w
    weights = weights.fillna(0.0)
    if max_weight is not None:
        weights = cap_weights(weights, max_weight)
    return weights


def cap_weights(weights: DataFrame, max_weight: float, max_iter: int = 50) -> DataFrame:
    """Cap absolute weights at ``max_weight`` per leg, redistributing the excess pro-rata.

    Capping is iterative (water-filling): after each clip the shortfall to the
    leg's original gross is spread over the names still below the cap, until
    no name exceeds the cap or every name in the leg sits at the cap (in which
    case the leg's gross is necessarily reduced).
    """
    out = weights.copy().fillna(0.0)
    for sign in (1.0, -1.0):
        leg = out.where(np.sign(out) == sign, 0.0).abs()
        target = leg.sum(axis=1)
        for _ in range(max_iter):
            over = leg > max_weight + 1e-12
            if not over.to_numpy().any():
                break
            leg = leg.clip(upper=max_weight)
            free = (leg > 0) & (leg < max_weight - 1e-12)
            shortfall = target - leg.sum(axis=1)
            free_sum = leg.where(free, 0.0).sum(axis=1).replace(0, np.nan)
            add = leg.where(free, 0.0).div(free_sum, axis=0).mul(shortfall, axis=0).fillna(0.0)
            leg = leg + add
        out = out.where(np.sign(out) != sign, leg * sign)
    return out


def leg_returns(weights: DataFrame, asset_returns: DataFrame) -> DataFrame:
    """Long-leg, short-leg and net returns per date from weights already aligned to returns."""
    r = asset_returns.reindex(index=weights.index, columns=weights.columns)
    long_r = (weights.clip(lower=0) * r).sum(axis=1)
    short_r = (weights.clip(upper=0) * r).sum(axis=1)
    return DataFrame({"long": long_r, "short": short_r, "net": long_r + short_r})
