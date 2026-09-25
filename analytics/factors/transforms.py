"""Cross-sectional signal transforms shared by factors and composites.

All functions operate row-wise (one row = one date, columns = securities) on a
wide DataFrame, or on a single cross-section given as a Series.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

Frame = Union[DataFrame, Series]


def zscore(x: Frame, eps: float = 1e-9) -> Frame:
    """Standardise each cross-section to mean 0, std 1 (population std)."""
    if isinstance(x, Series):
        return (x - x.mean()) / (x.std(ddof=0) + eps)
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1, ddof=0) + eps, axis=0)


def rank_pct(x: Frame) -> Frame:
    """Percentile rank each cross-section into ``(0, 1]`` (ties averaged)."""
    if isinstance(x, Series):
        return x.rank(pct=True)
    return x.rank(axis=1, pct=True)


def winsorize(x: Frame, lower: float = 0.01, upper: float = 0.99) -> Frame:
    """Clip each cross-section to its ``[lower, upper]`` quantiles."""
    if isinstance(x, Series):
        lo, hi = x.quantile(lower), x.quantile(upper)
        return x.clip(lower=lo, upper=hi)
    lo = x.quantile(lower, axis=1)
    hi = x.quantile(upper, axis=1)
    return x.clip(lower=lo, upper=hi, axis=0)


def neutralize(scores: DataFrame, groups: Series) -> DataFrame:
    """Demean scores within groups (e.g. GICS sector) on every date.

    Args:
        scores: date x security_id.
        groups: ``security_id -> group label``. Securities without a group are
            demeaned as their own group (left unchanged).
    """
    g = groups.reindex(scores.columns)
    out = scores.copy()
    for label, cols in g.groupby(g, dropna=False).groups.items():
        cols = list(cols)
        out[cols] = scores[cols].sub(scores[cols].mean(axis=1), axis=0)
    return out


def fill_neutral(x: Frame, value: float = 0.0) -> Frame:
    """Replace NaN with a neutral score (0 after z-scoring, 0.5 after ranking)."""
    return x.fillna(value)


def sample_on(signal: DataFrame, dates: pd.DatetimeIndex, forward_fill: bool = True) -> DataFrame:
    """Keep the signal only on ``dates`` and (optionally) carry it forward.

    Used to turn a daily-evolving signal into one that is refreshed on a
    rebalance schedule so portfolios do not churn on day-to-day noise.
    """
    keep = signal.index.isin(dates)
    out = signal.where(pd.Series(keep, index=signal.index), np.nan)
    return out.ffill() if forward_fill else out


def combine(
    signals: dict[str, DataFrame], weights: Optional[dict[str, float]] = None, method: str = "zscore", missing: Optional[float] = 0.0
) -> DataFrame:
    """Blend several signals into one composite.

    Args:
        signals: ``name -> date x security_id`` frames (aligned on union).
        weights: ``name -> weight`` (default equal). Weights are normalised.
        method: ``"zscore"`` (default) or ``"rank"`` per cross-section before blending.
        missing: value used for a component that is NaN for a security on a
            date (0 = neutral under z-scoring; ``None`` propagates NaN).

    Returns:
        Weighted sum of the transformed components.
    """
    if not signals:
        raise ValueError("combine() needs at least one signal")
    names = list(signals)
    w = {n: 1.0 for n in names} if weights is None else dict(weights)
    total = sum(abs(w.get(n, 0.0)) for n in names)
    if total == 0:
        raise ValueError("composite weights sum to zero")
    idx = signals[names[0]].index
    cols = signals[names[0]].columns
    for n in names[1:]:
        idx = idx.union(signals[n].index)
        cols = cols.union(signals[n].columns)
    out = DataFrame(0.0, index=idx, columns=cols)
    for n in names:
        s = signals[n].reindex(index=idx, columns=cols)
        if method == "zscore":
            t = zscore(s)
        elif method == "rank":
            t = rank_pct(s)
            t = t.sub(t.mean(axis=1), axis=0)  # centre so a neutral name scores 0
        else:
            raise ValueError(f"unknown combine method '{method}'")
        if missing is not None:
            t = t.fillna(missing)
        out = out + t * (w.get(n, 0.0) / total)
    return out
