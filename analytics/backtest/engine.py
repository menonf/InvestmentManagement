"""Turn weights and asset returns into portfolio returns.

Conventions
-----------
``weights.loc[t]`` is the portfolio held *going into* date ``t``'s return when
``shift_weights=0``, i.e. the caller has already anchored weights at t-1 (this
is how :func:`analytics.portfolio.weights.calculate_portfolio_constituent_weights`
builds benchmark weights). When weights are derived from a signal that uses
the close of ``t`` (a momentum score, a rank), pass ``shift_weights=1`` so the
position is only applied to the *next* day's return - otherwise the backtest
peeks one day ahead on every rebalance date.
"""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd
from pandas import DataFrame, Series

from .result import BacktestResult


def portfolio_returns(
    weights: DataFrame, asset_returns: DataFrame, shift_weights: int = 1, cost_bps: float = 0.0
) -> tuple[Series, Series, Series]:
    """Daily portfolio return for one weight frame.

    Args:
        weights: date x security_id target weights.
        asset_returns: date x security_id simple returns (``pct_change`` of prices).
        shift_weights: lag applied to weights before multiplying with returns.
        cost_bps: one-way transaction cost in basis points charged on turnover.

    Returns:
        ``(net_return, gross_return, cost)`` Series.
    """
    w = weights.reindex(index=asset_returns.index, columns=asset_returns.columns).fillna(0.0)
    held = w.shift(shift_weights).fillna(0.0) if shift_weights else w
    r = asset_returns.fillna(0.0)
    gross = (held * r).sum(axis=1)
    previous = held.shift(1).fillna(0.0)  # start from cash
    one_way_turnover = 0.5 * (held - previous).abs().sum(axis=1)
    cost = one_way_turnover * cost_bps / 1e4
    return gross - cost, gross, cost


def run_backtest(
    weights: Union[DataFrame, dict[str, DataFrame]],
    asset_returns: DataFrame,
    shift_weights: int = 1,
    cost_bps: float = 0.0,
    name: str = "strategy",
) -> BacktestResult:
    """Backtest one or several weight frames against the same asset returns.

    Args:
        weights: a single date x security frame, or ``{portfolio_name: frame}``.
        asset_returns: date x security simple returns.
        shift_weights: see module docstring (1 for signal-driven books).
        cost_bps: one-way cost in bps per unit of turnover.
        name: portfolio name when a single frame is given.
    """
    frames = weights if isinstance(weights, dict) else {name: weights}
    net, gross, cost = {}, {}, {}
    for pname, w in frames.items():
        net[pname], gross[pname], cost[pname] = portfolio_returns(w, asset_returns, shift_weights, cost_bps)
    return BacktestResult(returns=DataFrame(net), weights=dict(frames), gross_returns=DataFrame(gross), costs=DataFrame(cost))


def portfolio_returns_from_constituents(constituent_returns: DataFrame, constituent_weights: DataFrame) -> DataFrame:
    """Aggregate (portfolio, security) MultiIndex returns x weights into per-portfolio returns.

    This is the reconstruction path used with
    :func:`analytics.portfolio.returns.calculate_portfolio_constituent_returns`
    and :func:`analytics.portfolio.weights.calculate_portfolio_constituent_weights`,
    whose weights are already anchored at t-1 (no additional shift is applied).
    """
    weights = constituent_weights.reindex(index=constituent_returns.index, columns=constituent_returns.columns, fill_value=0.0)
    contrib = constituent_returns.fillna(0.0) * weights
    return contrib.T.groupby(level=0).sum().T


def reconstruction_backtest(constituent_returns: DataFrame, constituent_weights: DataFrame) -> BacktestResult:
    """Wrap :func:`portfolio_returns_from_constituents` in a :class:`BacktestResult`."""
    return BacktestResult(returns=portfolio_returns_from_constituents(constituent_returns, constituent_weights))


def prices_to_returns(prices: DataFrame) -> DataFrame:
    """Simple daily returns from a wide price panel (first row NaN)."""
    return prices.sort_index().pct_change(fill_method=None)


def align_frames(*frames: DataFrame) -> list[DataFrame]:
    """Reindex all frames onto their common dates and securities."""
    idx = frames[0].index
    cols = frames[0].columns
    for f in frames[1:]:
        idx = idx.intersection(f.index)
        cols = cols.intersection(f.columns)
    return [f.reindex(index=idx, columns=cols) for f in frames]


def to_datetime_index(frame: DataFrame) -> DataFrame:
    """Return a copy of ``frame`` with a ``DatetimeIndex`` (dates stored as ``date`` or str)."""
    out = frame.copy()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()
