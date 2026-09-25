"""Synthetic price / fundamentals generators shared by the unit tests and the offline demo notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame

from data_engineering.fundamentals.ratios import RATIO_COLUMNS


def synthetic_prices(n_securities: int = 40, n_days: int = 600, seed: int = 0, start: str = "2021-01-01") -> DataFrame:
    """Geometric random-walk prices with a persistent per-security drift (so momentum has something to find)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    drift = rng.normal(0.0003, 0.0006, size=n_securities)
    shocks = rng.normal(0.0, 0.015, size=(n_days, n_securities))
    log_px = np.cumsum(shocks + drift, axis=0)
    prices = DataFrame(100 * np.exp(log_px), index=dates, columns=list(range(1001, 1001 + n_securities)))
    prices.index.name = "as_of_date"
    prices.columns.name = "security_id"
    return prices


def synthetic_fundamentals(security_ids, seed: int = 1) -> DataFrame:
    """One ratio panel with plausible magnitudes (some negative earnings, some NaN)."""
    rng = np.random.default_rng(seed)
    n = len(security_ids)
    panel = DataFrame(index=pd.Index(security_ids, name="security_id"), columns=RATIO_COLUMNS, dtype=float)
    panel["P/E"] = rng.lognormal(np.log(18), 0.5, n)
    panel["P/B"] = rng.lognormal(np.log(3), 0.6, n)
    panel["P/S"] = rng.lognormal(np.log(2.5), 0.7, n)
    panel["EV/EBIT"] = rng.lognormal(np.log(14), 0.5, n)
    panel["RoE"] = rng.normal(0.12, 0.08, n)
    panel["ROCE"] = rng.normal(0.10, 0.06, n)
    panel["Debt/Equity"] = rng.lognormal(np.log(0.6), 0.5, n)
    panel["Debt Ratio"] = rng.uniform(0.1, 0.6, n)
    panel["Gross Profit Margin"] = rng.uniform(0.2, 0.7, n)
    panel["Asset Turnover"] = rng.uniform(0.3, 1.5, n)
    panel["Working Capital Ratio"] = rng.uniform(0.8, 2.5, n)
    panel["(CA-CL)/TA"] = rng.normal(0.1, 0.1, n)
    panel["RE/TA"] = rng.normal(0.25, 0.2, n)
    panel["EBIT/TA"] = rng.normal(0.08, 0.05, n)
    panel["Book Equity/TL"] = rng.lognormal(np.log(0.8), 0.5, n)
    panel["Op. In./Interest Expense"] = rng.lognormal(np.log(8), 0.8, n)
    panel["Cash Ratio"] = rng.uniform(0.1, 1.0, n)
    panel["Op. In./(NWC+FA)"] = rng.normal(0.15, 0.1, n)
    # a few loss-makers and a few gaps
    losers = rng.choice(n, size=max(1, n // 10), replace=False)
    panel.iloc[losers, panel.columns.get_loc("P/E")] = -rng.lognormal(np.log(20), 0.5, len(losers))
    gaps = rng.choice(n, size=max(1, n // 8), replace=False)
    panel.iloc[gaps, panel.columns.get_loc("Cash Ratio")] = np.nan
    return panel


def synthetic_fundamentals_history(security_ids, period_ends, seed: int = 2) -> DataFrame:
    """Long frame ``[security_id, effective_date, *RATIO_COLUMNS]`` with one row per period end."""
    frames = []
    for i, pe in enumerate(period_ends):
        panel = synthetic_fundamentals(security_ids, seed=seed + i).reset_index()
        panel.insert(1, "effective_date", pd.Timestamp(pe))
        frames.append(panel)
    return pd.concat(frames, ignore_index=True)


def long_market_frame(prices: DataFrame, portfolio_short_name: str = "TEST", port_id: int = 1, portfolio_type: str = "Portfolio", held_shares: float = 1.0) -> DataFrame:
    """Reshape a wide price panel into the long (date, portfolio, security) frame the weights/returns helpers consume."""
    long_df = prices.stack(future_stack=True).rename("adj_close").reset_index()
    long_df.columns = ["as_of_date", "security_id", "adj_close"]
    long_df["close"] = long_df["adj_close"]
    long_df["port_id"] = port_id
    long_df["portfolio_short_name"] = portfolio_short_name
    long_df["portfolio_type"] = portfolio_type
    long_df["held_shares"] = held_shares
    return long_df


def synthetic_membership(prices: DataFrame, start: Optional[str] = None, end: Optional[str] = None) -> pd.Series:
    """Point-in-time membership for the offline demo.

    Returns a date-indexed Series of ``set(security_id)`` - the same shape as
    :func:`data_engineering.loaders.portfolios.portfolio_members_on` - with every
    security a member on every priced day.
    """
    idx = prices.index
    if start is not None:
        idx = idx[idx >= pd.Timestamp(start)]
    if end is not None:
        idx = idx[idx <= pd.Timestamp(end)]
    ids = set(int(c) for c in prices.columns)
    return pd.Series([ids] * len(idx), index=idx, name="members")


def synthetic_security_master(security_ids, sectors: Optional[Sequence[str]] = None) -> DataFrame:
    """Minimal security master with the columns the screen enrichment reads.

    Provides ``security_id``, ``name``, ``sector``, ``industry`` and
    ``vendor_ticker`` (synthetic RICs ``SYN<id>.O``).
    """
    ids = list(security_ids)
    sectors = list(sectors or ["Technology", "Financials", "Health Care", "Consumer", "Energy", "Industrials"])
    rng = np.random.default_rng(7)
    rows = []
    for i, sid in enumerate(ids):
        rows.append({
            "security_id": int(sid),
            "name": f"Synthetic {sid}",
            "sector": sectors[i % len(sectors)],
            "industry": f"Sub-{sectors[i % len(sectors)]}",
            "vendor_ticker": f"SYN{int(sid)}.O",
        })
    return DataFrame(rows)


def synthetic_shares_outstanding(security_ids, seed: int = 4) -> pd.Series:
    """Float-adjusted shares outstanding per security (used for market cap)."""
    rng = np.random.default_rng(seed)
    ids = list(security_ids)
    shares = rng.lognormal(np.log(5e8), 0.8, size=len(ids))
    return pd.Series(shares, index=pd.Index([int(i) for i in ids], name="security_id"), name="shares_outstanding")
