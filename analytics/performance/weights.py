"""Portfolio constituent weight calculations.

Pure functions only. Preserves the legacy arithmetic exactly so existing
backtests and unit tests (which assert values to 6 decimals) keep passing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas import DataFrame


def calculate_portfolio_constituent_weights(
    portfolio_market_data: DataFrame,
    price_type: str,
    weightage_type: str = "equal_weighted",
    portfolio_specific_weights: Optional[Dict[str, Union[str, List[str]]]] = None,
    max_weight: Optional[float] = None,
) -> DataFrame:
    """Calculate asset weights in each portfolio for a given weightage method.

    Args:
        portfolio_market_data: DataFrame with columns including ``as_of_date``,
            ``port_id``, ``security_id``, ``portfolio_short_name``,
            ``portfolio_type``, ``shares_outstanding`` and ``price_type``.
        price_type: Price column name (e.g. ``"close"``, ``"adj_close"``).
        weightage_type: Default method: ``equal_weighted`` / ``market_weighted`` /
            ``price_weighted``.
        portfolio_specific_weights: Optional mapping of ``portfolio_short_name``
            (or list of names) to a specific weightage type.

    Returns:
        Pivot DataFrame indexed by ``as_of_date`` with a
        (portfolio_short_name, security_id) columns MultiIndex of weights.

        max_weight: Optional cap (0 < x <= 1) applied to market-weighted
            portfolios. Any constituent exceeding it is clipped to ``max_weight``
            and the remaining names are renormalized to sum to 1. This guards
            against a single mispriced security (e.g. a bad price feed value
            inflating one name's market cap to 40%+ of the index) silently
            dominating the reconstruction. Disabled when None.
    """
    portfolio_specific_weights = portfolio_specific_weights or {}
    df = portfolio_market_data.copy()

    # ---------------------------------------------------
    # 1. Compute market_value & held shares (for benchmark index reconstruction)
    # ---------------------------------------------------
    # Benchmark market_value uses the PRIOR-DAY price AND prior-day float shares,
    # i.e. the true prior-close market cap: shares[t-1] * P[t-1]. Index returns are
    # w_{t-1} * r_t, and the caller (notebook) shifts constituent returns by 1
    # (log_returns.shift(1)), so the weights must be anchored to t-1 to align with
    # return r_t. Using same-day shares (or same-day price) here mis-anchors the
    # cap weight and accumulates a systematic tracking error vs the real index
    # (and the first date, with no prior row, yields 0 -> 0/0, which is guarded).
    px_shift = df[price_type].groupby([df["port_id"], df["security_id"]], group_keys=False).shift(1)

    # `shares_outstanding` only reaches this frame when the loader actually
    # joined fundamentals for the requested portfolio securities (it is dropped
    # by a later `dropna(axis=1, how="all")` otherwise). Guard against its
    # absence so a missing fundamentals column degrades to an equal-weight
    # proxy instead of raising KeyError mid-backtest.
    has_shares = "shares_outstanding" in df.columns
    if "market_cap" in df.columns:
        df.loc[df["portfolio_type"] == "Benchmark", "market_value"] = df["market_cap"].shift(1)
    elif has_shares:
        sh_shift = df["shares_outstanding"].groupby([df["port_id"], df["security_id"]], group_keys=False).shift(1)
        df.loc[df["portfolio_type"] == "Benchmark", "market_value"] = sh_shift * px_shift
    else:
        # No shares-outstanding available: use held_shares as the market-value
        # proxy so benchmark reconstruction still runs (effectively equal-ish).
        df.loc[df["portfolio_type"] == "Benchmark", "market_value"] = df.get("held_shares", pd.Series(1.0, index=df.index))

    if "held_shares" not in df.columns:
        df["held_shares"] = pd.NA

    benchmark_mask = df["portfolio_type"] == "Benchmark"
    benchmark_totals = (
        df["market_value"].where(benchmark_mask).groupby([df["as_of_date"], df["portfolio_short_name"]]).transform("sum")
    )
    df.loc[benchmark_mask, "held_shares"] = (
        df.loc[benchmark_mask, "market_value"] / benchmark_totals.loc[benchmark_mask].replace(0, np.nan)
    ).fillna(0.0)

    # ---------------------------------------------------
    # 2. Pivot held_shares and prices
    # ---------------------------------------------------
    # Pivot each value separately. A constituent that is HELD but has no price for
    # the window (e.g. a de-listed leaver still in portfolio_holdings, or a
    # constituent whose EOD pull is missing) yields an all-NA column in the price
    # pivot but a real column in held_shares. Pivoting both together lets
    # pivot_table drop that column from BOTH, but they can otherwise diverge, so we
    # reindex prices onto held_shares' columns and backfill 0 (no price -> 0
    # weight) to keep the two frames column-aligned downstream.
    held_pivot = (
        df.pivot_table(
            index="as_of_date",
            columns=["portfolio_short_name", "security_id"],
            values="held_shares",
        )
        .fillna(0)
        .sort_index(axis=1, level=[0, 1])
    )
    price_pivot = (
        df.pivot_table(
            index="as_of_date",
            columns=["portfolio_short_name", "security_id"],
            values=price_type,
        )
        .fillna(0)
        .sort_index(axis=1, level=[0, 1])
    )

    held_shares = held_pivot
    prices = price_pivot.reindex(columns=held_shares.columns, fill_value=0.0)

    weights = pd.DataFrame(
        index=held_shares.index,
        columns=held_shares.columns,
        dtype=float,
    ).fillna(0.0)

    # ---------------------------------------------------
    # 3. Process each portfolio independently
    # ---------------------------------------------------
    for portfolio in df["portfolio_short_name"].unique():
        weight_method = portfolio_specific_weights.get(portfolio, weightage_type)
        if isinstance(weight_method, list):
            weight_method = weight_method[0]

        portfolio_cols = [c for c in held_shares.columns if c[0] == portfolio]
        if not portfolio_cols:
            continue

        shares = held_shares[portfolio_cols]
        px = prices[portfolio_cols]

        # A portfolio reconstructed as a Benchmark (e.g. the S&P 500 from .SPX
        # constituents) already carries its float-adjusted market-cap weights in
        # `held_shares`, now computed from the PRIOR-day price so it aligns with
        # the notebook's log_returns.shift(1) (-> w_{t-1} * r_t). Use them directly.
        is_bench = bool((df.loc[df["portfolio_short_name"] == portfolio, "portfolio_type"] == "Benchmark").all())

        if weight_method == "equal_weighted":
            active = (shares != 0).astype(int)
            counts = active.sum(axis=1)
            portfolio_weights = active.div(counts, axis=0).fillna(0)

        elif weight_method == "market_weighted":
            if is_bench:
                portfolio_weights = shares
            else:
                market_value = shares * px.shift(1)
                totals = market_value.sum(axis=1).replace(0, np.nan)
                portfolio_weights = market_value.div(totals, axis=0).fillna(0)
            # Optional hard cap: clip any name above max_weight, then renormalize
            # the survivors to 1 so the book stays fully invested. Applied to both
            # benchmark (held_shares) and standard market-weight paths.
            if max_weight is not None and 0 < max_weight <= 1:
                w = portfolio_weights.clip(upper=max_weight)
                row_sum = w.sum(axis=1).replace(0, np.nan)
                portfolio_weights = w.div(row_sum, axis=0).fillna(0)

        elif weight_method == "price_weighted":
            totals = px.sum(axis=1)
            portfolio_weights = px.div(totals, axis=0).fillna(0)

        else:
            raise ValueError(f"Unsupported weightage_type '{weight_method}' for portfolio '{portfolio}'")

        weights[portfolio_cols] = portfolio_weights

    return weights.sort_index(axis=1, level=[0, 1])


def calculate_held_shares(group_df: DataFrame) -> DataFrame:
    """Compute held shares (benchmark weights) for a single group.

    For rows where ``portfolio_type == 'Benchmark'``, sets
    ``held_shares = market_value / total_market_value``; otherwise leaves
    ``held_shares`` unchanged.
    """
    df = group_df.copy()
    if (df["portfolio_type"] == "Benchmark").all():
        total_market_value = df["market_value"].sum()
        if total_market_value == 0:
            df["held_shares"] = 0.0  # Prevent division by zero
        else:
            df["held_shares"] = df["market_value"] / total_market_value
    else:
        df["held_shares"] = df.get("held_shares", pd.NA)
    return df
