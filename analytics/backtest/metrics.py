"""Performance and signal-quality metrics (pure pandas / numpy)."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

TRADING_DAYS = 252
Returns = Union[Series, DataFrame]


def cumulative_returns(returns: Returns) -> Returns:
    """Compound simple returns: ``(1 + r).cumprod() - 1``."""
    return (1.0 + returns.fillna(0.0)).cumprod() - 1.0


def annualized_return(returns: Returns, periods_per_year: int = TRADING_DAYS) -> Union[float, Series]:
    """Geometric annualised return (CAGR)."""
    r = returns.dropna() if isinstance(returns, Series) else returns.dropna(how="all")
    n = len(r)
    if n == 0:
        return np.nan if isinstance(returns, Series) else Series(np.nan, index=returns.columns)
    total = (1.0 + r.fillna(0.0)).prod()
    return total ** (periods_per_year / n) - 1.0


def annualized_volatility(returns: Returns, periods_per_year: int = TRADING_DAYS) -> Union[float, Series]:
    """Annualised standard deviation of returns."""
    return returns.std(ddof=1) * np.sqrt(periods_per_year)


def sharpe_ratio(returns: Returns, risk_free: float = 0.0, periods_per_year: int = TRADING_DAYS) -> Union[float, Series]:
    """Annualised Sharpe ratio with a constant annual risk-free rate."""
    excess = returns - risk_free / periods_per_year
    return excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year)


def sortino_ratio(returns: Returns, periods_per_year: int = TRADING_DAYS) -> Union[float, Series]:
    """Annualised mean return over downside deviation."""
    downside = returns.where(returns < 0, 0.0)
    dd = np.sqrt((downside**2).mean())
    return returns.mean() / dd * np.sqrt(periods_per_year)


def max_drawdown(returns: Returns) -> Union[float, Series]:
    """Worst peak-to-trough decline of the compounded return path (negative number)."""
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return drawdown.min()


def calmar_ratio(returns: Returns, periods_per_year: int = TRADING_DAYS) -> Union[float, Series]:
    """CAGR divided by the absolute maximum drawdown."""
    mdd = max_drawdown(returns)
    return annualized_return(returns, periods_per_year) / abs(mdd)


def hit_rate(returns: Returns) -> Union[float, Series]:
    """Share of periods with a positive return."""
    return (returns > 0).sum() / returns.notna().sum()


def turnover(weights: DataFrame) -> Series:
    """One-way turnover per date: ``0.5 * sum(|w_t - w_{t-1}|)`` (the first date starts from cash)."""
    w = weights.fillna(0.0)
    return 0.5 * (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)


def summary_table(returns: DataFrame, periods_per_year: int = TRADING_DAYS, benchmark: Optional[str] = None) -> DataFrame:
    """One row per column of ``returns`` with the standard performance statistics.

    Args:
        returns: date x portfolio simple returns.
        benchmark: optional column name; adds beta, correlation and tracking
            error relative to it.
    """
    rows = {}
    for col in returns.columns:
        r = returns[col].dropna()
        rows[col] = {
            "total_return": float((1.0 + r).prod() - 1.0),
            "cagr": float(annualized_return(r, periods_per_year)),
            "volatility": float(annualized_volatility(r, periods_per_year)),
            "sharpe": float(sharpe_ratio(r, periods_per_year=periods_per_year)),
            "sortino": float(sortino_ratio(r, periods_per_year)),
            "max_drawdown": float(max_drawdown(r)),
            "calmar": float(calmar_ratio(r, periods_per_year)),
            "hit_rate": float(hit_rate(r)),
            "periods": int(len(r)),
        }
        if benchmark is not None and benchmark in returns.columns and col != benchmark:
            b = returns[benchmark].reindex(r.index)
            both = pd.concat([r, b], axis=1).dropna()
            cov = both.cov()
            rows[col]["beta"] = float(cov.iloc[0, 1] / cov.iloc[1, 1]) if cov.iloc[1, 1] else np.nan
            rows[col]["correlation"] = float(both.corr().iloc[0, 1])
            rows[col]["tracking_error"] = float((both.iloc[:, 0] - both.iloc[:, 1]).std(ddof=1) * np.sqrt(periods_per_year))
    return DataFrame.from_dict(rows, orient="index")


# ---------------------------------------------------------------------------
# Signal quality
# ---------------------------------------------------------------------------


def forward_returns(prices: DataFrame, horizon: int) -> DataFrame:
    """Return from t to t+``horizon`` rows, aligned to t."""
    return prices.shift(-horizon) / prices - 1.0


def information_coefficient(signal: DataFrame, fwd_returns: DataFrame, method: str = "spearman") -> Series:
    """Per-date correlation between the signal and subsequent returns.

    ``method="spearman"`` (default) rank-transforms both cross-sections first
    (computed as Pearson on ranks, so no scipy dependency); ``"pearson"`` uses
    raw values. Dates with fewer than three paired observations are NaN.
    """
    sig, fwd = signal.align(fwd_returns, join="inner")
    both_valid = sig.notna() & fwd.notna()
    s = sig.where(both_valid)
    f = fwd.where(both_valid)
    if method == "spearman":
        s = s.rank(axis=1)
        f = f.rank(axis=1)
    elif method != "pearson":
        raise ValueError(f"unknown method '{method}'")
    n = both_valid.sum(axis=1)
    s_c = s.sub(s.mean(axis=1), axis=0)
    f_c = f.sub(f.mean(axis=1), axis=0)
    cov = (s_c * f_c).sum(axis=1)
    denom = np.sqrt((s_c**2).sum(axis=1) * (f_c**2).sum(axis=1))
    ic = cov / denom.replace(0, np.nan)
    return ic.where(n > 2)


def ic_summary(ic: Series, periods_per_year: int = 12) -> dict[str, float]:
    """Mean IC, its t-stat, the IC information ratio and the share of positive periods."""
    ic = ic.dropna()
    if ic.empty:
        return {"mean_ic": np.nan, "t_stat": np.nan, "ic_ir": np.nan, "pct_positive": np.nan, "n": 0}
    mean, std = float(ic.mean()), float(ic.std(ddof=1))
    return {
        "mean_ic": mean,
        "t_stat": float(mean / std * np.sqrt(len(ic))) if std else np.nan,
        "ic_ir": float(mean / std * np.sqrt(periods_per_year)) if std else np.nan,
        "pct_positive": float((ic > 0).mean()),
        "n": int(len(ic)),
    }


def quantile_returns(signal: DataFrame, fwd_returns: DataFrame, n_quantiles: int = 5) -> DataFrame:
    """Mean forward return per signal quantile (1 = lowest signal ... n = highest) per date."""
    sig, fwd = signal.align(fwd_returns, join="inner")
    out = {}
    for dt in sig.index:
        pair = pd.concat([sig.loc[dt].rename("s"), fwd.loc[dt].rename("r")], axis=1).dropna()
        if len(pair) < n_quantiles:
            continue
        q = pd.qcut(pair["s"].rank(method="first"), n_quantiles, labels=range(1, n_quantiles + 1))
        out[dt] = pair.groupby(q, observed=True)["r"].mean()
    return DataFrame(out).T.sort_index()
