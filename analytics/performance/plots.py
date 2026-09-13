"""Plotting helpers for performance analytics (matplotlib-based)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame


def plot_cumulative_returns(asset_returns: DataFrame, start_at_zero: bool = True) -> "plt.Figure":
    """Plot cumulative returns for each column.

    Accepts EITHER a daily-return matrix (each column a per-period simple return)
    OR an already-cumulative level series. Daily returns are compounded properly
    as ``(1 + r).cumprod() - 1`` -- NOT ``exp(cumsum(r)) - 1``, which understates
    the true cumulative return (it omits the ``+1`` offset inside the product). The
    earlier exp-based formula made a correctly-built reconstruction plot ~1-2% below
    the ETF it was tracking, even when the underlying returns matched.

    Returns the matplotlib Figure so it can be displayed inline in a notebook
    (e.g. assign the result to a variable and let it be the cell's last
    expression, or pass it to IPython.display.display). A PNG is also written
    for headless execution.
    """
    import matplotlib.dates as mdates

    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        asset_returns = asset_returns.copy()
        asset_returns.index = pd.to_datetime(asset_returns.index)

    # Detect whether the input is already cumulative (last value far from a single
    # period's return) vs a daily-return stream, so callers can pass either form.
    fig, ax = plt.subplots(figsize=(10, 6))
    for column in asset_returns.columns:
        series = asset_returns[column].astype(float)
        # A per-period return series has typical abs values << 1; a cumulative level
        # series ends well above that. Treat |max| > 0.5 as already cumulative.
        if series.abs().max() > 0.5:
            cumulative_returns = series
        else:
            cumulative_returns = (1.0 + series).cumprod() - 1.0
        if start_at_zero:
            cumulative_returns = cumulative_returns - cumulative_returns.iloc[0]
        cumulative_returns *= 100
        ax.plot(asset_returns.index, cumulative_returns, label=str(column), linewidth=2)

    ax.set_title("Cumulative Returns Over Time", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Cumulative Returns (%)", fontsize=12)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", which="major", labelrotation=45, labelsize=9)
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    fig.tight_layout()
    fig.autofmt_xdate()

    # Headless-safe: save to PNG instead of blocking on plt.show()/mplcursors.
    # Guard the write so a cwd/permission failure never breaks the inline figure
    # (the returned fig still renders). Fall back to a temp path if the relative
    # write fails.
    try:
        fig.savefig("cumulative_returns.png", dpi=120)
    except OSError:
        try:
            import os
            import tempfile

            fig.savefig(os.path.join(tempfile.gettempdir(), "cumulative_returns.png"), dpi=120)
        except Exception:
            pass
    plt.close(fig)
    return fig


def plot_returns(asset_returns: DataFrame) -> None:
    """Plot periodic (non-cumulative) returns for each column."""
    if not isinstance(asset_returns.index, pd.DatetimeIndex):
        asset_returns = asset_returns.copy()
        asset_returns.index = pd.to_datetime(asset_returns.index)

    fig, ax = plt.subplots(figsize=(10, 6))
    for column in asset_returns.columns:
        returns_pct = asset_returns[column] * 100
        ax.plot(asset_returns.index, returns_pct, label=str(column), linewidth=1.5)

    ax.set_title("Periodic Returns Over Time", fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("Returns (%)", fontsize=12)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.xaxis.set_visible(False)
    fig.tight_layout()

    # Headless-safe: save to PNG instead of blocking on plt.show()/mplcursors.
    fig.savefig("periodic_returns.png", dpi=120)
    plt.close(fig)
