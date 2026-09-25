"""Matplotlib plotting helpers for performance reporting."""

from __future__ import annotations

from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import DataFrame, Series


def _ensure_datetime_index(frame: DataFrame) -> DataFrame:
    if isinstance(frame.index, pd.DatetimeIndex):
        return frame
    out = frame.copy()
    out.index = pd.to_datetime(out.index)
    return out


def plot_cumulative_returns(
    asset_returns: DataFrame,
    start_at_zero: bool = True,
    title: str = "Cumulative Returns Over Time",
    save_path: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> Figure:
    """Plot compounded cumulative returns (%) for each column.

    Accepts a per-period simple-return frame (compounded as ``(1 + r).cumprod()
    - 1``) or an already-cumulative level series (detected when the largest
    absolute value exceeds 0.5, i.e. far beyond a single period's return).

    Returns the Figure (display it with ``display(fig)`` in a notebook). Pass
    ``save_path`` to also write a PNG; nothing is written by default.
    """
    frame = _ensure_datetime_index(asset_returns)
    if ax is not None:
        fig = plt.gcf()
        axis = ax
    else:
        fig = plt.figure(figsize=(10, 6))
        axis = fig.add_subplot(111)
    for column in frame.columns:
        series = frame[column].astype(float)
        cumulative = series if series.abs().max() > 0.5 else (1.0 + series.fillna(0.0)).cumprod() - 1.0
        if start_at_zero:
            cumulative = cumulative - cumulative.iloc[0]
        axis.plot(frame.index, cumulative * 100, label=str(column), linewidth=2)

    axis.set_title(title, fontsize=14)
    axis.set_ylabel("Cumulative Returns (%)", fontsize=12)
    axis.axhline(y=0, color="black", linestyle="--", linewidth=1)
    axis.legend(loc="upper left", fontsize=9)
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axis.tick_params(axis="x", labelrotation=45, labelsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return fig


def plot_returns(asset_returns: DataFrame, title: str = "Periodic Returns Over Time", save_path: Optional[str] = None) -> Figure:
    """Plot periodic (non-cumulative) returns (%) for each column."""
    frame = _ensure_datetime_index(asset_returns)
    fig = plt.figure(figsize=(10, 6))
    axis = fig.add_subplot(111)
    for column in frame.columns:
        axis.plot(frame.index, frame[column] * 100, label=str(column), linewidth=1.5)
    axis.set_title(title, fontsize=14)
    axis.set_ylabel("Returns (%)", fontsize=12)
    axis.axhline(y=0, color="black", linestyle="--", linewidth=1)
    axis.legend(loc="upper left", fontsize=9)
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return fig


def plot_drawdowns(returns: DataFrame, title: str = "Drawdown", save_path: Optional[str] = None) -> Figure:
    """Plot the drawdown path (%) of each column's compounded returns."""
    frame = _ensure_datetime_index(returns)
    wealth = (1.0 + frame.fillna(0.0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    fig = plt.figure(figsize=(10, 4))
    axis = fig.add_subplot(111)
    for column in drawdown.columns:
        axis.plot(drawdown.index, drawdown[column] * 100, label=str(column), linewidth=1.5)
    axis.set_title(title, fontsize=14)
    axis.set_ylabel("Drawdown (%)", fontsize=12)
    axis.legend(loc="lower left", fontsize=9)
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return fig


def plot_score_distribution(screen: DataFrame, title: str = "Composite score distribution") -> Figure:
    """Histogram of the blended ``composite`` score, coloured by leg (LONG / SHORT)."""
    if "composite" not in screen.columns:
        raise KeyError("screen needs a 'composite' column to plot")
    fig = plt.figure(figsize=(9, 4.5))
    axis = fig.add_subplot(111)
    side = (
        screen.get("side", Series([""] * len(screen), index=screen.index))
        if "side" in screen.columns
        else Series([""] * len(screen), index=screen.index)
    )
    colors = {"LONG": "#2ca02c", "SHORT": "#d62728", "": "#888888"}
    for label, color in colors.items():
        mask = side == label
        if mask.any():
            axis.hist(screen.loc[mask, "composite"].dropna(), bins=30, alpha=0.6, color=color, label=label or "unselected")
    axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    axis.set_title(title, fontsize=14)
    axis.set_xlabel("composite score", fontsize=12)
    axis.set_ylabel("count", fontsize=12)
    axis.legend(fontsize=9)
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    return fig


def plot_screen_scatter(screen: DataFrame, x: str, y: str, title: str = "Signal scatter") -> Figure:
    """Scatter of two cross-sectional z-scores, coloured by leg, annotated by ticker."""
    if x not in screen.columns or y not in screen.columns:
        raise KeyError(f"screen needs '{x}' and '{y}' columns to plot")
    fig = plt.figure(figsize=(8, 8))
    axis = fig.add_subplot(111)
    side = screen["side"] if "side" in screen.columns else Series([""] * len(screen), index=screen.index)
    colors = {"LONG": "#2ca02c", "SHORT": "#d62728", "": "#888888"}
    for label, color in colors.items():
        mask = side == label
        if mask.any():
            axis.scatter(screen.loc[mask, x], screen.loc[mask, y], s=28, alpha=0.7, color=color, label=label or "unselected")
    ticks = screen.get("ticker")
    if ticks is not None:
        for idx in screen.index:
            if pd.notna(ticks.loc[idx]):
                axis.annotate(str(ticks.loc[idx]), (screen.loc[idx, x], screen.loc[idx, y]), fontsize=6, alpha=0.6)
    axis.axhline(0.0, color="black", linewidth=0.6, linestyle="--")
    axis.axvline(0.0, color="black", linewidth=0.6, linestyle="--")
    axis.set_xlabel(x, fontsize=12)
    axis.set_ylabel(y, fontsize=12)
    axis.set_title(title, fontsize=14)
    axis.legend(fontsize=9, loc="upper left")
    axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    fig.tight_layout()
    return fig
