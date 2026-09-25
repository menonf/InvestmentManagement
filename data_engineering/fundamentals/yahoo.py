"""Live fundamentals from Yahoo Finance (``yfinance``)."""

from __future__ import annotations

import logging
from typing import Any, Optional

from pandas import DataFrame

from .base import FundamentalsProvider
from .ratios import compute_ratios, empty_panel

log = logging.getLogger(__name__)

#: yfinance ``Ticker.info`` keys -> internal raw item names. Where Yahoo offers
#: several candidate keys the first non-null one wins.
YAHOO_INFO_FIELDS: dict[str, tuple[str, ...]] = {
    "price": ("currentPrice", "regularMarketPrice"),
    "shares": ("sharesOutstanding",),
    "ent_val": ("enterpriseValue",),
    "ebit": ("ebit",),
    "operating_income": ("operatingIncome", "ebit"),
    "revenue": ("totalRevenue",),
    "gross_profit": ("grossProfits",),
    "net_income": ("netIncomeToCommon", "netIncome"),
    "interest_exp": ("interestExpense",),
    "total_assets": ("totalAssets",),
    "total_liab": ("totalLiab", "totalLiabilities"),
    "book_equity": ("stockholdersEquity", "totalStockholderEquity"),
    "total_debt": ("totalDebt",),
    "current_assets": ("totalCurrentAssets",),
    "current_liab": ("totalCurrentLiabilities",),
    "cash": ("totalCash",),
    "fixed_assets": ("propertyPlantEquipment", "netPPE"),
    "retained_earnings": ("retainedEarnings",),
}


def info_to_items(info: dict[str, Any]) -> dict[str, Optional[float]]:
    """Map one ``Ticker.info`` dict onto the internal raw item names."""
    items: dict[str, Optional[float]] = {}
    for item, keys in YAHOO_INFO_FIELDS.items():
        val = next((info.get(k) for k in keys if info.get(k) is not None), None)
        items[item] = float(val) if val is not None else None
    # Yahoo reports book value per share; recover total book equity from it.
    if items.get("book_equity") is None and info.get("bookValue") is not None and items.get("shares"):
        items["book_equity"] = float(info["bookValue"]) * items["shares"]  # type: ignore[operator]
    return items


class YahooFundamentalsProvider(FundamentalsProvider):
    """Collect the 18 ratios live from Yahoo Finance.

    Useful to bootstrap ``security_fundamentals`` before switching scoring to
    the database provider. Yahoo only exposes the latest reported values, so
    ``as_of_date`` is accepted for interface compatibility only.
    """

    name = "yahoo"

    def get_panel(self, symbols: DataFrame, as_of_date: str) -> DataFrame:
        """Return the latest Yahoo fundamentals for ``symbols`` (NaN where missing)."""
        import yfinance as yf

        rows: dict[int, dict[str, Optional[float]]] = {}
        for _, row in symbols.iterrows():
            sec_id = int(row["security_id"])
            try:
                rows[sec_id] = info_to_items(yf.Ticker(row["symbol"]).info or {})
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not fail all
                log.warning("yahoo fundamentals skipped %s: %s", row["symbol"], exc)
        if not rows:
            return empty_panel(symbols["security_id"].tolist())
        items = DataFrame.from_dict(rows, orient="index")
        items.index.name = "security_id"
        ratios = compute_ratios(items)
        panel = empty_panel(symbols["security_id"].tolist())
        panel.loc[ratios.index.intersection(panel.index)] = ratios
        return panel
