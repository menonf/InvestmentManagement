"""The canonical fundamental-ratio contract and the formulas behind it.

Every fundamentals provider - whatever vendor it talks to - emits a *wide*
panel indexed by ``security_id`` whose columns are exactly ``RATIO_COLUMNS``.
The ML value factor is trained on that column set, so the names and order are
part of the model contract: changing them means retraining.

Vendors differ only in *which raw items* they can supply. They map their
fields onto the internal item names in :data:`RAW_ITEMS` and call
:func:`compute_ratios`; the formulas live in one place.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame

#: The 18 fundamental ratios the ML models consume, in the exact order used when
#: the pipelines were trained. DO NOT reorder or rename without retraining.
RATIO_COLUMNS: list[str] = [
    "EV/EBIT",
    "Op. In./(NWC+FA)",
    "P/E",
    "P/B",
    "P/S",
    "Op. In./Interest Expense",
    "Working Capital Ratio",
    "RoE",
    "ROCE",
    "Debt/Equity",
    "Debt Ratio",
    "Cash Ratio",
    "Asset Turnover",
    "Gross Profit Margin",
    "(CA-CL)/TA",
    "RE/TA",
    "EBIT/TA",
    "Book Equity/TL",
]

#: Internal names for the raw accounting items a vendor may supply. Missing
#: items are allowed; the ratios that need them come back as NaN.
RAW_ITEMS: list[str] = [
    "price",
    "mkt_cap",
    "shares",
    "ent_val",
    "ebit",
    "operating_income",
    "revenue",
    "gross_profit",
    "net_income",
    "interest_exp",
    "total_assets",
    "total_liab",
    "book_equity",
    "total_debt",
    "current_assets",
    "current_liab",
    "cash",
    "fixed_assets",
    "retained_earnings",
]

#: Ratios where a higher value means *cheaper* / better for a long-only value
#: screen. Used by :class:`analytics.factors.value.ValueFactor`.
VALUE_MULTIPLES_LOWER_IS_BETTER = ["EV/EBIT", "P/E", "P/B", "P/S"]


def empty_panel(security_ids: Iterable[int]) -> DataFrame:
    """Return an all-NaN ratio panel indexed by ``security_id``."""
    idx = pd.Index(list(security_ids), name="security_id")
    return DataFrame(index=idx, columns=RATIO_COLUMNS, dtype=float)


def _item(items: DataFrame, name: str) -> pd.Series:
    """Fetch one raw item as float64, NaN if the vendor did not supply it.

    Vendors can return text (``"NM"``, ``"-"``) for some fundamentals and pandas
    may hand back nullable dtypes; both are coerced to plain float so the ratio
    arithmetic below never raises on mixed dtypes.
    """
    if name not in items.columns:
        return pd.Series(np.nan, index=items.index, dtype=float)
    return pd.to_numeric(items[name], errors="coerce").astype(float)


def _safe_denominator(s: pd.Series) -> pd.Series:
    """Turn zero denominators into NaN so ratios are NaN rather than +/-inf."""
    return s.replace(0, np.nan)


def compute_ratios(items: DataFrame) -> DataFrame:
    """Compute the 18 ``RATIO_COLUMNS`` from a frame of raw accounting items.

    Args:
        items: any index (RIC, security_id, or (security_id, period_end)); columns
            drawn from :data:`RAW_ITEMS`. Unknown columns are ignored.

    Derivations applied when a direct item is missing (all standard identities):
        * ``mkt_cap``     = price x shares
        * ``book_equity`` = total_assets - total_liabilities
        * ``total_liab``  = total_assets - book_equity
        * ``ent_val``     = mkt_cap + total_debt (equity value + debt proxy)

    Returns:
        DataFrame with the same index as ``items`` and columns = ``RATIO_COLUMNS``
        (float64). Ratios whose components are missing are NaN, never 0.
    """
    price = _item(items, "price")
    shares = _item(items, "shares")
    mkt_cap = _item(items, "mkt_cap")
    mkt_cap = mkt_cap.where(mkt_cap.notna(), price * shares)

    ebit = _item(items, "ebit")
    operating_income = _item(items, "operating_income")
    revenue = _item(items, "revenue")
    gross_profit = _item(items, "gross_profit")
    net_income = _item(items, "net_income")
    interest_exp = _item(items, "interest_exp")
    total_assets = _item(items, "total_assets")
    total_debt = _item(items, "total_debt")
    current_assets = _item(items, "current_assets")
    current_liab = _item(items, "current_liab")
    cash = _item(items, "cash")
    fixed_assets = _item(items, "fixed_assets")
    retained_earnings = _item(items, "retained_earnings")

    total_liab = _item(items, "total_liab")
    book_equity = _item(items, "book_equity")
    book_equity = book_equity.where(book_equity.notna(), total_assets - total_liab)
    total_liab = total_liab.where(total_liab.notna(), total_assets - book_equity)

    ent_val = _item(items, "ent_val")
    ent_val = ent_val.where(ent_val.notna(), mkt_cap + total_debt)

    nwc = current_assets - current_liab

    out = DataFrame(index=items.index, columns=RATIO_COLUMNS, dtype=float)
    out["EV/EBIT"] = ent_val / _safe_denominator(ebit)
    out["Op. In./(NWC+FA)"] = operating_income / _safe_denominator(nwc + fixed_assets)
    out["P/E"] = mkt_cap / _safe_denominator(net_income)
    out["P/B"] = mkt_cap / _safe_denominator(book_equity)
    out["P/S"] = mkt_cap / _safe_denominator(revenue)
    out["Op. In./Interest Expense"] = operating_income / _safe_denominator(interest_exp.abs())
    out["Working Capital Ratio"] = current_assets / _safe_denominator(current_liab)
    out["RoE"] = net_income / _safe_denominator(book_equity)
    out["ROCE"] = ebit / _safe_denominator(book_equity + total_debt)
    out["Debt/Equity"] = total_debt / _safe_denominator(book_equity)
    out["Debt Ratio"] = total_debt / _safe_denominator(total_assets)
    out["Cash Ratio"] = cash / _safe_denominator(current_liab)
    out["Asset Turnover"] = revenue / _safe_denominator(total_assets)
    out["Gross Profit Margin"] = gross_profit / _safe_denominator(revenue)
    out["(CA-CL)/TA"] = nwc / _safe_denominator(total_assets)
    out["RE/TA"] = retained_earnings / _safe_denominator(total_assets)
    out["EBIT/TA"] = ebit / _safe_denominator(total_assets)
    out["Book Equity/TL"] = book_equity / _safe_denominator(total_liab)
    return out.astype("float64")
