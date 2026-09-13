"""Unit tests for the Refinitiv fundamentals ratio computation (no LSEG session)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics.factors.fundamentals import (
    RATIO_COLUMNS,
    REFINITIV_RAW_FIELDS,
    _compute_refinitiv_ratios,
)


def _raw_row():
    # A plausible fundamental snapshot for one RIC, keyed by the *display names*
    # LSEG's get_data returns (see REFINITIV_RAW_FIELDS). Only the fields this
    # LSEG setup actually returns are present.
    return {
        "Price Close": 100.0,
        "Market Capitalization": 1_000_000_000,
        "EBIT": 200_000_000,
        "Total Revenue": 1_000_000_000,
        "Total Debt": 400_000_000,
        "Total Assets": 2_000_000_000,
        "Total Current Assets": 800_000_000,
        "Current Liabilities": 400_000_000,
        "Total Liabilities": 1_400_000_000,
        "Gross Profit": 600_000_000,
        "Retained Earnings (Accumulated Deficit)": 300_000_000,
        "Interest Expense": -20_000_000,
        "Operating Income": 220_000_000,
        "Outstanding Shares": 10_000_000,
        "Net Income Incl Extra Before Distributions": 150_000_000,
    }


def test_all_fields_mapped():
    assert set(REFINITIV_RAW_FIELDS.values()) == {
        "price", "mkt_cap", "ebit", "revenue", "net_income", "total_debt",
        "total_assets", "current_assets", "current_liab", "total_liab",
        "gross_profit", "retained_earnings", "interest_exp", "operating_income",
        "shares",
    }


def test_ratios_computed_correctly():
    raw = pd.DataFrame([_raw_row()], index=["ABC.O"])
    out = _compute_refinitiv_ratios(raw)
    assert list(out.columns) == RATIO_COLUMNS
    # Fields this LSEG setup does NOT return -> these ratios are NaN.
    for nan_col in ["Op. In./(NWC+FA)", "Cash Ratio"]:
        assert pd.isna(out.loc["ABC.O", nan_col]), f"{nan_col} should be NaN"
    # Book equity derived = Total Assets - Total Liab = 2e9 - 1.4e9 = 0.6e9.
    be = 600_000_000
    mkt = 1_000_000_000
    ni = 150_000_000  # now returned via TR.NetIncome
    assert np.isclose(out.loc["ABC.O", "P/E"], mkt / ni)
    assert np.isclose(out.loc["ABC.O", "P/B"], mkt / be)
    assert np.isclose(out.loc["ABC.O", "RoE"], ni / be)
    assert np.isclose(out.loc["ABC.O", "ROCE"], 200_000_000 / (be + 400_000_000))
    assert np.isclose(out.loc["ABC.O", "Debt/Equity"], 400_000_000 / be)
    assert np.isclose(out.loc["ABC.O", "Book Equity/TL"], be / 1_400_000_000)
    # EV/EBIT proxy = (mkt_cap + total_debt) / ebit.
    assert np.isclose(out.loc["ABC.O", "EV/EBIT"], (mkt + 400_000_000) / 200_000_000)
    # Ratios computable from available fields.
    assert np.isclose(out.loc["ABC.O", "P/S"], 1_000_000_000 / 1_000_000_000)
    assert np.isclose(out.loc["ABC.O", "Debt Ratio"], 400_000_000 / 2_000_000_000)
    assert np.isclose(out.loc["ABC.O", "Gross Profit Margin"], 600_000_000 / 1_000_000_000)
    assert np.isclose(out.loc["ABC.O", "Working Capital Ratio"], 800_000_000 / 400_000_000)
    assert np.isclose(out.loc["ABC.O", "Asset Turnover"], 1_000_000_000 / 2_000_000_000)
    # NWC = 800-400 = 400M
    assert np.isclose(out.loc["ABC.O", "(CA-CL)/TA"], 400_000_000 / 2_000_000_000)
    assert np.isclose(out.loc["ABC.O", "RE/TA"], 300_000_000 / 2_000_000_000)
    assert np.isclose(out.loc["ABC.O", "EBIT/TA"], 200_000_000 / 2_000_000_000)
    # Interest expense stored negative; abs used
    assert np.isclose(out.loc["ABC.O", "Op. In./Interest Expense"], 220_000_000 / 20_000_000)


def test_derived_market_cap_when_direct_field_null():
    # This LSEG entitlement returns Market Capitalization as null but does return
    # PriceClose and SharesOutstanding. Verify mkt_cap is derived from price*shares
    # so P/E, P/B, P/S still populate.
    raw = pd.DataFrame(
        [{
            "Price Close": 200.0,
            "Market Capitalization": np.nan,  # LSEG returns this empty
            "Outstanding Shares": 5_000_000,
            "EBIT": 100_000_000,
            "Total Revenue": 500_000_000,
            "Net Income Incl Extra Before Distributions": 80_000_000,
            "Total Assets": 1_000_000_000,
            "Total Liabilities": 600_000_000,
        }],
        index=["DERIV.O"],
    )
    out = _compute_refinitiv_ratios(raw)
    # mkt_cap = 200 * 5e6 = 1e9
    assert np.isclose(out.loc["DERIV.O", "P/E"], 1_000_000_000 / 80_000_000)
    assert np.isclose(out.loc["DERIV.O", "P/B"], 1_000_000_000 / 400_000_000)
    assert np.isclose(out.loc["DERIV.O", "P/S"], 1_000_000_000 / 500_000_000)


def test_missing_components_yield_nan_not_inf():
    raw = pd.DataFrame(
        [{"Market Capitalization": 1_000_000_000, "Total Revenue": 0.0}],
        index=["ZERO.O"],
    )
    out = _compute_refinitiv_ratios(raw)
    # P/S -> divide by zero -> NaN (not inf)
    assert pd.isna(out.loc["ZERO.O", "P/S"])
