"""Ratio contract and formulas (no vendor access)."""

import numpy as np
import pandas as pd

from data_engineering.fundamentals import RATIO_COLUMNS, REFINITIV_RAW_FIELDS, compute_ratios, compute_refinitiv_ratios, panel_to_long
from data_engineering.fundamentals.yahoo import info_to_items


def _raw_row() -> dict[str, float]:
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


def test_contract_has_18_ratios() -> None:
    assert len(RATIO_COLUMNS) == 18
    assert len(set(RATIO_COLUMNS)) == 18


def test_all_refinitiv_fields_mapped() -> None:
    assert set(REFINITIV_RAW_FIELDS.values()) == {
        "price", "mkt_cap", "ebit", "revenue", "net_income", "total_debt", "total_assets", "current_assets",
        "current_liab", "total_liab", "gross_profit", "retained_earnings", "interest_exp", "operating_income", "shares",
    }


def test_refinitiv_ratios_unchanged_from_original() -> None:
    """The numbers asserted by the original test-suite must still come out of the shared formula set."""
    out = compute_refinitiv_ratios(pd.DataFrame([_raw_row()], index=["ABC.O"]))
    assert list(out.columns) == RATIO_COLUMNS
    for nan_col in ["Op. In./(NWC+FA)", "Cash Ratio"]:
        assert pd.isna(out.loc["ABC.O", nan_col])
    be, mkt, ni = 600_000_000, 1_000_000_000, 150_000_000
    assert np.isclose(out.loc["ABC.O", "P/E"], mkt / ni)
    assert np.isclose(out.loc["ABC.O", "P/B"], mkt / be)
    assert np.isclose(out.loc["ABC.O", "RoE"], ni / be)
    assert np.isclose(out.loc["ABC.O", "ROCE"], 200_000_000 / (be + 400_000_000))
    assert np.isclose(out.loc["ABC.O", "Debt/Equity"], 400_000_000 / be)
    assert np.isclose(out.loc["ABC.O", "Book Equity/TL"], be / 1_400_000_000)
    assert np.isclose(out.loc["ABC.O", "EV/EBIT"], (mkt + 400_000_000) / 200_000_000)
    assert np.isclose(out.loc["ABC.O", "P/S"], 1.0)
    assert np.isclose(out.loc["ABC.O", "Debt Ratio"], 0.2)
    assert np.isclose(out.loc["ABC.O", "Gross Profit Margin"], 0.6)
    assert np.isclose(out.loc["ABC.O", "Working Capital Ratio"], 2.0)
    assert np.isclose(out.loc["ABC.O", "Asset Turnover"], 0.5)
    assert np.isclose(out.loc["ABC.O", "(CA-CL)/TA"], 0.2)
    assert np.isclose(out.loc["ABC.O", "RE/TA"], 0.15)
    assert np.isclose(out.loc["ABC.O", "EBIT/TA"], 0.1)
    assert np.isclose(out.loc["ABC.O", "Op. In./Interest Expense"], 11.0)


def test_market_cap_derived_when_field_null() -> None:
    raw = pd.DataFrame(
        [{"Price Close": 200.0, "Market Capitalization": np.nan, "Outstanding Shares": 5_000_000, "EBIT": 100_000_000, "Total Revenue": 500_000_000,
          "Net Income Incl Extra Before Distributions": 80_000_000, "Total Assets": 1_000_000_000, "Total Liabilities": 600_000_000}],
        index=["DERIV.O"],
    )
    out = compute_refinitiv_ratios(raw)
    assert np.isclose(out.loc["DERIV.O", "P/E"], 1e9 / 80e6)
    assert np.isclose(out.loc["DERIV.O", "P/B"], 1e9 / 400e6)
    assert np.isclose(out.loc["DERIV.O", "P/S"], 2.0)


def test_zero_denominator_gives_nan_not_inf() -> None:
    out = compute_refinitiv_ratios(pd.DataFrame([{"Market Capitalization": 1e9, "Total Revenue": 0.0}], index=["ZERO.O"]))
    assert pd.isna(out.loc["ZERO.O", "P/S"])
    assert not np.isinf(out.to_numpy(dtype=float)).any()


def test_text_values_are_coerced() -> None:
    out = compute_refinitiv_ratios(pd.DataFrame([{"Market Capitalization": "1000", "Total Revenue": "NM"}], index=["TXT.O"]))
    assert pd.isna(out.loc["TXT.O", "P/S"])


def test_full_items_enable_all_ratios() -> None:
    items = pd.DataFrame([{"price": 10, "shares": 100, "ent_val": 1500, "ebit": 100, "operating_income": 110, "revenue": 1000, "gross_profit": 400,
                           "net_income": 80, "interest_exp": -10, "total_assets": 2000, "total_liab": 1200, "book_equity": 800, "total_debt": 500,
                           "current_assets": 600, "current_liab": 300, "cash": 150, "fixed_assets": 700, "retained_earnings": 200}], index=[1])
    out = compute_ratios(items)
    assert out.notna().all().all()
    assert np.isclose(out.loc[1, "Cash Ratio"], 0.5)
    assert np.isclose(out.loc[1, "Op. In./(NWC+FA)"], 110 / (300 + 700))
    assert np.isclose(out.loc[1, "EV/EBIT"], 15.0)  # direct EV used when present


def test_yahoo_info_mapping_recovers_book_equity_from_per_share_value() -> None:
    items = info_to_items({"currentPrice": 50.0, "sharesOutstanding": 1_000_000, "bookValue": 20.0, "netIncomeToCommon": 5_000_000})
    assert items["book_equity"] == 20_000_000
    out = compute_ratios(pd.DataFrame([items], index=[7]))
    assert np.isclose(out.loc[7, "P/B"], 2.5)
    assert np.isclose(out.loc[7, "P/E"], 10.0)


def test_panel_to_long_drops_nan_and_keys_rows() -> None:
    panel = pd.DataFrame({"P/E": [10.0, np.nan], "P/B": [2.0, 3.0]}, index=pd.Index([1, 2], name="security_id")).reindex(columns=RATIO_COLUMNS)
    rows = panel_to_long(panel, "2025-06-30", "test")
    assert len(rows) == 3
    assert set(rows["metric_type"]) == {"P/E", "P/B"}
    assert (rows["source_vendor"] == "test").all()
    assert rows["effective_date"].iloc[0] == pd.Timestamp("2025-06-30").date()
