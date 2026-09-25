"""Backward-compatible alias: fundamentals providers now live in :mod:`data_engineering.fundamentals`."""

from data_engineering.fundamentals import (  # noqa: F401
    RATIO_COLUMNS,
    REFINITIV_RAW_FIELDS,
    REFINITIV_REQUEST_FIELDS,
    FundamentalsProvider,
    RefinitivFundamentalsProvider,
    SimFinFundamentalsProvider,
    StaticFundamentalsProvider,
    YahooFundamentalsProvider,
    collect_and_store_fundamentals,
    compute_refinitiv_ratios,
    get_fundamentals_provider,
)
from data_engineering.fundamentals.ratios import empty_panel as _empty_panel  # noqa: F401

_compute_refinitiv_ratios = compute_refinitiv_ratios
