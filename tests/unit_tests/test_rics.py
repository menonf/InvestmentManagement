"""RIC helpers."""

import numpy as np

from data_engineering.refinitiv.rics import normalize_ric, strip_event_suffix, strip_exchange_qualifier


def test_strip_event_suffix() -> None:
    assert strip_event_suffix("BAC.N^I98") == "BAC.N"
    assert strip_event_suffix("AAPL.O") == "AAPL.O"
    assert strip_event_suffix(None) is None
    assert np.isnan(strip_event_suffix(np.nan))


def test_strip_exchange_qualifier_to_bare_ticker() -> None:
    assert strip_exchange_qualifier("MSFT.O") == "MSFT"
    assert strip_exchange_qualifier("MSFT.OQ") == "MSFT"
    assert strip_exchange_qualifier("IBM.N") == "IBM"
    assert strip_exchange_qualifier("BAC.N^I98") == "BAC"
    assert strip_exchange_qualifier(".SPX") == ".SPX"  # index RICs untouched
    assert strip_exchange_qualifier("SPY.P") == "SPY"
    assert strip_exchange_qualifier("") == ""


def test_normalize_ric_keeps_two_letter_exchanges() -> None:
    assert normalize_ric("ALIGN.OQ") == "ALIGN"
    assert normalize_ric("CRWD.O") == "CRWD"
    assert normalize_ric("V.PA") == "V.PA"  # Paris is a real two-letter exchange code
    assert normalize_ric("CTLT.N^L24") == "CTLT"
    assert normalize_ric(None) is None
    assert normalize_ric("   ") is None
