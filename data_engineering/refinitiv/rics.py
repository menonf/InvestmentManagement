"""RIC (Reuters Instrument Code) helpers.

Refinitiv identifies the same listing in several spellings, which is the root
cause of most "silently dropped securities" bugs the project has hit:

``AAPL``          bare ticker (what ``security_vendor_xref`` sometimes stores)
``AAPL.O``        exchange-qualified RIC (what the fundamentals endpoint needs)
``AAPL.OQ``       exchange + share-class qualifier (what the constituent feed emits)
``BAC.N^I98``     RIC with a ``^<event>`` corporate-action / delisting suffix

The helpers below make those forms comparable. Two different normalisations
are deliberately offered because the call sites need different things:

- :func:`strip_exchange_qualifier` -> bare ticker. Used to *join* a vendor
  frame back onto ``security_vendor_xref`` regardless of qualifier spelling.
- :func:`qualify_tickers` -> exchange-qualified RIC. Used before *requesting*
  data, because ``ld.get_data`` returns sparse/empty rows for bare US tickers.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

import pandas as pd

log = logging.getLogger(__name__)

#: Two-letter exchange mnemonics that are the whole exchange code and must not
#: be mistaken for an ``<exchange><share-class>`` pair such as ``OQ``.
KNOWN_TWO_LETTER_EXCHANGES = {
    "PA",
    "LN",
    "AX",
    "TO",
    "SW",
    "HK",
    "SS",
    "BR",
    "AS",
    "SG",
    "TW",
    "KS",
    "MI",
    "MX",
    "SA",
    "JP",
    "VX",
    "TR",
    "ST",
}

#: Manual ticker -> RIC overrides for names ``symbol_conversion`` mis-resolves.
RIC_OVERRIDES: dict[str, str] = {"ANSS": "ANSS.OQ^G25"}


def strip_event_suffix(ric: Any) -> Any:
    """Drop a Refinitiv ``^<event>`` suffix: ``BAC.N^I98`` -> ``BAC.N``.

    Non-string input is returned unchanged so the helper can be mapped over a
    column containing NaN.
    """
    if isinstance(ric, str) and "^" in ric:
        return ric.split("^", 1)[0]
    return ric


def strip_exchange_qualifier(ric: Any) -> Any:
    """Reduce a RIC to its bare ticker: ``MSFT.O`` / ``MSFT.OQ`` -> ``MSFT``.

    Rules:
    - index / special RICs starting with ``.`` (``.SPX``) are left intact;
    - a ``^<event>`` suffix is removed first;
    - any trailing ``.<letters>`` exchange qualifier is removed.

    Non-string / empty input is returned unchanged.
    """
    if not isinstance(ric, str) or not ric:
        return ric
    if ric.startswith("."):
        return ric
    text = strip_event_suffix(ric)
    head, sep, tail = text.rpartition(".")
    if sep and head and tail.isalpha():
        return head
    return text


def normalize_ric(ric: Any) -> Optional[str]:
    """Canonical matching key used by the index-constituent resolver.

    Behaves like :func:`strip_exchange_qualifier` for one-letter qualifiers
    (``.O``/``.N``) and ``<exchange><class>`` pairs (``.OQ``), but keeps genuine
    two-letter exchange codes such as ``.PA`` or ``.LN`` so European listings
    are not collapsed onto the wrong ticker. Returns ``None`` for blank input.
    """
    if ric is None or (isinstance(ric, float) and pd.isna(ric)):
        return None
    text = strip_event_suffix(str(ric).strip())
    if not text or "." not in text:
        return text or None
    head, _, tail = text.rpartition(".")
    if len(tail) == 1 and tail.isalpha():
        return str(head)
    if len(tail) == 2 and tail.isalpha() and tail not in KNOWN_TWO_LETTER_EXCHANGES:
        return str(head)
    return str(text)


def qualify_tickers(tickers: Iterable[str], keep_special: bool = True) -> dict[str, str]:
    """Map bare tickers to exchange-qualified RICs via LSEG ``symbol_conversion``.

    Args:
        tickers: bare tickers (``AAPL``). Already-qualified RICs and special
            RICs (``.SPX``, ``SPY.P``) may be included; they map to themselves.
        keep_special: leave RICs starting with ``.`` or containing ``.`` alone.

    Returns:
        ``{input: qualified_ric}`` for *every* input. Unresolved inputs map to
        themselves so callers never lose a security silently. Results are keyed
        by the returned index (the input symbol), never positionally - the
        conversion service drops unresolved rows, which mis-aligns any zip.
    """
    from .session import ensure_session

    todo = sorted({t for t in tickers if isinstance(t, str) and t})
    out = {t: RIC_OVERRIDES.get(t, t) for t in todo}
    to_convert = [t for t in todo if t not in RIC_OVERRIDES and not (keep_special and "." in t)]
    if not to_convert:
        return out

    ensure_session()
    from lseg.data.content import symbol_conversion

    try:
        resp = symbol_conversion.Definition(
            symbols=to_convert,
            from_symbol_type=symbol_conversion.SymbolTypes.TICKER_SYMBOL,
            to_symbol_types=[symbol_conversion.SymbolTypes.RIC],
        ).get_data()
        df = resp.data.df
        ric_col = "RIC" if "RIC" in df.columns else df.columns[-1]
        resolved = df[ric_col].dropna().to_dict()
        out.update({k: str(v) for k, v in resolved.items() if k in out})
        unresolved = [t for t in to_convert if t not in resolved]
        if unresolved:
            log.warning("symbol_conversion left %d ticker(s) unresolved: %s", len(unresolved), unresolved[:10])
    except Exception as exc:  # noqa: BLE001 - degrade to bare tickers
        log.warning("symbol_conversion failed (%s); using bare tickers", exc)
    return out
