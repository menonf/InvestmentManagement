"""Build the EOD price universe for a backtest."""

from __future__ import annotations

import logging
from typing import Any, Iterable

import pandas as pd
from pandas import DataFrame

from data_engineering.database import database as db
from data_engineering.refinitiv import qualify_tickers, strip_event_suffix

from .portfolios import portfolio_member_ids

log = logging.getLogger(__name__)


def build_backtest_universe(
    session: Any,
    engine: Any,
    index_id: int,
    extra_rics: Iterable[str] = (),
    extra_portfolios: Iterable[str] = (),
    vendor: str = "Refinitiv",
) -> DataFrame:
    """Return ``[security_id, symbol, ric]`` for every security whose prices a backtest needs.

    The universe is every member of ``index_id`` in ``reference.index_constituents``
    (point-in-time joiners and leavers included), plus ``extra_rics`` (e.g. the
    ETF and index level used to validate a reconstruction) and every security
    held by ``extra_portfolios`` (e.g. a thesis book).

    RICs come from ``security_vendor_xref``. A ``^<event>`` suffix is stripped
    (``BAC^I98`` -> ``BAC``: the base listing is what carries prices) and bare
    tickers are exchange-qualified via LSEG so ``get_data`` returns full history.
    ``symbol`` is set equal to ``ric`` so the frame can go straight to a vendor.
    """
    import sqlalchemy as sa

    xref = db.read_security_vendor_xref(session, engine, vendor=vendor)
    xref = xref.dropna(subset=["vendor_ticker"])
    xref = xref[xref["vendor_ticker"].astype(bool)]

    needed: set[int] = set()
    members = pd.read_sql_query(
        sa.text("SELECT DISTINCT security_id FROM reference.index_constituents WHERE index_id = :i"),
        engine,
        params={"i": int(index_id)},
    )
    needed.update(int(s) for s in members["security_id"].dropna())
    for ric in extra_rics:
        hit = xref.loc[xref["vendor_ticker"] == ric, "security_id"]
        if len(hit):
            needed.add(int(hit.iloc[0]))
        else:
            log.warning("extra RIC %s not found in security_vendor_xref", ric)
    for name in extra_portfolios:
        needed.update(portfolio_member_ids(engine, name))

    uni = xref[xref["security_id"].isin(needed)][["security_id", "vendor_ticker"]].copy()
    uni = uni.sort_values("security_id").drop_duplicates("security_id")
    uni["ric"] = uni["vendor_ticker"].map(strip_event_suffix)
    qmap = qualify_tickers(uni["ric"].tolist())
    uni["ric"] = uni["ric"].map(lambda t: qmap.get(t, t))
    uni["symbol"] = uni["ric"]
    uni["security_id"] = uni["security_id"].astype(int)
    log.info(
        "backtest universe: %d securities (index_id=%s + %d extra RICs + %s)",
        len(uni),
        index_id,
        len(list(extra_rics)),
        list(extra_portfolios),
    )
    return uni[["security_id", "symbol", "ric", "vendor_ticker"]].reset_index(drop=True)
