"""Portfolio and holdings set-up helpers (replace the notebook ``%%sql`` cells).

All writes are idempotent: portfolios are looked up by short name before being
created, and holdings are replaced per portfolio rather than appended.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Mapping, Optional

import pandas as pd
from pandas import DataFrame

from data_engineering.database import database as db

log = logging.getLogger(__name__)

UPSERT_BY = "data_engineering.loaders.portfolios"


def business_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """Monday-Friday calendar days in ``[start_date, end_date]`` (holidays included)."""
    return pd.bdate_range(start_date, end_date)


def get_or_create_portfolio(session: Any, short_name: str, name: str, portfolio_type: str) -> int:
    """Return the ``port_id`` for ``short_name``/``portfolio_type``, creating the row if needed."""
    existing = (
        session.query(db.Portfolio.port_id)
        .filter(db.Portfolio.portfolio_short_name == short_name, db.Portfolio.portfolio_type == portfolio_type)
        .order_by(db.Portfolio.port_id)
        .first()
    )
    if existing is not None:
        log.info("portfolio %s exists (port_id=%s)", short_name, existing[0])
        return int(existing[0])
    row = db.Portfolio(portfolio_short_name=short_name, portfolio_name=name, portfolio_type=portfolio_type, is_active="1")
    session.add(row)
    session.commit()
    log.info("created portfolio %s (port_id=%s)", short_name, row.port_id)
    return int(row.port_id)


def resolve_security_ids_by_ric(session: Any, engine: Any, rics: list[str], vendor: str = "Refinitiv") -> dict[str, int]:
    """Map vendor tickers (RICs) to ``security_id`` via ``security_vendor_xref``.

    Inputs may be bare tickers (the MAG8 book uses ``MSFT``) while the xref now
    stores the exchange-qualified RIC (``MSFT.O``); both sides are normalised
    (exchange qualifier and ``^event`` suffix stripped) before matching so the
    lookup tolerates either spelling. The returned dict is keyed by the *input*
    ticker, so a caller that passed ``MSFT`` still receives ``{"MSFT": id}``.
    """
    from data_engineering.refinitiv import strip_exchange_qualifier

    xref = db.read_security_vendor_xref(session, engine, vendor=vendor)
    xref = xref.dropna(subset=["vendor_ticker"]).copy()
    xref["_norm"] = xref["vendor_ticker"].map(strip_exchange_qualifier)
    want = {strip_exchange_qualifier(str(r)): r for r in rics}
    matched = xref[xref["_norm"].isin(want)].sort_values("is_primary", ascending=False).drop_duplicates("_norm")
    # itertuples mangles underscore-prefixed column names, so iterate rows.
    return {want[row["_norm"]]: int(row["security_id"]) for _, row in matched.iterrows()}


def ensure_security(
    session: Any,
    engine: Any,
    ric: str,
    name: str,
    security_type: str,
    asset_class: str = "Equity",
    vendor: str = "Refinitiv",
) -> int:
    """Return the ``security_id`` for ``ric``, inserting master + xref rows when missing.

    Used for instruments that are not index constituents (an ETF such as
    ``SPY.P`` or an index level such as ``.SPX``) so holdings can reference them.
    """
    found = resolve_security_ids_by_ric(session, engine, [ric], vendor=vendor)
    if ric in found:
        return found[ric]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    master = db.SecurityMaster(
        name=name, security_type=security_type, asset_class=asset_class, is_active=True, upsert_date=now, upsert_by=UPSERT_BY
    )
    session.add(master)
    session.flush()
    sec_id = int(master.security_id)
    session.add(
        db.SecurityVendorXref(
            security_id=sec_id, vendor=vendor, vendor_ticker=ric, is_primary=True, is_active=True, upsert_date=now, upsert_by=UPSERT_BY
        )
    )
    session.commit()
    log.info("inserted security_master + xref for %s (security_id=%s)", ric, sec_id)
    return sec_id


def replace_portfolio_holdings(session: Any, port_id: int, holdings: DataFrame) -> int:
    """Delete every holding of ``port_id`` and insert ``holdings``.

    Args:
        holdings: columns ``as_of_date``, ``security_id``, ``held_shares``.

    Returns:
        Number of rows written.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = holdings.copy()
    rows["port_id"] = int(port_id)
    rows["as_of_date"] = pd.to_datetime(rows["as_of_date"]).dt.strftime("%Y-%m-%d")
    rows["upsert_date"] = now
    rows["upsert_by"] = UPSERT_BY
    session.query(db.PortfolioHoldings).filter(db.PortfolioHoldings.port_id == int(port_id)).delete(synchronize_session=False)
    session.bulk_insert_mappings(
        db.PortfolioHoldings,
        rows[["as_of_date", "port_id", "security_id", "held_shares", "upsert_date", "upsert_by"]].to_dict(orient="records"),
    )
    session.commit()
    log.info("wrote %d holdings rows for port_id=%s", len(rows), port_id)
    return len(rows)


def write_constant_holdings(
    session: Any, port_id: int, shares_by_security: Mapping[int, float], start_date: str, end_date: str
) -> int:
    """Hold a fixed number of shares per security on every business day of the window.

    This is the "buy and hold" thesis book (e.g. the Magnificent 8) and the
    single-line ETF / index-level portfolios used to validate a reconstruction.
    """
    dates = business_days(start_date, end_date)
    frame = DataFrame(
        [(d, int(sec), float(qty)) for d in dates for sec, qty in shares_by_security.items()],
        columns=["as_of_date", "security_id", "held_shares"],
    )
    return replace_portfolio_holdings(session, port_id, frame)


def write_index_holdings(
    session: Any, engine: Any, port_id: int, index_id: int, start_date: str, end_date: str, held_shares: float = 1.0
) -> int:
    """Expand ``reference.index_constituents`` membership intervals into daily holdings.

    Every business day on which a constituent is a member gets one row with
    ``held_shares`` (1.0 by default - the market-cap weight is computed later
    from float-adjusted shares outstanding by the weights module).
    """
    members = db.read_index_constituents(session, engine, start_date, end_date)
    members = members[(members["index_id"] == index_id) & members["security_id"].notna()].copy()
    if members.empty:
        raise ValueError(f"No index_constituents rows for index_id={index_id}")
    members["start"] = pd.to_datetime(members["start_date"])
    members["end"] = pd.to_datetime(members["end_date"]).fillna(pd.Timestamp(end_date))
    dates = business_days(start_date, end_date)
    rows = []
    for r in members.itertuples(index=False):
        active = dates[(dates >= r.start) & (dates <= r.end)]
        rows.append(DataFrame({"as_of_date": active, "security_id": int(r.security_id), "held_shares": held_shares}))
    frame = pd.concat(rows, ignore_index=True).drop_duplicates(["as_of_date", "security_id"])
    return replace_portfolio_holdings(session, port_id, frame)


def portfolio_member_ids(engine: Any, portfolio_short_name: str) -> list[int]:
    """Distinct ``security_id``s ever held by a portfolio."""
    import sqlalchemy as sa

    q = sa.text(
        "SELECT DISTINCT ph.security_id FROM dbo.portfolio_holdings ph JOIN dbo.portfolio p ON p.port_id = ph.port_id WHERE p.portfolio_short_name = :name"
    )
    return [int(x) for x in pd.read_sql_query(q, engine, params={"name": portfolio_short_name})["security_id"].dropna()]


def portfolio_members_on(engine: Any, portfolio_short_name: str) -> Optional[pd.Series]:
    """Series ``as_of_date -> set(security_id)`` of holdings per date (point-in-time membership)."""
    import sqlalchemy as sa

    q = sa.text(
        "SELECT ph.as_of_date, ph.security_id FROM dbo.portfolio_holdings ph JOIN dbo.portfolio p ON p.port_id = ph.port_id WHERE p.portfolio_short_name = :name"
    )
    h = pd.read_sql_query(q, engine, params={"name": portfolio_short_name})
    if h.empty:
        return None
    h["as_of_date"] = pd.to_datetime(h["as_of_date"])
    return h.groupby("as_of_date")["security_id"].apply(lambda s: set(int(x) for x in s)).sort_index()
