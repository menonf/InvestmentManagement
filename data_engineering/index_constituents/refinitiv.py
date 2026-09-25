"""
Refinitiv-based index constituent reconstruction and float-adjusted shares.

This module:

1. Reconstructs historical index membership from a start-date snapshot plus
   joiner/leaver events (:class:`IndexConstituents`, :func:`build_index_constituents`).
2. Maps constituents onto internal ``security_id``s, creating security-master
   rows for names that are not yet known (:func:`enrich_with_security_master`).
3. Builds float-adjusted shares outstanding, the input for market-cap
   weighting (:func:`build_float_adjusted_shares`).

LSEG plumbing (session, RIC normalisation, chunked/retried requests) comes from
:mod:`data_engineering.refinitiv`; nothing here opens a session at import time,
so the module is importable without LSEG Workspace.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from data_engineering.database import database
from data_engineering.refinitiv import ensure_session, get_data_chunked, normalize_ric, strip_event_suffix

log = logging.getLogger(__name__)

# Re-exported for callers that imported the private helper from here.
_normalize_ric = normalize_ric

INDEX_ATTRIBUTE_FIELDS = [
    "TR.CommonName",
    "TR.ISIN",
    "TR.SEDOL",
    "TR.CUSIP",
    "TR.ExchangeCountryCode",
    "TR.Currency",
    "TR.GICSSector",
    "TR.GICSIndustryGroup",
    "TR.GICSIndustry",
    "TR.ExchangeTicker",
    "TR.ExchangeCode",
]


def fetch_constituent_attributes(rics: list[str]) -> pd.DataFrame:
    """Pull identifiers / GICS / ticker attributes for constituent RICs (chunked + retried)."""
    return get_data_chunked(rics, INDEX_ATTRIBUTE_FIELDS, chunk_size=200, max_retries=8)


class IndexConstituents:
    """Service for reconstructing historical index membership using Refinitiv index constituent data."""

    def get_historical_constituents(
        self,
        index: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Reconstruct historical index membership over a date range.

        Parameters
        ----------
        index : str
            Index RIC (e.g. ".NDX").
        start : str
            Start date in ISO format (YYYY-MM-DD).
        end : str
            End date in ISO format (YYYY-MM-DD).

        Returns
        -------
        pd.DataFrame
            Interval-based index membership containing:
            - Constituent RIC
            - Exchange Ticker
            - Start Date
            - End Date
        """
        initial = self.get_constituents_as_of(index, start)
        changes = self.get_constituent_changes(index, start, end)
        return self.update_constituents(start, initial, changes)

    def get_constituents_as_of(
        self,
        ric: str,
        date: str,
    ) -> pd.DataFrame:
        """
        Retrieve index constituents active on a specific date.

        Parameters
        ----------
        ric : str
            Index RIC.
        date : str
            Date in ISO format (YYYY-MM-DD).

        Returns
        -------
        pd.DataFrame
            Constituents active on the given date with initial start and end placeholders.

        """
        ensure_session()
        import lseg.data as ld

        universe = [f"0#{ric}({date.replace('-', '')})"]

        df = ld.get_data(
            universe=universe,
            fields=["TR.PriceClose", "TR.ExchangeTicker"],
            parameters={"SDATE": date, "EDATE": date},
        )

        df = df.rename(columns={"Instrument": "Constituent RIC"})
        df["Start Date"] = date
        df["End Date"] = None

        return df[["Constituent RIC", "Exchange Ticker", "Start Date", "End Date"]]

    def get_constituent_changes(
        self,
        ric: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Retrieve joiner and leaver events for an index.

        Parameters
        ----------
        ric : str
            Index RIC.
        start : str
            Start date (YYYY-MM-DD).
        end : str
            End date (YYYY-MM-DD).

        Returns
        -------
        pd.DataFrame
            Constituent change events including:
            - Constituent RIC
            - Exchange Ticker
            - Date
            - Change type

        """
        ensure_session()
        import lseg.data as ld

        const_changes = ld.get_data(
            universe=[ric],
            fields=[
                "TR.IndexJLConstituentChangeDate",
                "TR.IndexJLConstituentRIC",
                "TR.IndexJLConstituentName",
                "TR.IndexJLConstituentituentChange",
            ],
            parameters={"SDATE": start, "EDATE": end, "IC": "B"},
        )

        tickers = ld.get_data(
            universe=const_changes["Constituent RIC"].unique(),
            fields=["TR.TickerSymbol", "TR.RIC"],
        )

        const_changes = const_changes.merge(
            tickers,
            left_on="Constituent RIC",
            right_on="RIC",
            how="left",
        )

        const_changes = const_changes.rename(columns={"Ticker Symbol": "Exchange Ticker"})

        return const_changes

    def update_constituents(
        self,
        start: str,
        constituents: pd.DataFrame,
        constituent_changes: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Construct continuous membership intervals from seed constituents and join/leave events.

        Parameters
        ----------
        start : str
            Reconstruction window start date.
        constituents : pd.DataFrame
            Constituents active at the start date.
        constituent_changes : pd.DataFrame
            Chronologically ordered join and leave events.

        Returns
        -------
        pd.DataFrame
            Interval-based membership history withadjusted start and end dates.

        """
        constituent_changes = constituent_changes.copy()
        constituent_changes["Date"] = pd.to_datetime(constituent_changes["Date"])

        start_dt = pd.to_datetime(start)

        df = pd.DataFrame(
            {
                "Constituent RIC": constituents["Constituent RIC"],
                "Exchange Ticker": constituents["Exchange Ticker"],
                "Start Date": start_dt,
                "End Date": pd.NaT,
            }
        )

        for _, change in constituent_changes.sort_values("Date").iterrows():
            ric = change["Constituent RIC"]
            change_date = change["Date"]
            change_type = change["Change"]
            ticker = change.get("Exchange Ticker", "")

            if change_type == "Joiner":
                seed_mask = (df["Constituent RIC"] == ric) & (df["Start Date"] == start_dt) & (df["End Date"].isna())

                if seed_mask.any():
                    idx = df[seed_mask].index[0]
                    df.loc[idx, "Start Date"] = change_date
                    df.loc[idx, "Exchange Ticker"] = ticker
                else:
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    "Constituent RIC": [ric],
                                    "Exchange Ticker": [ticker],
                                    "Start Date": [change_date],
                                    "End Date": [pd.NaT],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )

            elif change_type == "Leaver":
                mask = (df["Constituent RIC"] == ric) & (df["End Date"].isna()) & (df["Start Date"] <= change_date)

                if mask.any():
                    idx = df[mask].sort_values("Start Date").index[-1]
                    df.loc[idx, "End Date"] = change_date
                else:
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    "Constituent RIC": [ric],
                                    "Exchange Ticker": [ticker],
                                    "Start Date": [start_dt],
                                    "End Date": [change_date],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )

        return df


def build_index_constituents(index: str, start: str, end: str) -> pd.DataFrame:
    """
    High-level wrapper to reconstruct index membership.

    Parameters
    ----------
    index : str
        Index RIC.
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Historical index membership intervals.

    """
    ic = IndexConstituents()
    return ic.get_historical_constituents(index=index, start=start, end=end)


def enrich_with_security_master(
    df: pd.DataFrame,
    session: Optional[Any] = None,
    engine: Optional[Any] = None,
    index_id: int = 1,
) -> pd.DataFrame:
    """Map index constituents to internal security identifiers.

    Cascading resolution that mirrors the security-master loader's own
    ``resolve_security_ids`` precedence (RIC -> ISIN -> CUSIP -> FIGI), with a
    normalised-RIC tier inserted between exact RIC and ISIN to absorb Refinitiv's
    trailing share-class qualifiers (e.g. ``ALIGN.OQ`` vs the stored ``ALIGN.O``).

    The input frame is expected to carry ``Constituent RIC`` plus, when present,
    ``ISIN``/``SEDOL``/``CUSIP`` (the Refinitiv attribute fetch returns these).

    Args:
        df: constituent frame from :func:`build_index_constituents` merged with
            :func:`fetch_constituent_attributes`.
        session / engine: DB handles. A connection is opened when omitted.
        index_id: value written to the ``index_id`` column (2 = S&P 500 in the
            demo pipeline).
    """
    if session is None or engine is None:
        engine, _connection, _conn_str, session = database.get_db_connection()

    # RIC tiers come from the vendor xref (vendor_ticker).
    xref = database.read_security_vendor_xref(session, engine, "Refinitiv")
    if not xref.empty:
        xref = xref.copy()
        xref["_ric_norm"] = xref["vendor_ticker"].map(_normalize_ric)
        ric_map = xref.dropna(subset=["vendor_ticker"]).set_index("vendor_ticker")["security_id"].to_dict()
        ric_norm_map = xref.dropna(subset=["_ric_norm"]).set_index("_ric_norm")["security_id"].to_dict()
    else:
        ric_map, ric_norm_map = {}, {}

    # ISIN / SEDOL / CUSIP / FIGI tiers come from security_master.
    # NOTE: pandas dropna() does NOT drop empty strings, so a security_master row
    # with a blank SEDOL (e.g. security_id 520, name 'CTLT.N^L24') would build a
    # map entry {"": 520}. Every constituent row with a blank SEDOL would then
    # map() to 520, collapsing hundreds of unrelated securities onto one id.
    # Strip/blank-coalesce identifiers so "" never becomes a map key.
    sm = database.read_security_master(session, engine)

    def _id_map(col: str) -> Any:
        if col not in sm.columns:
            return {}
        s = sm[col].astype("string").str.strip().replace({"": pd.NA}).dropna()
        return s.map(sm["security_id"]).to_dict()

    isin_map = _id_map("isin")
    cusip_map = _id_map("cusip")
    figi_map = _id_map("figi")
    sedol_map = _id_map("sedol")

    # Priority 1: exact RIC
    df["security_id"] = df["Constituent RIC"].map(ric_map)

    # Priority 2: normalised RIC (absorbs .OQ vs .O etc.)
    missing = df["security_id"].isna()
    if missing.any():
        df.loc[missing, "security_id"] = df.loc[missing, "Constituent RIC"].map(_normalize_ric).map(ric_norm_map)

    # Priority 2b: Refinitiv forward-event RICs carry a "^<event>" suffix
    # (e.g. ``DFS.N^E25`` = a pending index add/remove). The underlying security
    # is the plain RIC (``DFS.N``), so retry the exact + normalised tiers on the
    # suffix-stripped base before falling through to the identifier cascade.
    missing = df["security_id"].isna()
    if missing.any():
        base = df.loc[missing, "Constituent RIC"].map(strip_event_suffix)
        df.loc[missing, "security_id"] = base.map(ric_map)
        still = df["security_id"].isna()
        if still.any():
            df.loc[still, "security_id"] = base[still].map(_normalize_ric).map(ric_norm_map)

    # Priority 3: ISIN -> SEDOL -> CUSIP -> FIGI
    for col, mapping in [("ISIN", isin_map), ("SEDOL", sedol_map), ("CUSIP", cusip_map), ("FIGI", figi_map)]:
        missing = df["security_id"].isna()
        if missing.any() and col in df.columns:
            df.loc[missing, "security_id"] = df.loc[missing, col].map(mapping)

    # ---------------------------------------------------------------------------
    # Create master + xref rows for constituents that still have no security_id.
    # This is required for a *clean* (truncated) run: a fresh security_master
    # has no rows, so every historical name would otherwise map to NaN and be
    # dropped. We synthesise a canonical security_master row per unresolved RIC
    # (RIC used as the name/symbol stand-in) and a Refinitiv xref row, then
    # re-resolve so downstream code always has a stable security_id.
    # ---------------------------------------------------------------------------
    unresolved_mask = df["security_id"].isna()
    if unresolved_mask.any():
        # Collapse by NORMALIZED RIC so Refinitiv's variants of the same
        # security (e.g. ``ABC.O`` from the as-of snapshot vs ``ABC.OQ`` from the
        # joiner/leaver feed) resolve to ONE security_master row / security_id
        # instead of spawning a duplicate per variant.  The xref row is stored
        # under the normalized ticker so both variants (and the bare ticker)
        # keep resolving to it on every subsequent run.
        norm_rics = df.loc[unresolved_mask, "Constituent RIC"].map(_normalize_ric)
        new_norm_rics = sorted({r for r in norm_rics.dropna().unique() if r})
        # Map each normalized RIC to a representative qualified RIC (Refinitiv's
        # exchange-qualified spelling, event suffix stripped). The xref row is
        # stored under this form (e.g. ``AVB.N``) rather than the bare ticker
        # (``AVB``) so downstream ``qualify_tickers`` never has to ask LSEG
        # ``symbol_conversion`` and risk mis-resolving to a different instrument
        # (observed in the wild: AVB -> AVB.TR Turkish fund, BFb -> BAMFDc1
        # future, HOLX -> HOLX.B^J07 delisted share). Grouping/collapse still
        # uses the normalized key, so Refinitiv variants (ABC.O / ABC.OQ) keep
        # resolving to one security_id.
        repr_ric: dict[str, str] = {}
        for raw in df.loc[unresolved_mask, "Constituent RIC"]:
            n = _normalize_ric(raw)
            if n and n not in repr_ric:
                repr_ric[n] = str(strip_event_suffix(raw))
        log.info("creating %d new security_master rows for unresolved normalized RICs", len(new_norm_rics))
        created = []
        for nric in new_norm_rics:
            qric = repr_ric.get(nric, nric)
            now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            # Insert master row; let the DB assign security_id.
            rec = {
                "name": qric,
                "security_type": "EQUITY",
                "asset_class": "EQUITY",
                "is_active": True,
                "upsert_date": now,
                "upsert_by": "data_engineering.index_constituents.refinitiv.py",
            }
            # Capture the freshly assigned id directly from the ORM insert rather
            # than re-querying security_master by name (which can match a
            # pre-existing or concurrently-inserted row and return the WRONG id,
            # collapsing many distinct RICs onto one security_id).
            new_sm = database.SecurityMaster(**{k: v for k, v in rec.items()})
            session.add(new_sm)
            session.flush()  # assigns new_sm.security_id without committing
            new_sec_id = int(new_sm.security_id)
            created.append((nric, new_sec_id))
            xref_rec = {
                "security_id": new_sec_id,
                "vendor": "Refinitiv",
                "vendor_ticker": qric,
                "is_primary": True,
                "is_active": True,
                "upsert_date": now,
                "upsert_by": "data_engineering.index_constituents.refinitiv.py",
            }
            database.write_security_vendor_xref(pd.DataFrame([xref_rec]), session, vendor="Refinitiv")

        # Map every unresolved raw RIC to its normalized-RIC security_id.
        new_norm_to_sec = {nric: sid for nric, sid in created}
        df.loc[unresolved_mask, "security_id"] = df.loc[unresolved_mask, "Constituent RIC"].map(_normalize_ric).map(new_norm_to_sec)
        # If attributes (ISIN/etc.) exist, backfill them into the new master rows.
        attr_cols = [c for c in ("ISIN", "SEDOL", "CUSOL", "CUSIP", "FIGI") if c in df.columns]
        if attr_cols and created:
            from data_engineering.database.database import SecurityMaster as _SM

            for nric, sid in created:
                # Match on the NORMALIZED ric; df carries the raw RIC variants.
                sub = df[df["Constituent RIC"].map(_normalize_ric) == nric]
                if sub.empty:
                    continue
                upd = {}
                for c in attr_cols:
                    val = sub[c].dropna()
                    if not val.empty:
                        upd[c.lower()] = str(val.iloc[0])
                if upd:
                    import sqlalchemy

                    session.execute(
                        sqlalchemy.update(_SM)
                        .where(_SM.security_id == sid)
                        .values(**{k: v for k, v in upd.items() if k != "security_id"})
                    )
            session.commit()

    df["security_id"] = df["security_id"].astype("Int64")
    df["index_id"] = index_id
    df["source_vendor"] = "refinitiv"
    df["upsert_date"] = pd.Timestamp.now().floor("s")
    df["upsert_by"] = "data_engineering.index_constituents.refinitiv.py"

    # The Exchange Ticker column appears in BOTH the historical constituent feed
    # (index_df) and the Refinitiv attribute fetch (asset_attributes), so the
    # merge yields "Exchange Ticker_x" and "Exchange Ticker_y". Either side can
    # blank a ticker for a given member (data variance across Refinitiv calls),
    # so coalesce across all candidate columns instead of trusting a single one.
    # As a final fallback use the normalised RIC (exchange suffix stripped) so a
    # constituent is never blocked purely because its ticker field came back null.
    ticker_sources = [c for c in ("Exchange Ticker", "Exchange Ticker_x", "Exchange Ticker_y") if c in df.columns]
    df["exchange_ticker"] = pd.NA
    for c in ticker_sources:
        df["exchange_ticker"] = df["exchange_ticker"].fillna(df[c])
    if df["exchange_ticker"].isna().any():
        df["exchange_ticker"] = df["exchange_ticker"].fillna(df["Constituent RIC"].map(_normalize_ric))
    # Start/End Date appear in both the historical feed and the attribute fetch,
    # so after the merge they may be suffixed "_x"/"_y". Coalesce across all
    # variants instead of relying on a bare rename that silently no-ops.
    for target, candidates in (
        ("start_date", ("Start Date", "Start Date_x", "Start Date_y")),
        ("end_date", ("End Date", "End Date_x", "End Date_y")),
    ):
        sources = [c for c in candidates if c in df.columns]
        df[target] = pd.NA
        for c in sources:
            df[target] = df[target].fillna(df[c])

    df["end_date"] = df["end_date"].replace({pd.NaT: None})

    # Keep Constituent RIC so build_float_adjusted_shares (which needs
    # Constituent RIC + exchange_ticker + security_id) can consume this frame.
    return df[
        [
            "Constituent RIC",
            "index_id",
            "security_id",
            "exchange_ticker",
            "start_date",
            "end_date",
            "source_vendor",
            "upsert_date",
            "upsert_by",
        ]
    ]


def _ld_get_data_retry(
    universe: Any, fields: Any, parameters: Optional[dict[str, Any]] = None, max_retries: int = 8, chunk_size: int = 200
) -> Any:
    """Compatibility wrapper around :func:`data_engineering.refinitiv.get_data_chunked`."""
    return get_data_chunked(universe, fields, parameters or {}, chunk_size=chunk_size, max_retries=max_retries)


def _ld_get_data_chunked(
    universe: Any, fields: Any, parameters: Any, chunk_size: int = 200, max_retries: int = 8, inter_chunk_sleep: float = 3.0
) -> Any:
    """Compatibility wrapper around :func:`data_engineering.refinitiv.get_data_chunked`."""
    return get_data_chunked(
        universe, fields, parameters, chunk_size=chunk_size, max_retries=max_retries, inter_chunk_sleep=inter_chunk_sleep
    )


def build_float_adjusted_shares(
    df_constituents: pd.DataFrame,
    start: str = "2025-01-01",
    end: str = "2025-12-31",
    chunk_size: int = 20,
) -> pd.DataFrame:
    """Build float-adjusted shares outstanding -- the source for market-cap weighting.

    Pulls ``TR.SharesOutstanding`` (daily) and ``TR.FreeFloatPct`` (monthly) from
    Refinitiv for every distinct ``Constituent RIC`` in ``df_constituents`` and
    computes ``float_adjusted_shares = shares_outstanding * free_float_pct/100``.

    Parameters
    ----------
    df_constituents : constituent frame. Must carry ``Constituent RIC`` (the
        QUALIFIED Refinitiv RIC, e.g. ``AAPL.O`` -- unqualified bare tickers
        return only sparse/latest data), ``exchange_ticker`` and ``security_id``.
    start, end : ISO date window for the pull. Parameterised so a multi-year
        reconstruction can pull time-varying float shares across the full
        backtest window.
    chunk_size : instruments per Refinitiv request. The full S&P 500 over many
        years must be chunked or Refinitiv's gateway times out (the original
        single monolithic request was the root cause of the
        ``Gateway Time-out`` failures).

    Returns
    -------
    DataFrame with columns
    ``security_id, metric_type, metric_value, source_vendor, effective_date, end_date``
    (``metric_type`` = ``shares_outstanding``). Rows with null ``metric_value`` or
    ``effective_date`` are dropped (the DB columns are NOT NULL) so a handful of
    bad points can never roll back an entire chunk's insert.
    """
    universe = df_constituents["Constituent RIC"].drop_duplicates().tolist()
    log.info("pulling float-adjusted shares for %d instruments over %s..%s in chunks of %d", len(universe), start, end, chunk_size)

    # Pull shares up to a few days past ``end`` so the final snapshot has data.
    shares_end = (pd.to_datetime(end) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    shares = _ld_get_data_chunked(
        universe,
        fields=["TR.SharesOutstanding", "TR.SharesOutstanding.Date"],
        parameters={"SDate": start, "EDate": shares_end, "Frq": "D"},
        chunk_size=chunk_size,
    )
    if shares.empty:
        log.warning("no shares outstanding returned; returning empty metrics frame")
        return pd.DataFrame(columns=["security_id", "metric_type", "metric_value", "source_vendor", "effective_date", "end_date"])

    # Free-float is best-effort: Refinitiv fails the WHOLE request when any single
    # RIC lacks TR.FreeFloatPct, so retry only once and fall back to raw shares
    # (float factor 100%) for affected securities rather than dropping them.
    free_float = pd.DataFrame()
    try:
        free_float = _ld_get_data_chunked(
            universe,
            fields=["TR.FreeFloatPct", "TR.FreeFloatPct.Date"],
            parameters={"SDate": start, "EDate": shares_end, "Frq": "M"},
            chunk_size=chunk_size,
            max_retries=1,
        )
    except Exception as e:
        log.warning("free-float pull failed (%s); falling back to raw shares: %s", type(e).__name__, str(e)[:120])

    if not free_float.empty:
        ff = (
            free_float.dropna(subset=["Free Float (Percent)"])
            .drop_duplicates(subset=["Instrument", "Date"], keep="last")
            .assign(Date=lambda x: pd.to_datetime(x["Date"]))
            .sort_values(["Instrument", "Date"])
        )
        ff = ff.set_index("Date").groupby("Instrument")["Free Float (Percent)"].resample("D").ffill().reset_index()
        merged = shares.merge(ff, how="left")
        missing_float = merged["Free Float (Percent)"].isna()
        # Where free-float is genuinely absent, use 100% (raw shares).
        merged.loc[missing_float, "Free Float (Percent)"] = 100.0
    else:
        merged = shares.copy()
        merged["Free Float (Percent)"] = 100.0

    joined = (
        df_constituents[["Constituent RIC", "exchange_ticker", "security_id"]]
        .drop_duplicates()
        .merge(merged, left_on="Constituent RIC", right_on="Instrument", how="right")
    )
    joined = joined.ffill()

    joined["Shares Outstanding"] = joined["Outstanding Shares"] * (joined["Free Float (Percent)"].round(0).clip(0, 100) / 100)

    metrics = joined.rename(columns={"Shares Outstanding": "metric_value", "Date": "effective_date"})
    metrics["metric_type"] = "shares_outstanding"
    metrics["source_vendor"] = "refinitiv"
    metrics["end_date"] = None
    metrics = metrics[["security_id", "metric_type", "metric_value", "source_vendor", "effective_date", "end_date"]].dropna(
        subset=["security_id"]
    )
    metrics = metrics.dropna(subset=["metric_value", "effective_date"])
    return metrics


#: Futures / volatility tickers that the ``.SPX`` constituent feed occasionally
#: returns as members. They carry real prices and would inflate a cap-weighted
#: reconstruction, so they are dropped by default.
NON_EQUITY_TICKERS = frozenset({"ES", "NQ", "VIX", "YM", "RTY"})


def load_index_constituents_and_shares(
    index_ric: str,
    start: str,
    end: str,
    index_id: int,
    session: Any,
    engine: Any,
    exclude_tickers: frozenset[str] = NON_EQUITY_TICKERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end reference-data load for one index: membership + float shares.

    Returns ``(constituents, shares_metrics)`` *after* writing both to the DB.
    Writes happen only once both pulls have succeeded so a late gateway timeout
    cannot leave ``reference.index_constituents`` half-written.
    """
    membership = build_index_constituents(index=index_ric, start=start, end=end)
    attributes = fetch_constituent_attributes(membership["Constituent RIC"].drop_duplicates().tolist())
    joined = membership.merge(attributes, left_on="Constituent RIC", right_on="Instrument", how="left")
    constituents = enrich_with_security_master(joined, session=session, engine=engine, index_id=index_id)
    if exclude_tickers:
        bad = constituents["exchange_ticker"].astype(str).str.upper().isin(exclude_tickers)
        if bad.any():
            log.info(
                "dropping %d non-equity ticker(s) from %s constituents: %s",
                int(bad.sum()),
                index_ric,
                sorted(constituents.loc[bad, "exchange_ticker"].unique()),
            )
            constituents = constituents[~bad]
    unresolved = int(constituents["security_id"].isna().sum())
    log.info("RIC->security_id coverage: %d/%d (%d unresolved)", len(constituents) - unresolved, len(constituents), unresolved)
    shares = build_float_adjusted_shares(constituents, start=start, end=end)
    database.write_index_constituents(constituents, session)
    database.write_security_fundamentals(shares, session)
    log.info("wrote %d constituent rows (index_id=%d) and %d shares rows", len(constituents), index_id, len(shares))
    return constituents, shares


if __name__ == "__main__":
    SPX_INDEX_ID = 2
    _engine, _connection, _conn_str, _session = database.get_db_connection()
    try:
        load_index_constituents_and_shares(".SPX", "2001-01-01", "2026-08-31", SPX_INDEX_ID, _session, _engine)
    finally:
        _session.close()
        _connection.close()
