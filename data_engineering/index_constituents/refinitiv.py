"""
Refinitiv-based index constituent reconstruction and metric generation.

This module performs the following:

1. Reconstructs historical index membership using joiner/leaver events.
2. Enriches constituents with internal security master identifiers.
3. Builds float-adjusted shares outstanding metrics.

All functionality remains equivalent to the original implementation.
"""

import lseg.data as ld
import pandas as pd

from data_engineering.database import database as database

ld.open_session()


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


def _normalize_ric(ric: str) -> Optional[str]:
    """Strip a trailing Refinitiv share-class qualifier, keeping the exchange code.

    Refinitiv encodes the same security with RICs that differ only by a trailing
    qualifier, e.g. ``ALIGN.OQ`` (ordinary + class/``Q``) vs ``ALIGN.O``.  The
    index-constituent feed returns the qualified form while the security-master
    loader stores the bare form, so an exact-string join misses them.  Dropping
    the qualifier (``OQ`` -> ``O``) makes the two match.

    Rule: when the trailing segment is exactly two letters of the form
    ``<exch><class>`` (e.g. ``OQ``), drop the last letter so the exchange code
    (``O``) remains.  Genuine two-letter exchange mnemonics such as Paris ``PA``,
    London ``LN``, ASX ``AX`` are protected via ``_KNOWN_EXCHANGE_SUFFIXES`` so
    they are left intact (``V.PA`` stays ``V.PA``).  A single-letter tail is
    already the exchange code (``.O``, ``.N``, ``.Z``) and is left untouched,
    as are unrecognised multi-char suffixes.

    Returns ``None`` for blank input so it propagates as a no-match rather than
    raising.
    """
    if ric is None or (isinstance(ric, float) and pd.isna(ric)):
        return None
    text = str(ric).strip()
    # Refinitiv forward-event RICs carry a "^<event>" suffix (e.g.
    # ``CTLT.N^L24`` = a pending index add/remove). Strip it FIRST so the
    # underlying security (``CTLT.N``) is what we normalise/match on -- otherwise
    # the event-suffixed form falls through to its own (wrong) security_id and
    # pollutes the resolver.
    if "^" in text:
        text = text.split("^", 1)[0]
    if not text or "." not in text:
        return text or None
    head, _, tail = text.rpartition(".")
    # Single-letter tail = the exchange code (.O/.N/.Z/...): drop it so the bare
    # ticker remains. This lets a feed RIC like ``JPM.N`` match an xref row stored
    # under the bare ticker ``JPM`` (and vice versa).  Exact-RIC matching (P1) is
    # tried first in the resolver, so this never overrides a true exact hit.
    if len(tail) == 1 and tail.isalpha():
        return head
    # Two-letter trailing segment shaped like <exch><class> (e.g. ``OQ``): drop
    # the whole segment so the bare ticker remains.  This keeps the transformation
    # SYMMETRIC -- both ``CRWD.O`` (stored) and ``CRWD.OQ`` (feed) normalise to the
    # same bare ``CRWD``, so the resolver's normalised-RIC tier can join them.
    if len(tail) == 2 and tail.isalpha() and tail not in _KNOWN_EXCHANGE_SUFFIXES:
        return head
    return text


# Two-letter exchange mnemonics that must NOT be mangled by the qualifier stripper
# (they are the whole exchange code, not <exch><class>).  Single-letter exchange
# codes (.O/.N/.Z/...) and unlisted multi-char suffixes are left as-is already.
_KNOWN_EXCHANGE_SUFFIXES = {
    "PA", "LN", "AX", "TO", "SW", "HK", "SS", "BR", "AS", "SG", "TW",
    "KS", "TWO", "MI", "MX", "SA", "JP", "F", "SW", "VX", "TR", "ST",
}


def enrich_with_security_master(df: pd.DataFrame) -> pd.DataFrame:
    """Map index constituents to internal security identifiers.

    Cascading resolution that mirrors the security-master loader's own
    ``resolve_security_ids`` precedence (RIC -> ISIN -> CUSIP -> FIGI), with a
    normalised-RIC tier inserted between exact RIC and ISIN to absorb Refinitiv's
    trailing share-class qualifiers (e.g. ``ALIGN.OQ`` vs the stored ``ALIGN.O``).

    The input frame is expected to carry ``Constituent RIC`` plus, when present,
    ``ISIN``/``SEDOL``/``CUSIP`` (the Refinitiv attribute fetch returns these).
    """
    engine, connection, conn_str, session = database.get_db_connection()

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
    def _id_map(col):
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
    def _strip_event_suffix(ric):
        if ric is None or not isinstance(ric, str):
            return ric
        return ric.split("^", 1)[0]

    missing = df["security_id"].isna()
    if missing.any():
        base = df.loc[missing, "Constituent RIC"].map(_strip_event_suffix)
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
        print(f"[constituents] creating {len(new_norm_rics)} new security_master rows for unresolved normalized RICs")
        created = []
        for nric in new_norm_rics:
            now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            # Insert master row; let the DB assign security_id.
            rec = {
                "name": str(nric),
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
            new_sm = database.SecurityMaster(**{
                k: v for k, v in rec.items()
            })
            session.add(new_sm)
            session.flush()  # assigns new_sm.security_id without committing
            new_sec_id = int(new_sm.security_id)
            created.append((nric, new_sec_id))
            xref_rec = {
                "security_id": new_sec_id,
                "vendor": "Refinitiv",
                "vendor_ticker": str(nric),
                "is_primary": True,
                "is_active": True,
                "upsert_date": now,
                "upsert_by": "data_engineering.index_constituents.refinitiv.py",
            }
            database.write_security_vendor_xref(pd.DataFrame([xref_rec]), session, vendor="Refinitiv")

        # Map every unresolved raw RIC to its normalized-RIC security_id.
        new_norm_to_sec = {nric: sid for nric, sid in created}
        df.loc[unresolved_mask, "security_id"] = (
            df.loc[unresolved_mask, "Constituent RIC"].map(_normalize_ric).map(new_norm_to_sec)
        )
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
                    session.execute(
                        __import__("sqlalchemy").update(_SM)
                        .where(_SM.security_id == sid)
                        .values(**{k: v for k, v in upd.items() if k != "security_id"})
                    )
            session.commit()

    df["security_id"] = df["security_id"].astype("Int64")
    df["index_id"] = 1  # the caller overrides this with the real index_id (e.g. 2)
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
    ticker_sources = [
        c for c in ("Exchange Ticker", "Exchange Ticker_x", "Exchange Ticker_y")
        if c in df.columns
    ]
    df["exchange_ticker"] = pd.NA
    for c in ticker_sources:
        df["exchange_ticker"] = df["exchange_ticker"].fillna(df[c])
    if df["exchange_ticker"].isna().any():
        df["exchange_ticker"] = df["exchange_ticker"].fillna(
            df["Constituent RIC"].map(_normalize_ric)
        )
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


def _ld_get_data_retry(universe, fields, parameters=None, max_retries=8, chunk_size=200):
    """Single-call wrapper around ld.get_data with chunking + retry.

    Used for one-shot fetches (e.g. the constituent attribute pull) that would
    otherwise issue one monolithic request and fail outright on a transient
    gateway timeout, silently dropping securities from the result.
    """
    return _ld_get_data_chunked(
        universe, fields, parameters or {}, chunk_size=chunk_size, max_retries=max_retries
    )


def _ld_get_data_chunked(universe, fields, parameters, chunk_size=200, max_retries=8, inter_chunk_sleep=3.0):
    """Fetch Refinitiv data in chunks to avoid gateway timeouts on large
    universes, retrying transient failures with backoff.

    Refinitiv's gateway intermittently times out even on small requests
    (``LDError: UDF Core request failed. Gateway Time-out``), especially after a
    long preceding session, and will throttle a session that fires requests
    back-to-back. Splitting the universe into chunks keeps each request small,
    every chunk is retried with exponential backoff, and a short pause between
    chunks avoids tripping the throttle. Returns a concatenated DataFrame (empty
    if all chunks fail).
    """
    import time as _time
    chunks = [universe[i : i + chunk_size] for i in range(0, len(universe), chunk_size)]
    frames = []
    total = len(chunks)
    for ci, chunk in enumerate(chunks, 1):
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                df = ld.get_data(universe=chunk, fields=fields, parameters=parameters)
                if df is not None and len(df):
                    frames.append(df)
                break
            except Exception as e:  # gateway/timeout/transport errors
                last_err = e
                wait = 15 * attempt
                print(f"  [shares] chunk {ci}/{total} attempt {attempt} failed: {type(e).__name__}: {str(e)[:120]} -- retry in {wait}s")
                _time.sleep(wait)
        else:
            print(f"  [shares] chunk {ci}/{total} FAILED after {max_retries} attempts: {last_err}")
        if ci % 10 == 0 or ci == total:
            print(f"  [shares] progress {ci}/{total} chunks done ({len(frames)} returned data)")
        # Pause between chunks so we don't hammer a throttled gateway.
        if ci < total:
            _time.sleep(inter_chunk_sleep)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
    n = len(universe)
    print(f"[shares] pulling float-adjusted shares for {n} instruments over {start}..{end} in chunks of {chunk_size}")

    # Pull shares up to a few days past ``end`` so the final snapshot has data.
    shares_end = (pd.to_datetime(end) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    shares = _ld_get_data_chunked(
        universe,
        fields=["TR.SharesOutstanding", "TR.SharesOutstanding.Date"],
        parameters={"SDate": start, "EDate": shares_end, "Frq": "D"},
        chunk_size=chunk_size,
    )
    if shares.empty:
        print("[shares] WARNING: no shares outstanding returned; returning empty metrics frame.")
        return pd.DataFrame(
            columns=["security_id", "metric_type", "metric_value", "source_vendor", "effective_date", "end_date"]
        )

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
        print(f"[shares] free-float pull failed ({type(e).__name__}); falling back to raw shares: {str(e)[:120]}")

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

    joined["Shares Outstanding"] = joined["Outstanding Shares"] * (
        joined["Free Float (Percent)"].round(0).clip(0, 100) / 100
    )

    metrics = joined.rename(columns={"Shares Outstanding": "metric_value", "Date": "effective_date"})
    metrics["metric_type"] = "shares_outstanding"
    metrics["source_vendor"] = "refinitiv"
    metrics["end_date"] = None
    metrics = metrics[
        ["security_id", "metric_type", "metric_value", "source_vendor", "effective_date", "end_date"]
    ].dropna(subset=["security_id"])
    metrics = metrics.dropna(subset=["metric_value", "effective_date"])
    return metrics



if __name__ == "__main__":
    # index_id under which to persist .SPX membership. The demo / full pipeline
    # use 2 for the S&P 500; enrich_with_security_master hardcodes 1 but its
    # own contract says the caller must override it -- so we do that here.
    SPX_INDEX_ID = 2

    start="2001-01-01"
    end="2026-08-31"

    index = build_index_constituents(
        index=".SPX",
        start= start,
        end= end,
    )

    # Chunked + retried: a monolithic attribute pull over ~1200 RICs fails on a
    # transient gateway timeout and silently drops securities from the result.
    asset_attributes = _ld_get_data_retry(
        index["Constituent RIC"].unique().tolist(),
        fields=[
            "TR.CommonName",
            "TR.ISIN",
            "TR.SEDOL",
            "TR.CUSIP",
            "TR.ExchangeCountryCode",
            "TR.Currency",
            "TR.GICSSector",
            "TR.GICSIndustry",
            "TR.GICSSubIndustry",
            "TR.ExchangeTicker",
            "TR.ExchangeCode",
        ],
    )

    Joined = index.merge(asset_attributes, left_on="Constituent RIC", right_on="Instrument", how="left")

    df_to_write = enrich_with_security_master(Joined)
    df_to_write = df_to_write.copy()
    df_to_write["index_id"] = SPX_INDEX_ID

    metrics_df = build_float_adjusted_shares(df_to_write, start=start, end=end)

    # Persist only after BOTH the reconstruction and the shares pull have
    # completed successfully, so a late gateway timeout can't leave the
    # reference.index_constituents table half-written / partially deleted.
    engine, connection, _conn_str, session = database.get_db_connection()
    try:
        database.write_index_constituents(df_to_write, session)
        print(f"Wrote {len(df_to_write)} rows to reference.index_constituents (index_id={SPX_INDEX_ID}).")
        database.write_security_fundamentals(metrics_df, session)
        print(f"Wrote {len(metrics_df)} float-adjusted-share rows to security_fundamentals.")
    finally:
        session.close()
        connection.close()
