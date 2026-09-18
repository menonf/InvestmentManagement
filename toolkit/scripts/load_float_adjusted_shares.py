"""Resumable, chunk-by-chunk float-adjusted-shares loader.

Processes the constituent universe in small RIC chunks. For EACH chunk it pulls
shares + free-float, merges with only that chunk's securities, builds the
float-adjusted metric, and WRITES IMMEDIATELY to dbo.security_fundamentals. A
killed process therefore loses at most the in-flight chunk; re-running resumes
because already-loaded security_ids are skipped.

Run repeatedly (foreground, 600s cap) until it reports 0 remaining.
"""
import time

import pandas as pd
from sqlalchemy import text

import data_engineering.index_constituents.refinitiv as r
from data_engineering.database import database as database


def _pull_chunk_metrics(universe_chunk, securities_chunk, start, shares_end):
    """Pull shares+freefloat for one RIC chunk and return metrics rows (DataFrame).

    If Refinitiv cannot return free-float for one or more identifiers (it fails
    the WHOLE request when any single RIC lacks the field), free-float is
    treated as unavailable and those securities fall back to raw shares
    outstanding (float factor 100%) rather than being dropped entirely.
    """
    shares = r._ld_get_data_chunked(
        universe_chunk,
        fields=["TR.SharesOutstanding", "TR.SharesOutstanding.Date"],
        parameters={"SDate": start, "EDate": shares_end, "Frq": "D"},
        chunk_size=len(universe_chunk),
    )
    if shares.empty:
        return pd.DataFrame()
    # Free-float is best-effort: a single bad identifier fails the whole request,
    # so catch and fall back to raw shares for affected securities. Retry only
    # once -- a free-float "unable to collect" error is deterministic for the
    # offending RIC, so hammering it just wastes time and risks the process being
    # reaped before the raw-shares fallback can run.
    free_float = pd.DataFrame()
    try:
        free_float = r._ld_get_data_chunked(
            universe_chunk,
            fields=["TR.FreeFloatPct", "TR.FreeFloatPct.Date"],
            parameters={"SDate": start, "EDate": shares_end, "Frq": "M"},
            chunk_size=len(universe_chunk),
            max_retries=1,
        )
    except Exception as e:
        print(f"  [shares] free-float pull failed ({type(e).__name__}); falling back to raw shares for this chunk: {str(e)[:120]}")
        free_float = pd.DataFrame()

    if not free_float.empty:
        ff = (free_float.dropna(subset=["Free Float (Percent)"])
              .drop_duplicates(subset=["Instrument", "Date"], keep="last")
              .assign(Date=lambda x: pd.to_datetime(x["Date"]))
              .sort_values(["Instrument", "Date"]))
        ff = ff.set_index("Date").groupby("Instrument")["Free Float (Percent)"].resample("D").ffill().reset_index()
        merged = shares.merge(ff, how="left")
        missing_float = merged["Free Float (Percent)"].isna()
        # Where free-float is genuinely absent, use 100% (raw shares).
        merged.loc[missing_float, "Free Float (Percent)"] = 100.0
    else:
        merged = shares.copy()
        merged["Free Float (Percent)"] = 100.0

    joined = (securities_chunk[["ric", "security_id", "exchange_ticker"]]
              .drop_duplicates()
              .merge(merged, left_on="ric", right_on="Instrument", how="right"))
    joined = joined.ffill()
    joined["Shares Outstanding"] = joined["Outstanding Shares"] * (joined["Free Float (Percent)"].round(0).clip(0, 100) / 100)
    metrics = joined.rename(columns={"Shares Outstanding": "metric_value", "Date": "effective_date"})
    metrics["metric_type"] = "shares_outstanding"
    metrics["source_vendor"] = "refinitiv"
    metrics["end_date"] = None
    metrics = metrics[["security_id", "metric_type", "metric_value", "source_vendor", "effective_date", "end_date"]].dropna(subset=["security_id"])
    metrics = metrics.dropna(subset=["metric_value", "effective_date"])
    return metrics


def main(start="2001-01-01", end="2026-08-31", chunk_size=1):
    engine, connection, _cs, session = database.get_db_connection()

    const = pd.read_sql_query(text(
        "SELECT DISTINCT ic.security_id, ic.exchange_ticker, xr.vendor_ticker AS ric "
        "FROM reference.index_constituents ic "
        "LEFT JOIN dbo.security_vendor_xref xr "
        "  ON xr.security_id = ic.security_id AND xr.vendor='Refinitiv' AND xr.is_active=1 "
        "WHERE ic.security_id IS NOT NULL"
    ), con=engine)
    const = const.dropna(subset=["security_id"])
    const["ric"] = const["ric"].fillna(const["exchange_ticker"])
    const = const.dropna(subset=["ric"])
    # ric_norm is used ONLY as the grouping/join key back to security_id. The
    # ACTUAL Refinitiv pulls must use the QUALIFIED vendor_ticker (e.g. AAPL.OQ),
    # not the normalized bare ticker -- ld.get_data returns only sparse/latest
    # data for unqualified tickers.
    const["ric_norm"] = const["ric"].map(r._normalize_ric)
    const = const.dropna(subset=["ric_norm"])
    const["security_id"] = const["security_id"].astype(int)

    done = pd.read_sql_query(text(
        "SELECT DISTINCT security_id FROM dbo.security_fundamentals "
        "WHERE metric_type='shares_outstanding' AND source_vendor='refinitiv'"
    ), con=engine)["security_id"].astype(int).tolist()
    done_set = set(int(x) for x in done)

    remaining = const[~const["security_id"].isin(done_set)]
    print(f"constituents total={len(const)}  already loaded={len(done_set)}  remaining={len(remaining)}")
    if remaining.empty:
        print("Nothing to do -- all constituents already have shares_outstanding.")
        connection.close(); session.close()
        return

    # Process in RIC-norm chunks so each completed chunk is written immediately.
    uniq = remaining.drop_duplicates("ric_norm")[["ric", "ric_norm", "security_id", "exchange_ticker"]]
    chunks = [uniq.iloc[i:i + chunk_size] for i in range(0, len(uniq), chunk_size)]
    shares_end = (pd.to_datetime(end) + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    print(f"pulling in {len(chunks)} chunks of ~{chunk_size}; window {start}..{shares_end}")

    total_written = 0
    for ci, chunk in enumerate(chunks, 1):
        universe_chunk = chunk["ric"].tolist()  # qualified RICs for the API
        sec_ids = [int(x) for x in chunk["security_id"].tolist()]
        # Idempotent: drop any pre-existing shares_outstanding rows for these
        # securities before (re)inserting, so re-runs / resumed runs never
        # accumulate duplicates. Use an expanding bind for the IN list.
        from sqlalchemy import bindparam
        session.execute(
            text(
                "DELETE FROM dbo.security_fundamentals "
                "WHERE security_id IN :sids AND metric_type='shares_outstanding' AND source_vendor='refinitiv'"
            ).bindparams(bindparam("sids", expanding=True)),
            {"sids": sec_ids},
        )
        session.commit()
        metrics = _pull_chunk_metrics(universe_chunk, chunk, start, shares_end)
        if metrics.empty:
            print(f"  chunk {ci}/{len(chunks)}: no shares returned (skipped)")
            continue
        database.write_security_fundamentals(metrics, session)
        total_written += len(metrics)
        print(f"  chunk {ci}/{len(chunks)}: wrote {len(metrics)} rows (cumulative {total_written})")

    print(f"DONE this run. Total written: {total_written}. Re-run until remaining=0.")
    connection.close(); session.close()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"elapsed {time.time()-t0:.1f}s")
