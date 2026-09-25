"""Chunked, retried wrappers around ``ld.get_data`` / ``ld.get_history``.

LSEG's gateway intermittently times out even on small requests and throttles
sessions that fire requests back-to-back. Every Refinitiv loader therefore
needs the same three things: split the universe into chunks, retry each chunk
with back-off, and pause between chunks. This module is the single place that
does it.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Sequence

import pandas as pd

from .session import ensure_session

log = logging.getLogger(__name__)


def _chunks(items: Sequence[Any], size: int) -> list[list[Any]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _call_with_retry(
    fn: Callable[[], Any],
    *,
    label: str,
    max_retries: int,
    backoff_seconds: float,
    linear_backoff: bool,
) -> Optional[pd.DataFrame]:
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - gateway/transport errors
            last_err = exc
            if attempt < max_retries:
                wait = backoff_seconds * attempt if linear_backoff else backoff_seconds
                log.warning(
                    "%s attempt %d/%d failed: %s: %s -- retry in %.0fs",
                    label,
                    attempt,
                    max_retries,
                    type(exc).__name__,
                    str(exc)[:120],
                    wait,
                )
                time.sleep(wait)
    log.error("%s FAILED after %d attempts: %s", label, max_retries, last_err)
    return None


def get_data_chunked(
    universe: Sequence[str],
    fields: Sequence[str],
    parameters: Optional[dict[str, Any]] = None,
    *,
    chunk_size: int = 200,
    max_retries: int = 8,
    backoff_seconds: float = 15.0,
    inter_chunk_sleep: float = 3.0,
) -> pd.DataFrame:
    """``ld.get_data`` over a large universe, chunk by chunk, with retry.

    Returns the concatenated frame (empty if every chunk failed). Chunks that
    fail every retry are logged and skipped - callers that need completeness
    should diff the returned ``Instrument`` column against ``universe``.
    """
    ensure_session()
    import lseg.data as ld

    frames: list[pd.DataFrame] = []
    parts = _chunks(list(universe), chunk_size)
    for i, chunk in enumerate(parts, start=1):
        df = _call_with_retry(
            lambda: ld.get_data(universe=chunk, fields=list(fields), parameters=parameters or {}),
            label=f"get_data chunk {i}/{len(parts)}",
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            linear_backoff=True,
        )
        if df is not None and len(df):
            frames.append(df)
        if i % 10 == 0 or i == len(parts):
            log.info("get_data progress %d/%d chunks (%d returned data)", i, len(parts), len(frames))
        if i < len(parts) and inter_chunk_sleep:
            time.sleep(inter_chunk_sleep)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def get_history_chunked(
    universe: Sequence[str],
    fields: Sequence[str],
    parameters: Optional[dict[str, Any]] = None,
    *,
    chunk_size: int = 15,
    max_retries: int = 3,
    backoff_seconds: float = 5.0,
    on_chunk: Optional[Callable[[list[str], pd.DataFrame], None]] = None,
) -> list[tuple[list[str], pd.DataFrame]]:
    """``ld.get_history`` chunk by chunk, returning ``[(rics, frame), ...]``.

    ``get_history`` returns a DatetimeIndex x (RIC, field) MultiIndex frame that
    is awkward to concatenate across chunks, so each chunk is returned (and
    optionally streamed to ``on_chunk``) separately. Chunks that fail every
    retry or return no rows are omitted.
    """
    ensure_session()
    import lseg.data as ld

    out: list[tuple[list[str], pd.DataFrame]] = []
    parts = _chunks(list(universe), chunk_size)
    for i, chunk in enumerate(parts, start=1):
        hist = _call_with_retry(
            lambda: ld.get_history(universe=chunk, fields=list(fields), parameters=parameters or {}),
            label=f"get_history chunk {i}/{len(parts)}",
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            linear_backoff=False,
        )
        if hist is None or not isinstance(hist, pd.DataFrame) or hist.empty:
            log.warning("get_history chunk %d/%d returned no data (%d RICs)", i, len(parts), len(chunk))
            continue
        log.info("get_history chunk %d/%d OK: %d rows x %d cols", i, len(parts), len(hist), hist.shape[1])
        if on_chunk is not None:
            on_chunk(chunk, hist)
        out.append((chunk, hist))
    return out
