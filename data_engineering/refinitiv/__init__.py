"""Shared LSEG / Refinitiv plumbing used by every Refinitiv-backed loader.

Before this package existed, session handling, RIC normalisation and
chunked/retried ``ld.get_data`` calls were re-implemented (slightly
differently) in the EOD vendor, the fundamentals provider, the index
constituent loader and two notebooks. Everything Refinitiv-specific but
*not* data-specific now lives here:

- :mod:`.session`  - lazy, idempotent session opening (never at import time)
- :mod:`.rics`     - RIC helpers: strip event suffixes, normalise exchange
  qualifiers, qualify bare tickers via ``symbol_conversion``
- :mod:`.client`   - ``get_data`` / ``get_history`` wrappers with chunking,
  retry and back-off

Nothing in this package imports ``lseg.data`` at module import time, so the
rest of the code base stays importable (and unit-testable) on machines
without LSEG Workspace.
"""

from .client import get_data_chunked, get_history_chunked
from .rics import normalize_ric, qualify_tickers, strip_event_suffix, strip_exchange_qualifier
from .session import ensure_session

__all__ = [
    "ensure_session",
    "normalize_ric",
    "strip_event_suffix",
    "strip_exchange_qualifier",
    "qualify_tickers",
    "get_data_chunked",
    "get_history_chunked",
]
