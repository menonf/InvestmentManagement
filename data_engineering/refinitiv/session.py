"""Lazy, idempotent LSEG session management."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_SESSION_OPEN = False


def ensure_session() -> None:
    """Open the LSEG desktop/SSO session once per process.

    Safe to call repeatedly; the first call opens ``lseg.data``'s session and
    (optionally) the legacy ``refinitiv.data`` session used by some older
    scripts. Requires LSEG Workspace to be running locally.
    """
    global _SESSION_OPEN
    if _SESSION_OPEN:
        return

    import lseg.data as ld

    ld.open_session()
    try:  # optional legacy library
        import refinitiv.data as rd

        rd.open_session()
    except Exception:  # noqa: BLE001 - legacy lib is optional
        pass
    _SESSION_OPEN = True
    log.info("LSEG session opened")
