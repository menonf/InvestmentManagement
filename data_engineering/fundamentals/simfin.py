"""SimFin fundamentals provider (structured stub)."""

from __future__ import annotations

import logging
from typing import Optional

from pandas import DataFrame

from .base import FundamentalsProvider
from .ratios import empty_panel

log = logging.getLogger(__name__)


class SimFinFundamentalsProvider(FundamentalsProvider):
    """Collect ratios via the SimFin API.

    Only the shares-outstanding helper exists today
    (:func:`data_engineering.eod_data.simfin.fetch_fundamentals_simfin`). Full
    ratio extraction needs the SimFin bulk fundamentals download and an API
    token; until then this provider returns an all-NaN panel and logs a notice.
    """

    name = "simfin"

    def __init__(self, token: Optional[str] = None):
        """Store the optional API token (falls back to keyring/env when used)."""
        self._token = token

    def get_panel(self, symbols: DataFrame, as_of_date: str) -> DataFrame:
        """Return an all-NaN panel until ratio extraction is implemented."""
        log.info("SimFin fundamentals provider is a stub; returning an empty panel.")
        return empty_panel(symbols["security_id"].tolist())
