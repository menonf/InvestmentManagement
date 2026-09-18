"""Centralized secret/credential access.

Credentials are resolved from the OS keyring first (matching the existing
``ihub_sql_connection`` convention used by the database layer), then from
environment variables, then from a local ``.env`` if present. No secrets are
ever stored in source code.
"""

from __future__ import annotations

import os
from typing import Optional


def _read_keyring(service: str, username: str) -> Optional[str]:
    try:
        import keyring

        return keyring.get_password(service, username)
    except Exception:
        return None


def get_secret(service: str, username: str, env_var: Optional[str] = None) -> Optional[str]:
    """Resolve a secret from keyring -> env var -> None.

    Args:
        service: Keyring service name (e.g. ``"tiingo"``).
        username: Keyring username/key (e.g. ``"api_token"``).
        env_var: Optional env var name to consult as a fallback.
    """
    val = _read_keyring(service, username)
    if val:
        return val
    if env_var:
        return os.environ.get(env_var)
    return None


# Convenience accessors for each vendor / data source.

def tiingo_token() -> Optional[str]:
    """Return the Tiingo API token from keyring or environment."""
    return get_secret("tiingo", "api_token", env_var="TIINGO_API_TOKEN")


def marketstack_key() -> Optional[str]:
    """Return the Marketstack API key from keyring or environment."""
    return get_secret("marketstack", "api_key", env_var="MARKETSTACK_API_KEY")


def simfin_token() -> Optional[str]:
    """Return the SimFin API token from keyring or environment."""
    return get_secret("simfin", "api_token", env_var="SIMFIN_API_TOKEN")
