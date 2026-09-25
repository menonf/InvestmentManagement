"""SQL Server connection factory (Azure SQL or a local trusted-connection instance).

Credentials come from the OS keyring under ``service_name`` (default
``ihub_sql_connection``): ``db``, ``uid``, ``pwd`` and optionally ``server`` +
``trusted=1`` for a local Windows-auth instance.
"""

from __future__ import annotations

import time
from typing import Tuple
from urllib import parse

import keyring
import sqlalchemy as sql
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session


def get_db_connection(
    service_name: str = "ihub_sql_connection",
    server: str = "ops-store-server.database.windows.net",
    driver: str = "ODBC Driver 18 for SQL Server",
    max_retries: int = 3,
    retry_interval_minutes: int = 2,
) -> Tuple[sql.Engine, sql.Connection, str, Session]:
    """
    Establish a connection to SQL Server with retry logic.

    Returns
    -------
    Tuple of (engine, connection, connection_string, session).
    """
    db = keyring.get_password(service_name, "db")
    db_user = keyring.get_password(service_name, "uid")
    db_password = keyring.get_password(service_name, "pwd")

    # Local-instance override: if keyring stores a `server` entry plus a
    # `trusted=1` flag, build a Windows-auth (Trusted_Connection) connection
    # string for a local SQL Server (e.g. MENONPC\SQLEXPRESS). This lets the
    # whole app switch from the Azure host to a local instance without editing
    # any caller. Setting `trusted=0` (or removing it) reverts to Azure.
    local_server = keyring.get_password(service_name, "server")
    trusted = keyring.get_password(service_name, "trusted") == "1"

    if local_server and trusted:
        # Use a normal SQLAlchemy URL (server + database in the authority) so
        # both the ORM engine AND the ipython-sql `%sql` magic resolve the same
        # database. The bare `mssql+pyodbc:///?odbc_connect=...` form makes
        # ipython-sql fall back to a default DB (master), so INSERTs via the
        # magic silently land in the wrong place / don't persist.
        connection_string = (
            f"mssql+pyodbc://{local_server}/{db}"
            f"?trusted_connection=yes&driver={parse.quote_plus(driver)}"
            f"&TrustServerCertificate=yes&Encrypt=no&autocommit=true"
        )
    else:
        connection_string = (
            f"mssql+pyodbc://{db_user}:{db_password}"
            f"@{server}:1433/{db}"
            f"?driver={parse.quote_plus(driver)}&Encrypt=yes&TrustServerCertificate=no&autocommit=true"
        )

    for attempt in range(1, max_retries + 1):
        try:
            engine = sql.create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_recycle=1800,
            )
            connection = engine.connect()
            session = Session(engine)
            print("Database connection successful.")
            return engine, connection, connection_string, session
        except OperationalError as e:
            print(f"Attempt {attempt} failed with error:\n{e}")
            if attempt < max_retries:
                print(f"Retrying in {retry_interval_minutes} minutes...")
                time.sleep(retry_interval_minutes * 60)
            else:
                print("All retry attempts failed. Exiting.")
                raise

    raise RuntimeError("Database connection failed: maximum retries exceeded")
