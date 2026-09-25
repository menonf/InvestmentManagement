"""Database package: ORM models, connection factory and read/write helpers.

Usage::

    from data_engineering.database import database as db

    engine, connection, conn_str, session = db.get_db_connection()
    prices = db.read_market_data(session, engine, "2025-01-01", "2025-12-31")

Modules:
    models            ORM models for the core ``dbo`` / ``reference`` tables
    connection        keyring-backed SQL Server connection factory
    database          read_* / write_* helpers and composite queries (facade)
    schema_analytics  analytics-layer models (factor_scores, portfolio_returns, attribution)
    schema_fx         fx_rates / risk_snapshots / factor_exposures models
    databricks        alternative Databricks backend with the same function names
"""
