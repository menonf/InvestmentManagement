from data_engineering.database import database


def test_parse_date_validates_iso_date_strings() -> None:
    assert database._parse_date("2024-01-15") == "2024-01-15"
    assert database._parse_date("2024-01-15", "%Y-%m-%d") == "2024-01-15"


def test_models_are_reexported_from_facade() -> None:
    from data_engineering.database import models

    assert database.MarketData is models.MarketData
    assert database.SecurityFundamentals is models.SecurityFundamentals
    assert callable(database.get_db_connection)
