from data_engineering.database import database


def test_parse_date_validates_iso_date_strings() -> None:
    assert database._parse_date("2024-01-15") == "2024-01-15"
    assert database._parse_date("2024-01-15", "%Y-%m-%d") == "2024-01-15"
