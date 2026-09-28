from main import date_check, extract_dob, parse_records, validate_records


def test_valid_dates() -> None:
    for value in ["15.08.2001", "1/1/1900", "31-12-2025"]:
        assert date_check(value)


def test_leap_years() -> None:
    assert date_check("29.02.2000")
    assert date_check("29.02.2024")
    assert not date_check("29.02.1900")
    assert not date_check("29.02.2023")


def test_invalid_dates() -> None:
    for value in ["31.04.2000", "00.01.2000", "01.13.2000", "01.01.1899", "01.01.2026", "1/1-2000", "abc"]:
        assert not date_check(value)


def test_extract_dob() -> None:
    assert extract_dob(["Имя: Анна", "Дата рождения: 15.08.2001"]) == "15.08.2001"
    assert extract_dob(["Имя: Анна"]) is None


def test_parse_records() -> None:
    assert parse_records("A\nB\n\nC\nD") == [["A", "B"], ["C", "D"]]


def test_validate_records() -> None:
    valid = ["Дата рождения: 29.02.2024"]
    invalid = ["Дата рождения: 31.04.2024"]
    missing = ["Имя: Анна"]
    assert validate_records([valid, invalid, missing]) == ([invalid, missing], [valid])
