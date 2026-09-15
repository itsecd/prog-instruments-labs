"""Тесты для модуля uno."""

from __future__ import annotations

import pytest

from uno import find_incorrect_birthdates, is_valid_real_date

@pytest.mark.parametrize(
    ("date_str", "expected"),
    [
        ("01.01.2000", True),
        ("29.02.2024", True),
        ("29.02.2023", False),   # не високосный
        ("31.04.2020", False),   # апреля 31 нет
        ("32.01.2000", False),
        ("00.01.2000", False),
        ("01.13.2000", False),
        ("01-01-2000", False),   # дефис не допускается
        ("01/01/2000", False),   # слэш не допускается
        ("1899.01.2000", False),
        ("01.01.2100", False),
        (" 01.01.2000 ", True),  # пробелы по краям допускаются
    ],
)
def test_is_valid_real_date(date_str: str, expected: bool) -> None:
    assert is_valid_real_date(date_str) is expected