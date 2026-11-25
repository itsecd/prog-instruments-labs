import pytest
from patterns import PATTERNS

def test_telephone_valid():
    assert PATTERNS["telephone"].fullmatch("+7-(999)-123-45-67") is not None

@pytest.mark.parametrize("phone, is_valid", [
    ("+7-(999)-123-45-67", True),
    ("+7-(999)-123-45-6", False),
    ("8-999-123-45-67", False),
    ("+7-(1234)-123-45-67", False),
])
def test_telephone_patterns(phone, is_valid):
    match = PATTERNS["telephone"].fullmatch(phone)
    assert (match is not None) == is_valid

@pytest.mark.parametrize("date_str, is_valid", [
    ("2026-11-08", True),
    ("9999-99-99", False),
    ("2026-13-01", False),
    ("2026-02-30", True),
])
def test_date_pattern(date_str, is_valid):
    match = PATTERNS["date"].fullmatch(date_str)
    assert (match is not None) == is_valid