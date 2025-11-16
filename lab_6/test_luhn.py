import pytest
from luhn import luhn_check


@pytest.mark.parametrize("number, expected", [
    ("4532015112830366", True),
    ("4111111111111111", True),
    ("1234567812345670", True),
    ("1234567812345678", False),
])
def test_luhn(number, expected):
    assert luhn_check(number) == expected
