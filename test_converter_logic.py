"""Unit tests for converter_logic module (pytest)."""
from unittest.mock import MagicMock

import pytest
from bs4 import BeautifulSoup

from converter_logic import parse_rate, convert_amount


@pytest.fixture
def soup():
    html = """
    <table class="tablesorter ratesTable">
        <tr><td>US Dollar</td><td>0.012</td></tr>
        <tr><td>Euro</td><td>0.011</td></tr>
        <tr><td>British Pound</td><td>0.0095</td></tr>
    </table>
    """
    return BeautifulSoup(html, 'html.parser')


def test_convert_amount_basic():
    assert convert_amount(1, 0.012) == '0.01'


def test_convert_amount_zero():
    assert convert_amount(0, 0.012) == '0.00'


def test_convert_amount_negative_amount_raises():
    with pytest.raises(ValueError, match="Amount must be non-negative"):
        convert_amount(-10, 0.012)


def test_convert_amount_negative_rate_raises():
    with pytest.raises(ValueError, match="Rate must be non-negative"):
        convert_amount(10, -0.5)


def test_parse_rate_returns_float(soup):
    assert parse_rate(soup, "US Dollar") == 0.012


def test_parse_rate_missing_currency(soup):
    with pytest.raises(ValueError, match="not found"):
        parse_rate(soup, "Unknown")


def test_parse_rate_none_soup():
    with pytest.raises(ValueError, match="not loaded"):
        parse_rate(None, "US Dollar")


def test_parse_rate_invalid_rate_value():
    bad_soup = BeautifulSoup(
        '<table class="tablesorter ratesTable">'
        '<tr><td>US Dollar</td><td>N/A</td></tr></table>',
        'html.parser'
    )
    with pytest.raises(ValueError, match="Invalid rate value"):
        parse_rate(bad_soup, "US Dollar")


@pytest.mark.parametrize("currency, expected", [
    ("US Dollar", 0.012),
    ("Euro", 0.011),
    ("British Pound", 0.0095),
])
def test_parse_rate_parametrized(soup, currency, expected):
    assert parse_rate(soup, currency) == expected


@pytest.mark.parametrize("amount, rate, expected", [
    (1, 0.012, '0.01'),
    (100, 0.012, '1.20'),
    (5000, 0.011, '55.00'),
])
def test_convert_amount_parametrized(amount, rate, expected):
    assert convert_amount(amount, rate) == expected


def test_parse_rate_with_mocked_table_missing():
    fake_soup = MagicMock()
    fake_soup.find.return_value = None
    with pytest.raises(ValueError, match="Rates table not found"):
        parse_rate(fake_soup, "US Dollar")