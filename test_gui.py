"""GUI and load_rates tests with mocked Tkinter/requests (headless)."""
from unittest.mock import MagicMock, patch

import pytest
import requests

import currency_converter
from currency_converter import load_rates


@pytest.fixture
def fake_root():
    return MagicMock()


def test_load_rates_returns_none_on_request_error():
    with patch("currency_converter.req.get",
               side_effect=requests.RequestException("boom")):
        assert load_rates("http://x", {}) is None


def test_load_rates_returns_soup_on_success():
    fake_response = MagicMock()
    fake_response.text = '<table class="tablesorter ratesTable"></table>'
    with patch("currency_converter.req.get", return_value=fake_response):
        soup = load_rates("http://x", {})
    assert soup is not None
    assert soup.find('table') is not None


def test_convert_class_sets_window_title(fake_root):
    with patch("currency_converter.Soup", None):
        currency_converter.Convert(fake_root)
    fake_root.title.assert_called_once_with(
        "Currency Converter | Foreign Exchange Rates"
    )


def test_convert_class_sets_geometry(fake_root):
    with patch("currency_converter.Soup", None):
        currency_converter.Convert(fake_root)
    fake_root.geometry.assert_called_once_with("1350x800+0+0")


def test_convert_class_configures_background(fake_root):
    with patch("currency_converter.Soup", None):
        currency_converter.Convert(fake_root)
    fake_root.configure.assert_called_once_with(background="#F2C496")