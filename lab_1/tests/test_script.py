"""Тесты для функций частотного анализа и расшифровки."""

from script import decrypt_with_key, freq_liter, parse_alfa


def test_parse_alfa_letters() -> None:
    text = "Ю = 0.096456\nШ = 0.075312"
    result = parse_alfa(text)
    assert result["Ю"] == 0.096456
    assert result["Ш"] == 0.075312


def test_parse_alfa_ignores_garbage() -> None:
    text = "мусор без знака равно\nЮ = 0.096456"
    result = parse_alfa(text)
    assert result == {"Ю": 0.096456}


def test_freq_liter_counts() -> None:
    freq = freq_liter("aab", 3)
    assert freq["a"] == 2 / 3
    assert freq["b"] == 1 / 3


def test_freq_liter_empty() -> None:
    freq = freq_liter("", 1)
    assert freq == {}


def test_decrypt_with_key_basic() -> None:
    key = ["X:A", "Y:B"]
    assert decrypt_with_key(key, "XY") == "AB"


def test_decrypt_with_key_ignores_lines_without_colon() -> None:
    key = ["X:A", "мусор без двоеточия", "Y:B"]
    assert decrypt_with_key(key, "XY") == "AB"
