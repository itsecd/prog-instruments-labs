import pytest
import sys
import os
from unittest.mock import patch, mock_open
from polybius import encrypt, decrypt, save, read, POLYBIUS, REVERSE


def test_encrypt_simple():
    """Тест шифрования простого текста."""
    assert encrypt("АБВ") == "111213"


def test_encrypt_with_space():
    """Тест шифрования пробела (пробел = 63)."""
    assert encrypt("А Б") == "116312"


def test_decrypt_simple():
    """Тест дешифрования простого текста."""
    assert decrypt("111213") == "АБВ"


def test_encrypt_decrypt_cycle():
    """Тест полного цикла: шифрование -> дешифрование."""
    original = "ПРИВЕТ МИР"
    assert decrypt(encrypt(original)) == original.upper()


def test_encrypt_unknown_char():
    """Тест: символ, которого нет в квадрате Полибия, остаётся без изменений."""
    assert encrypt("!") == "!"


@pytest.mark.parametrize("text, expected", [
    ("А", "11"),
    ("Б", "12"),
    ("Я", "62"),
    (" ", "63"),
    ("АБ", "1112"),
    ("ЯЯ", "6262"),
])
def test_encrypt_parametrized(text, expected):
    """Параметризованный тест шифрования."""
    assert encrypt(text) == expected


@pytest.mark.parametrize("encrypted, expected", [
    ("11", "А"),
    ("12", "Б"),
    ("62", "Я"),
    ("63", " "),
    ("1112", "АБ"),
])
def test_decrypt_parametrized(encrypted, expected):
    """Параметризованный тест дешифрования."""
    assert decrypt(encrypted) == expected


def test_read_file_with_mock():
    """Тест чтения файла с моком open."""
    mock_content = "Тестовый текст"
    with patch("builtins.open", mock_open(read_data=mock_content)) as mock_file:
        result = read("fake_file.txt")
        assert result == mock_content
        mock_file.assert_called_once_with("fake_file.txt", "r", encoding="utf-8")


def test_save_file_with_mock():
    """Тест сохранения файла с моком open."""
    with patch("builtins.open", mock_open()) as mock_file:
        save("fake_file.txt", "контент")
        mock_file.assert_called_once_with("fake_file.txt", "w", encoding="utf-8")
        handle = mock_file()
        handle.write.assert_called_once_with("контент")


def test_read_file_not_found():
    """Тест: если файл не найден, выбрасывается FileNotFoundError."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            read("nonexistent.txt")
