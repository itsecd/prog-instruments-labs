import pytest
from unittest.mock import patch

import vigenere as vmodule


def test_get_key_symb_basic():
    key = "кошка"
    idx = 2
    assert vmodule.get_key_symb(key, idx) == "ш"


def test_get_key_symb_empty_key():
    with pytest.raises(ValueError):
        vmodule.get_key_symb("", 6)


@pytest.mark.parametrize(
    "key, index, expected",
    [
        ("cat", 0, "c"),
        ("CAT", 5, "T"),
        ("goodbye", 1, "o"),
        ("рюкзак", 9, "з"),
        ("ноутбук", 6, "к"),
        ("dog", 1234567, None)
    ]
)
def test_get_key_symb_param(key, index, expected):
    if expected is None:
        expected = key[index % len(key)]
    assert vmodule.get_key_symb(key, index) == expected


@pytest.mark.parametrize(
    "old_sym, key_sym, expected",
    [
        ("а", "б", "б"),
        ("А", "б", "Б"),
        ("!", "б", "!"),
        ("щ", "ю", "ч"),
        ("Э", "я", "Ь")
    ]
)
def test_get_encrypted_symb_basic(old_sym, key_sym, expected):
    assert vmodule.get_encrypted_symb(old_sym, key_sym) == expected


@pytest.mark.parametrize(
    "encrypted_sym, key_sym, expected",
    [
        ("б", "б", "а"),
        ("Б", "б", "А"),
        ("!", "б", "!"),
        ("х", "ю", "ч"),
        ("Ь", "я", "Э"),
        (" ", "е", " ")
    ]
)
def test_get_decrypted_symb_basic(encrypted_sym, key_sym, expected):
    assert vmodule.get_decrypted_symb(encrypted_sym, key_sym) == expected


@pytest.mark.parametrize(
    "original_text, key",
    [
        ("«Евгений Онегин» — роман в стихах русского поэта Александра"
         " Сергеевича Пушкина, начат 9 мая 1823 года и закончен 5 октября 1831"
         " года, одно из самых значительных произведений"
         " русской словесности.", "кот"),
        ("это тестовая строка для проверки работы шифра", "секрет"),
        ("Питон — мультипарадигменный высокоуровневый язык программирования"
         " общего назначения с динамической строгой типизацией и автоматическим"
         " управлением памятью", "питон"),
        ("длинный текст с множеством слов для проверки корректности", "мышь"),
        ("«Граф Лев Николаевич Толстой — один из наиболее известных русских"
         " писателей и мыслителей, один из величайших в мире"
         " писателей-романистов. Участник обороны Севастополя.»", "ноутбук")
    ]
)
def test_vigenere_cipher_decrypt(original_text, key):
    encrypted = vmodule.vigenere_cipher_encrypt(original_text, key)
    decrypted = vmodule.vigenere_cipher_decrypt(encrypted, key)
    assert decrypted == original_text


def test_vigenere_cipher_encrypt_with_mock():
    text = "тестовый текст для проверки"
    key = "привет"

    with patch("vigenere.get_encrypted_symb", side_effect=lambda old, k: old) as mock_func:
        encrypted = vmodule.vigenere_cipher_encrypt(text, key)
        assert encrypted == text
        assert mock_func.call_count == len(text)


def test_vigenere_cipher_encrypt_symb():
    text = ("Альберт Эйнштейн — швейцарский, немецкий и американский"
            " физик-теоретик и общественный деятель-гуманист, один из"
            " основателей современной теоретической физики. Лауреат Нобелевской"
            " премии по физике 1921 года.")
    key = "фильм"

    def stub(old_sym, key_sym):
        return "*"

    with patch("vigenere.get_encrypted_symb", side_effect=stub):
        encrypted = vmodule.vigenere_cipher_encrypt(text, key)

    assert encrypted == "*" * len(text)
