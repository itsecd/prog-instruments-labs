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