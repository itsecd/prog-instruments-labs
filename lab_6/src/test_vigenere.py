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