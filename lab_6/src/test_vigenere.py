import pytest
from unittest.mock import patch

import vigenere as vmodule


def test_get_key_symb_basic():
    key = "кошка"
    idx = 2
    assert vmodule.get_key_symb(key, idx) == "ш"
