import pytest
from module_for_tests import calculate_symbol_frequency, make_key, decryption_cod3, text_encryption, text_decryption

TEST_TEXT = "aabbcc"

EXPECTED_FREQ_BASIC = {'a': 1/3, 'b': 1/3, 'c': 1/3}

def test_calculate_symbol_frequency_basic():

    result = calculate_symbol_frequency(TEST_TEXT)
    assert result == EXPECTED_FREQ_BASIC