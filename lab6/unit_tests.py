import pytest
from module_for_tests import calculate_symbol_frequency, make_key, decryption_cod3, text_encryption, text_decryption

TEST_TEXT = "aabbcc"
FREQ_DATA = {'a': 0.3, 'b': 0.3, 'c': 0.4}
FREQ_TASK = {'x': 0.4, 'y': 0.3, 'z': 0.3}
DECRYPTION_DATA = "x!y"
DECRYPTION_KEY = {'x': 'a', 'y': 'b'}


EXPECTED_FREQ_BASIC = {'a': 1/3, 'b': 1/3, 'c': 1/3}
EXPECTED_KEY_RESULT = {'x': 'a', 'y': 'b', 'z': 'c'}
EXPECTED_DECRYPTION = "a!b"

def test_calculate_symbol_frequency_basic():

    result = calculate_symbol_frequency(TEST_TEXT)
    assert result == EXPECTED_FREQ_BASIC


def test_make_key_creation():
    result = make_key(FREQ_DATA, FREQ_TASK)
    assert result == EXPECTED_KEY_RESULT


def test_decryption_cod3_simple():
    result = decryption_cod3(DECRYPTION_DATA, DECRYPTION_KEY)
    assert result == EXPECTED_DECRYPTION