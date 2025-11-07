import pytest
from NIST_tests.frequency import *

RANDOM_SEQUENCE = "00110100011100101101010011001010"
NON_BINARY_SEQUENCE = "1012"
EMPTY_SEQUENCE = ""

def test_freq_test_calculates_p_value():
    result = freq_test(RANDOM_SEQUENCE)
    assert isinstance(result, float)
    assert 0 <= result <= 1

def test_freq_test_empty_sequence():
    with pytest.raises(ValueError):
        freq_test(EMPTY_SEQUENCE)

def test_freq_test_non_binary_sequence():
    with pytest.raises(ValueError):
        freq_test(NON_BINARY_SEQUENCE)
