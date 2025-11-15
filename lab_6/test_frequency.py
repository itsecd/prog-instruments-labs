import pytest
from NIST_tests.frequency import *

RANDOM_SEQUENCE = "00110100011100101101010011001010"
NON_BINARY_SEQUENCE = "1012"
EMPTY_SEQUENCE = ""

def test_freq_test_calculates_p_value():
    """
    Test the frequency test on a valid binary sequence.
    The result should be a float value between 0 and 1, inclusive, representing the p-value.
    """
    result = freq_test(RANDOM_SEQUENCE)
    assert isinstance(result, float)
    assert 0 <= result <= 1

def test_freq_test_empty_sequence():
    """
    Test the frequency test on an empty sequence.
    An empty sequence should raise a ValueError since it's not a valid input for the test.
    """
    with pytest.raises(ValueError):
        freq_test(EMPTY_SEQUENCE)

def test_freq_test_non_binary_sequence():
    """
    Test the frequency test on a non-binary sequence.
    The sequence contains a non-binary character ('2'), so it should raise a ValueError.
    """
    with pytest.raises(ValueError):
        freq_test(NON_BINARY_SEQUENCE)
