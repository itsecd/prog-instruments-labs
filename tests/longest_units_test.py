import pytest
from NIST_tests.longest_units import *

RANDOM_SEQUENCE = "01010101010101010101010101010101"
NON_BINARY_SEQUENCE = "01020101"
EMPTY_SEQUENCE = ""
pi_values =  [0.2148, 0.3672, 0.2305, 0.1875]

def test_longest_units_test_calculates_p_value():
    result = longest_units_test(RANDOM_SEQUENCE, pi_values)
    assert isinstance(result, float)
    assert 0 <= result <= 1

def test_longest_units_test_empty_sequence():
    with pytest.raises(ValueError):
        longest_units_test(EMPTY_SEQUENCE, pi_values)

def test_longest_units_test_non_binary_sequence():
    with pytest.raises(ValueError):
        longest_units_test(NON_BINARY_SEQUENCE, pi_values)