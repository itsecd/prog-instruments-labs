import pytest
from NIST_tests.consecutive import *

# Constants
REQUIRED_VALUE = 0.01
EXPECTED_ZERO_PVALUE = 0

# Sequence samples
ALL_ONES_SEQUENCE = "1" * 128
ALL_ZEROS_SEQUENCE = "0" * 128
BALANCED_SEQUENCE = "00110100011100101101010011001010"
UNBALANCED_SEQUENCE = "1" * 100 + "0" * 28
NON_BINARY_SEQUENCE = "1012"
EMPTY_SEQUENCE = ""

def test_consecutive_bit_test_balanced_sequence():
    result = consecutive_bit_test(BALANCED_SEQUENCE)
    assert result >= REQUIRED_VALUE, f"Expected p-value >= {REQUIRED_VALUE}, got {result}"

def test_consecutive_bit_test_unbalanced_sequence():
    result = consecutive_bit_test(UNBALANCED_SEQUENCE)
    assert result < REQUIRED_VALUE, f"Expected p-value < {REQUIRED_VALUE}, got {result}"

def test_consecutive_bit_test_all_ones():
    result = consecutive_bit_test(ALL_ONES_SEQUENCE)
    assert result == EXPECTED_ZERO_PVALUE, f"Expected p-value = {EXPECTED_ZERO_PVALUE}, got {result}"

def test_consecutive_bit_test_all_zeros():
    result = consecutive_bit_test(ALL_ZEROS_SEQUENCE)
    assert result == EXPECTED_ZERO_PVALUE, f"Expected p-value = {EXPECTED_ZERO_PVALUE}, got {result}"

def test_consecutive_bit_test_empty_sequence():
    with pytest.raises(ValueError):
        consecutive_bit_test(EMPTY_SEQUENCE)

def test_consecutive_bit_test_non_binary_sequence():
    with pytest.raises(ValueError):
        consecutive_bit_test(NON_BINARY_SEQUENCE)
