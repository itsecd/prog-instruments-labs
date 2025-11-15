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
    """
    Test the consecutive bit test on a balanced sequence.
    The p-value should be >= REQUIRED_VALUE because the sequence is random and well-distributed.
    """
    result = consecutive_bit_test(BALANCED_SEQUENCE)
    assert result >= REQUIRED_VALUE, f"Expected p-value >= {REQUIRED_VALUE}, got {result}"

def test_consecutive_bit_test_unbalanced_sequence():
    """
    Test the consecutive bit test on an unbalanced sequence.
    The p-value should be < REQUIRED_VALUE because the sequence has a clear bias (more ones than zeros).
    """
    result = consecutive_bit_test(UNBALANCED_SEQUENCE)
    assert result < REQUIRED_VALUE, f"Expected p-value < {REQUIRED_VALUE}, got {result}"

def test_consecutive_bit_test_all_ones():
    """
    Test the consecutive bit test on a sequence of all ones.
    The p-value should be exactly 0, as the sequence has no variability and is completely biased.
    """
    result = consecutive_bit_test(ALL_ONES_SEQUENCE)
    assert result == EXPECTED_ZERO_PVALUE, f"Expected p-value = {EXPECTED_ZERO_PVALUE}, got {result}"

def test_consecutive_bit_test_all_zeros():
    """
    Test the consecutive bit test on a sequence of all zeros.
    Similar to the all ones sequence, the p-value should be 0 due to lack of variability.
    """
    result = consecutive_bit_test(ALL_ZEROS_SEQUENCE)
    assert result == EXPECTED_ZERO_PVALUE, f"Expected p-value = {EXPECTED_ZERO_PVALUE}, got {result}"

def test_consecutive_bit_test_empty_sequence():
    """
    Test the consecutive bit test on an empty sequence.
    An empty sequence should raise a ValueError as it's not a valid input.
    """
    with pytest.raises(ValueError):
        consecutive_bit_test(EMPTY_SEQUENCE)

def test_consecutive_bit_test_non_binary_sequence():
    """
    Test the consecutive bit test on a non-binary sequence.
    The sequence contains a value ('2') that is not binary, so it should raise a ValueError.
    """
    with pytest.raises(ValueError):
        consecutive_bit_test(NON_BINARY_SEQUENCE)
