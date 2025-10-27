import pytest

from nist_tests import *


def test_frequency_bit_test_balanced_sequence():
    data = "11001001011011010001100100011100111101101111011010101110101001100100000000011111011110101100011110001011110010000000111100011100"
    result = frequency_bit_test(data)
    assert result >= 0.01


def test_frequency_bit_test_unbalanced_sequence():
    data = "1"*100 + "0"*28
    result = frequency_bit_test(data)
    assert result < 0.01


def test_frequency_bit_test_all_ones():
    data = "1"*128
    result = frequency_bit_test(data)
    assert result < 0.01 or result == 0


def test_equally_consecutive_bits_balanced_sequence():
    data = "11001001011011010001100100011100111101101111011010101110101001100100000000011111011110101100011110001011110010000000111100011100"
    result = equally_consecutive_bits(data)
    assert result >= 0.01


def test_equally_consecutive_bits_unbalanced_sequence():
    data = "1"*100 + "0"*28
    result = equally_consecutive_bits(data)
    assert result >= 0.01


def test_equally_consecutive_bits_all_ones():
    data = "1"*128
    result = equally_consecutive_bits(data)
    assert result >= 0.01


def test_longest_sequence_balanced_sequence():
    data = "11001001011011010001100100011100111101101111011010101110101001100100000000011111011110101100011110001011110010000000111100011100"
    result = longest_sequence_test(data)
    assert result >= 0.01


def test_longest_sequence_unbalanced_sequence():
    data = "1"*100 + "0"*28
    result = longest_sequence_test(data)
    assert result >= 0.01


def test_longest_sequence_all_ones():
    data = "1"*128
    result = longest_sequence_test(data)
    assert result >= 0.01


@pytest.mark.parametrize("ones_count,zeros_count,expected_behavior", [
    (66, 62, "balanced"),
    (96, 32, "unbalanced"),
    (13, 115, "highly_unbalanced"),
    (100, 28, "test_case"),
])
def test_frequency_bit_parametrized(ones_count, zeros_count, expected_behavior):
    data = "1" * ones_count + "0" * zeros_count
    result = frequency_bit_test(data)
    if expected_behavior == "balanced":
        assert result >= 0.01
    else:
        assert result < 0.01
