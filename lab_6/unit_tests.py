import pytest

from nist_tests import *


def test_frequency_bit_test_balanced_sequence():
    data = "1100100100001111110110101010001000100001011010001100001000110100110001001100011001100010100010111000"
    result = frequency_bit_test(data)
    assert result >= 0.01


def test_frequency_bit_test_unbalanced_sequence():
    data = "1"*100 + "0"*28
    result = frequency_bit_test(data)
    assert result >= 0.01


def test_frequency_bit_test_all_ones():
    data = "1"*128
    result = frequency_bit_test(data)
    assert result >= 0.01


def test_equally_consecutive_bits_balanced_sequence():
    data = "1100100100001111110110101010001000100001011010001100001000110100110001001100011001100010100010111000"
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
    data = "1100100100001111110110101010001000100001011010001100001000110100110001001100011001100010100010111000"
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
    (64, 64, "balanced"),
    (96, 32, "unbalanced"),
    (13, 115, "highly_unbalanced"),
    (100, 28, "test_case"),
])
def test_frequency_bit_parametrized(ones_count, zeros_count, expected_behavior):
    data = "1" * ones_count + "0" * zeros_count
    result = frequency_bit_test(data)
    assert result >= 0.01, f"P-value={result} < 0.01. Sequence is not random"


@pytest.mark.parametrize("ones_count,zeros_count,expected_behavior", [
    (64, 64, "balanced"),
    (96, 32, "unbalanced"),
    (13, 115, "highly_unbalanced"),
    (100, 28, "test_case"),
])
def test_equally_consecutive_bits_parametrized(ones_count, zeros_count, expected_behavior):
    data = "1" * ones_count + "0" * zeros_count
    result = equally_consecutive_bits(data)
    assert result >= 0.01, f"P-value={result} < 0.01. Sequence is not random"


@pytest.mark.parametrize("ones_count,zeros_count,expected_behavior", [
    (64, 64, "balanced"),
    (96, 32, "unbalanced"),
    (13, 115, "highly_unbalanced"),
    (100, 28, "test_case"),
])
def test_longest_sequence_parametrized(ones_count, zeros_count, expected_behavior):
    data = "1" * ones_count + "0" * zeros_count
    result = longest_sequence_test(data)
    assert result >= 0.01, f"P-value={result} < 0.01. Sequence is not random"