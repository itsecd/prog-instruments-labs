import pytest

from nist_tests import (
    equally_consecutive_bits,
    frequency_bit_test,
    longest_sequence_test
)


ALL_ONES_SEQUENCE = "1"*128
BALANCED_SEQUENCE = "11001001011011010001100100011100111101101111011010101110101001100100000000011111011110101100011110001011110010000000111100011100"
UNBALANCED_SEQUENCE_ONES = "1" * 100 + "0" * 28
UNBALANCED_SEQUENCE_ZEROS = "10" * 50 + "0" * 28

REQUIRED_VALUE = 0.01
EXPECTED_GAMMAINC_VALUE = 0.5


def test_frequency_bit_test_balanced_sequence():
    result = frequency_bit_test(BALANCED_SEQUENCE)
    assert result >= REQUIRED_VALUE


def test_frequency_bit_test_unbalanced_sequence():
    result = frequency_bit_test(UNBALANCED_SEQUENCE_ONES)
    assert result < REQUIRED_VALUE


def test_frequency_bit_test_all_ones():
    result = frequency_bit_test(ALL_ONES_SEQUENCE)
    assert result < REQUIRED_VALUE


def test_equally_consecutive_bits_balanced_sequence():
    result = equally_consecutive_bits(BALANCED_SEQUENCE)
    assert result >= REQUIRED_VALUE


def test_equally_consecutive_bits_unbalanced_sequence():
    result = equally_consecutive_bits(UNBALANCED_SEQUENCE_ZEROS)
    assert result < REQUIRED_VALUE


def test_equally_consecutive_bits_all_ones():
    result = equally_consecutive_bits(ALL_ONES_SEQUENCE)
    assert result < REQUIRED_VALUE


def test_longest_sequence_balanced_sequence():
    result = longest_sequence_test(BALANCED_SEQUENCE)
    assert result >= REQUIRED_VALUE


def test_longest_sequence_unbalanced_sequence():
    result = longest_sequence_test(UNBALANCED_SEQUENCE_ONES)
    assert result >= REQUIRED_VALUE


def test_longest_sequence_all_ones():
    result = longest_sequence_test(ALL_ONES_SEQUENCE)
    assert result >= REQUIRED_VALUE


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
        assert result >= REQUIRED_VALUE
    else:
        assert result < REQUIRED_VALUE


def test_longest_sequence_test_with_mock(monkeypatch):
    def mock_gammainc(a, x):
        return EXPECTED_GAMMAINC_VALUE

    monkeypatch.setattr("scipy.special.gammainc", mock_gammainc)

    result = longest_sequence_test(BALANCED_SEQUENCE)

    assert result == EXPECTED_GAMMAINC_VALUE
    assert result >= REQUIRED_VALUE
