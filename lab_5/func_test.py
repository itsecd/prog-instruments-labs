import pytest
from nist_tests import bitwise_test, same_bits_test, length_test, split_bits, largest_number_of_units
from workfiles import read_text, write_text


@pytest.mark.parametrize("input, expect", [
    ("101010", 1.0),
    ("", None)
 ])
def test_bitwise_test(input, expect):
    result = bitwise_test(input)
    assert result == expect


@pytest.mark.parametrize("input, expect", [
    ("101010", 0.10247043485974938),
    ("", None)
 ])
def test_same_bits_test(input, expect):
    result = same_bits_test(input)
    assert result == expect


@pytest.mark.parametrize("input, expect", [
    (largest_number_of_units(split_bits('101010')), 0.886226925452758),
    (largest_number_of_units(split_bits("")), 0.886226925452758)
 ])
def test_length_test(input, expect):
    result = length_test(input)
    assert result == expect


@pytest.mark.parametrize("input, expect", [
    ("10101010101010", ['10101010']),
    ("", [])
 ])
def test_split_bits(input, expect):
    result = split_bits(input)
    assert result == expect


@pytest.mark.parametrize("input, expect", [
    (split_bits('1010101010'), {1: 1}),
    (split_bits(''), {})
 ])
def test_largest_number_of_units(input, expect):
    result = largest_number_of_units(input)
    assert result == expect


def test_read_text():
    path = "test.txt"
    with open(path, "w", encoding="utf-8") as f:
        data = 'yourmamalama'
        f.write(data)
    result = read_text(path)
    assert result == data


def test_write_text():
    path = "test.txt"
    data = 'yourmamalama'
    write_text(path, data)
    with open(path, "r", encoding="utf-8") as f:
        text=f.read()
    assert text == data