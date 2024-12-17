import pytest

from main import read_json, frequency_bitwise_test, identical_bit_test, longest_subsequence

def test_read_json():
    data = read_json("sequences.json")
    word = "00000000011111101001011101111011011010001011011000110111111100100011111101010011100111001110110100000011010001001101110010110001"
    assert word == data["c++"]

def test_frequency_bitwise_test():
    test_result = 0.4795001221869535
    word = "00000000011111101001011101111011011010001011011000110111111100100011111101010011100111001110110100000011010001001101110010110001"
    assert test_result == float(frequency_bitwise_test(word))

def test_identical_bit_test():
    test_result = 0.39923839789091575
    word = "00000000011111101001011101111011011010001011011000110111111100100011111101010011100111001110110100000011010001001101110010110001"
    assert test_result == float(identical_bit_test(word))

def test_longest_subsequence():
    test_result = 0.5982971518374901
    word = "00000000011111101001011101111011011010001011011000110111111100100011111101010011100111001110110100000011010001001101110010110001"
    assert test_result == float(longest_subsequence(word))

@pytest.mark.parametrize("generator, test_result", [
    ("c++", 0.4795001221869535),
    ("java", 0.7236736098317631)
])
def test_frequency_bitwise_test_with_read_json(generator, test_result):
    data = read_json("sequences.json")
    assert test_result == float(frequency_bitwise_test(data[generator]))

@pytest.mark.parametrize("generator, test_result", [
    ("c++", 0.39923839789091575),
    ("java", 0.2934250460371556)
])
def test_identical_bit_test_with_read_json(generator, test_result):
    data = read_json("sequences.json")
    assert test_result == float(identical_bit_test(data[generator]))

@pytest.mark.parametrize("generator, test_result", [
    ("c++", 0.5982971518374901),
    ("java", 0.5047565424394964)
])
def test_longest_subsequence_with_read_json(generator, test_result):
    data = read_json("sequences.json")
    assert test_result == float(longest_subsequence(data[generator]))
