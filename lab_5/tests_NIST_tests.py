import pytest
from NIST_tests import (frequency_test, 
                       consecutive_bits_test, longest_run_of_ones_test)
from r_files import read_txt_file


def test_reading_file():
    test_sequence = "10111001001100110000011100011100000011011101011000100111000101111011111111101100101010101110100110101111111111111010010111000001"
    file_path = "lab_5/generator_cc.txt"
    
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(test_sequence)
        
    text = read_txt_file(file_path)
    assert text == test_sequence


def test_frequency_test():
    expected_frequency = 0.11161176829829222
    sequence = "10111001001100110000011100011100000011011101011000100111000101111011111111101100101010101110100110101111111111111010010111000001"
    
    assert frequency_test(sequence) == expected_frequency


@pytest.mark.parametrize("path, expected_value", [
    ("lab_5/generator_cc.txt", 0.11161176829829222)
])
def test_frequency_test_with_reading(path, expected_value):
    sequence = read_txt_file(path)
    assert frequency_test(sequence) == expected_value


def test_consecutive_bits_test():
    expected_frequency = 0.39320937215824525
    sequence = "10111001001100110000011100011100000011011101011000100111000101111011111111101100101010101110100110101111111111111010010111000001"
    
    assert consecutive_bits_test(sequence) == expected_frequency


@pytest.mark.parametrize("path, expected_value", [
    ("lab_5/generator_cc.txt", 0.39320937215824525)
])
def test_consecutive_bits_test_with_reading(path, expected_value):
    sequence = read_txt_file(path)
    assert consecutive_bits_test(sequence) == expected_value


def test_longest_run_of_ones_test():
    expected_frequency = 0.9662110430474438
    sequence = "10111001001100110000011100011100000011011101011000100111000101111011111111101100101010101110100110101111111111111010010111000001"
    
    assert longest_run_of_ones_test(sequence) == expected_frequency


@pytest.mark.parametrize("path, expected_value", [
    ("lab_5/generator_cc.txt", 0.9662110430474438)
])
def test_longest_run_of_ones_test_with_reading(path, expected_value):
    sequence = read_txt_file(path)
    assert longest_run_of_ones_test(sequence) == expected_value
