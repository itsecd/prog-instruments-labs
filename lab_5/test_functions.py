import pytest

from functions import (read_the_file, frequency_bitwise_test,
                       same_consecutive_bits_test, longest_sequence_of_units_in_a_block_test)


def test_reading_ability():
    with open("lab_5/sequences/cxx.txt", "w", encoding="utf-8") as file:
        file.write("11001000001111111010100100100110101011101101101110100111111001000000000101000110110000001001011000111110001010110001111000101110")
    text = read_the_file("lab_5/sequences/cxx.txt")
    assert text == "11001000001111111010100100100110101011101101101110100111111001000000000101000110110000001001011000111110001010110001111000101110"

def test_frequency_bitwise_counting_ability():
    frequency = 0.8596837951986662
    sequence = "11001000001111111010100100100110101011101101101110100111111001000000000101000110110000001001011000111110001010110001111000101110"
    assert frequency == frequency_bitwise_test(sequence)

@pytest.mark.parametrize("path, value", [("lab_5/sequences/java.txt", 0.7236736098317631)])
def test_frequency_bitwise_counting_ability_with_reading(path, value):
    sequence = read_the_file(path)
    frequency = frequency_bitwise_test(sequence)
    assert frequency == value

def test_same_consecutive_bits_ability():
    frequency = 0.5977098069466218
    sequence = "11001000001111111010100100100110101011101101101110100111111001000000000101000110110000001001011000111110001010110001111000101110"
    assert frequency == same_consecutive_bits_test(sequence)

@pytest.mark.parametrize("path, value", [("lab_5/sequences/java.txt", 0.2934250460371556)])
def test_same_consecutive_bits_ability_with_reading(path, value):
    sequence = read_the_file(path)
    frequency = same_consecutive_bits_test(sequence)
    assert frequency == value

def test_longest_sequence_of_units_in_a_block_checking():
    frequency = 0.16070143844190707
    sequence = "11001000001111111010100100100110101011101101101110100111111001000000000101000110110000001001011000111110001010110001111000101110"
    assert frequency == longest_sequence_of_units_in_a_block_test(sequence)

@pytest.mark.parametrize("path, value", [("lab_5/sequences/java.txt", 0.854898667197296)])
def test_longest_sequence_of_units_in_a_block_checking_with_reading(path, value):
    sequence = read_the_file(path)
    frequency = longest_sequence_of_units_in_a_block_test(sequence)
    assert frequency == value