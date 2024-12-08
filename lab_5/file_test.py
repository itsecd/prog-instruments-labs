import os
import csv
import shutil
import json
import mpmath

import pytest

from nist_test import frequency_bit_test, identical_consecutive_bits, load, longest_sequence_of_ones_test, longest_sequence
@pytest.mark.parametrize("input_text, expected_result", [
    ("111", 0.08326451666355043), ("0110110", 0.7054569861112734), ("010101", 1), ("101010", 1),
    
])
def test_frequency_bit_test(input_text, expected_result):
    result = frequency_bit_test(input_text)
    assert result == expected_result

def test_identical_consecutive_bits1():
    input_text = "101010"
    result = identical_consecutive_bits(input_text)
    assert result == 0.10247043485974938
    
def test_identical_consecutive_bits2():
    input_text = "111"
    result = identical_consecutive_bits(input_text)
    assert result == None

def test_identical_consecutive_bits3():
    input_text = "0110110"
    result = identical_consecutive_bits(input_text)
    assert result == 0.6592430036926307
    
def test_identical_consecutive_bits4():
    input_text = "010101"
    result = identical_consecutive_bits(input_text)
    assert result == 0.10247043485974938

def test_load():
    path = "test.json"
    with open(path, "w", encoding="utf-8") as f:
        data = {
    "cpp": "01",
    "java": "10"
                }
        json.dump(data, f)
    result = load(path)
    assert result == data
    
@pytest.mark.parametrize("input_text, expected_result", [
    ("11111111", 0.00916933636715453), ("00000000", 0.00410908458189325), ("01100110", 0.00908576547968168), ("11110000", 0.00916933636715453),
    
])
def test_longest_sequence_of_ones_test(input_text, expected_result):
    result = longest_sequence_of_ones_test(input_text)
    assert int(result) == int(expected_result)
        
@pytest.mark.parametrize("input_text, , input_text2, expected_result", [
    ("11111111", "1", 8), ("00000000", "1", 0), ("01100110", "1", 2), ("11110000", "1", 4),
    
])
def test_longest_sequence(input_text,input_text2, expected_result):
    result = longest_sequence(input_text,input_text2)
    assert result == expected_result